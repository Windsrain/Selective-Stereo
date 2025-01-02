import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.update import SpatialAttentionExtractor, ChannelAttentionEnhancement, BasicSelectiveMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder
from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from core.utils.utils import coords_grid

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        
        self.hidden_dim = hdim = self.args.hidden_dim
        self.context_dim = cdim = self.args.hidden_dim

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)  
        self.cnet = MultiBasicEncoder(norm_fn='batch', downsample=args.n_downsample)
        self.update_block = BasicSelectiveMultiUpdateBlock(self.args, self.args.hidden_dim)
        self.sam = SpatialAttentionExtractor()
        self.cam = ChannelAttentionEnhancement(self.hidden_dim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_disp(self, img):
        """ Disp is represented as difference between two coordinate grids Disp = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        # disparity computed as difference: disp = coords1 - coords0
        return coords0, coords1
    
    def upsample_disp(self, disp, mask):
        """ Upsample disp field [H/4, W/4, 1] -> [H, W, 1] using convex combination """
        N, D, H, W = disp.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(factor * disp, [3,3], padding=1)
        up_disp = up_disp.view(N, D, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, D, factor * H, factor * W)

    def forward(self, image1, image2, iters=22, disp_init=None, test_mode=False):
        """ Estimate disparity between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            fmap1 = self.fnet(image1)
            fmap2 = self.fnet(image2)
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        if self.args.corr_implementation == "reg": # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "alt": # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "reg_cuda": # Faster version of reg
            corr_block = CorrBlockFast1D
        elif self.args.corr_implementation == "alt_cuda": # Faster version of alt
            corr_block = AlternateCorrBlock

        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        # run the context network
        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            cnet = self.cnet(image1)
            net = [torch.tanh(x[0]) for x in cnet]
            inp = [torch.relu(x[1]) for x in cnet]
            inp = [self.cam(x) * x for x in inp]
            att = [self.sam(x) for x in inp]

        coords0, coords1 = self.initialize_disp(net[0])

        if disp_init is not None:
            coords1 = coords1 + disp_init

        disp_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            disp = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
                net, up_mask, delta_disp = self.update_block(net, inp, corr, disp, att)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_disp

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters - 1:
                continue
            
            # upsample predictions
            disp_up = self.upsample_disp(coords1 - coords0, up_mask)
            disp_predictions.append(disp_up)

        if test_mode:
            return coords1 - coords0, disp_up
            
        return disp_predictions