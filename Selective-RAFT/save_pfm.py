import sys
sys.path.append('core')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.raft import RAFT
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt

DEVICE = 'cuda'

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(RAFT(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            starter.record()
            _, disp = model(image1, image2, iters=args.valid_iters, test_mode=True)

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)

            disp = -disp.cpu().numpy()
            disp = padder.unpad(disp).squeeze()
            file_stem = imfile1.split('/')[-2]
            filedir = Path(os.path.join(output_directory, file_stem))
            filedir.mkdir(exist_ok=True)
            filename = os.path.join(output_directory, file_stem, 'disp0Selective-RAFT.pfm')
            with open(filename, 'wb') as f:
                H, W = disp.shape
                headers = ["Pf\n", f"{W} {H}\n", "-1\n"]
                for header in headers:
                    f.write(str.encode(header))
                array = np.flip(disp, axis=0).astype(np.float32)
                f.write(array.tobytes())

            filename = os.path.join(output_directory, file_stem, 'timeSelective-RAFT.txt')
            with open(filename, 'wb') as f:
                time = '%.2f' % (curr_time / 1000)
                f.write(str.encode(time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', default=None, help="restore checkpoint")
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default=None)
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default=None)
    parser.add_argument('--output_directory', help="directory to save output", default=None)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument("--precision_dtype",default="float16",choices=["float16", "bfloat16", "float32"],help="Choose precision type: float16 or bfloat16 or float32")
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=128, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)
