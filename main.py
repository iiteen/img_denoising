# test.py
import os, time

import numpy as np
import glob

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from PIL import Image

from Denoise import Denoise
import argparse
import logging

logging.basicConfig(level=logging.INFO)

# Argument for Denoise
parser = argparse.ArgumentParser(description="Denoise")
parser.add_argument(
    "--n_resblocks", type=int, default=32, help="number of residual blocks"
)
parser.add_argument("--n_feats", type=int, default=64, help="number of feature maps")
parser.add_argument("--res_scale", type=float, default=1, help="residual scaling")
parser.add_argument("--scale", type=str, default=1, help="super resolution scale")
parser.add_argument("--patch_size", type=int, default=300, help="output patch size")
parser.add_argument(
    "--n_colors", type=int, default=3, help="number of input color channels to use"
)
parser.add_argument(
    "--o_colors", type=int, default=3, help="number of output color channels to use"
)
args = parser.parse_args()

torch.manual_seed(0)

input_dir = "./test/low/"
m_path = "./Model/denoise-lerelu-ps-300-b-31/"
m_name = "denoise_e0035.pth"
result_dir = "./test/predicted/"

# get test IDs
test_fns = glob.glob(input_dir + "/*.png")
test_ids = []
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append(test_fn[0:-4])

ps = args.patch_size  # patch size for training


def load_image(path):
    img = Image.open(path)
    img = np.array(img).astype(np.float32) / 255.0
    return img


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info("reached 1")
model = Denoise(args).to(device)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.load_state_dict(torch.load(m_path + m_name, map_location=device))
model.to(device)

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)


cnt = 0
with torch.no_grad():
    logging.info("loop started")
    for test_id in test_ids:
        # test the first image in each sequence
        in_files = glob.glob(input_dir + test_id + ".png")
        for k in range(len(in_files)):
            in_path = in_files[k]

            input_full = load_image(in_path)

            input_full = np.expand_dims(input_full, axis=0)

            input_full = np.minimum(input_full, 1.0)

            in_img = torch.from_numpy(input_full).permute(0, 3, 1, 2).to(device)
            st = time.time()
            cnt += 1
            out_img = model(in_img)
            print("%d\tTime: %.3f" % (cnt, time.time() - st))

            output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
            output = np.minimum(np.maximum(output, 0), 1)

            output = output[0, :, :, :]
            origin_full = input_full[0, :, :, :]

            # print("psnr: %.4f" % psnr[-1])
            if not os.path.isdir(result_dir):
                os.makedirs(result_dir)

            plt.imsave(result_dir + "%05d_00_out.png" % int(test_id), output)

print("\n\n---------------------------------")
print(f"output files are stored in {result_dir}")
