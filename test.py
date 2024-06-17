import os, time, scipy.io

import numpy as np
import glob

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import skimage.measure as skm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import exposure

from PIL import Image

from Denoise import Denoise
import argparse

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

input_dir = "/content/drive/MyDrive/dataset/testsplit/low/"
gt_dir = "/content/drive/MyDrive/dataset/testsplit/high/"
m_path = "/content/drive/MyDrive/img_code/Final/denoise-ps-300-b-32/"
m_name = "denoise_e0020.pth"
result_dir = "/content/drive/MyDrive/img_code/result/"

# get test IDs
test_fns = glob.glob(gt_dir + "/*.png")
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

model = Denoise(args).to(device)
model.load_state_dict(torch.load(m_path + m_name, map_location=device))
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

psnr = []
ssim = []
cnt = 0
with torch.no_grad():
    for test_id in test_ids:
        # test the first image in each sequence
        in_files = glob.glob(input_dir + test_id + ".png")
        for k in range(len(in_files)):
            in_path = in_files[k]
            gt_files = glob.glob(gt_dir + test_id + ".png")
            gt_path = gt_files[0]

            input_full = load_image(in_path)
            gt_full = load_image(gt_path)

            input_full = np.expand_dims(input_full, axis=0)
            gt_full = np.expand_dims(gt_full, axis=0)

            input_full = np.minimum(input_full, 1.0)

            in_img = torch.from_numpy(input_full).permute(0, 3, 1, 2).to(device)
            st = time.time()
            cnt += 1
            out_img = model(in_img)
            print("%d\tTime: %.3f" % (cnt, time.time() - st))

            output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
            output = np.minimum(np.maximum(output, 0), 1)

            output = output[0, :, :, :]
            gt_full = gt_full[0, :, :, :]
            origin_full = input_full[0, :, :, :]
            input_full = (
                input_full * np.mean(gt_full) / np.mean(input_full)
            )  # scale the input image to the same mean of the ground truth

            psnr.append(compare_psnr(gt_full[:, :, :], output[:, :, :]))
            ssim.append(
                compare_ssim(gt_full[:, :, :], output[:, :, :], multichannel=True)
            )
            print("psnr: ", psnr[-1], "ssim: ", ssim[-1])

            if not os.path.isdir(result_dir):
                os.makedirs(result_dir)

            plt.imsave(result_dir + "%05d_00_ori.png" % int(test_id), origin_full)
            plt.imsave(result_dir + "%05d_00_out.png" % int(test_id), output)
            plt.imsave(result_dir + "%05d_00_gt.png" % int(test_id), gt_full)
print("\n\n---------------------------------")
print("mean psnr: ", np.mean(psnr))
print("mean ssim: ", np.mean(ssim))
print("done")
