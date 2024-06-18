import os, time
import numpy as np
from PIL import Image
import glob

import torch
import torch.nn as nn
import torch.optim as optim

from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from Denoise import Denoise
import argparse
import logging

logging.basicConfig(level=logging.INFO)

# Argument for denoise
parser = argparse.ArgumentParser(description="denoise")
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

input_dir = "../dataset/trainsplit/low/"
gt_dir = "../dataset/trainsplit/high/"
model_dir = "./Model/"
test_name = (
    "denoise-lerelu-ps-" + str(args.patch_size) + "-b-" + str(args.n_resblocks) + "/"
)


save_freq = 5
total_epoch = 501


# get train and test IDs
train_fns = glob.glob(gt_dir + "*.png")
train_ids = []
for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i])
    train_ids.append(train_fn[0:-4])

ps = args.patch_size  # patch size for training


def load_image(path):
    img = Image.open(path)
    img = np.array(img).astype(np.float32) / 255.0

    return img


def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()


# Load the ground truth and input images
gt_images = [None] * 6000
input_images = [None] * len(train_ids)

g_loss = np.zeros((5000, 1))

allfolders = glob.glob(model_dir + test_name + "denoise_e*.pth")
lastepoch = 0
latest_model = None

# Find the latest saved model
if allfolders:
    allfolders.sort()
    latest_model = allfolders[-1]
    lastepoch = int(latest_model.split("e")[-1].split(".")[0]) + 1

learning_rate = 1e-4
model = Denoise(args)  # Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

opt = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=10, verbose=True
)  # Learning rate scheduler

# Load the latest model if available
if latest_model:
    checkpoint = torch.load(latest_model, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Resuming training from epoch {lastepoch}")


for epoch in range(lastepoch, total_epoch):
    psnr = []
    ssim = []
    if os.path.isdir("result/%04d" % epoch):
        continue
    cnt = 0
    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + train_id + ".png")
        in_path = in_files[np.random.randint(0, len(in_files))]
        gt_files = glob.glob(gt_dir + train_id + ".png")
        gt_path = gt_files[0]

        st = time.time()
        cnt += 1

        if input_images[ind] is None:
            in_image = load_image(in_path)
            gt_image = load_image(gt_path)

            input_images[ind] = np.expand_dims(in_image, axis=0)
            gt_images[ind] = np.expand_dims(gt_image, axis=0)

        # crop
        H = input_images[ind].shape[1]
        W = input_images[ind].shape[2]

        xx = np.random.randint(0, W - ps * args.scale)
        yy = np.random.randint(0, H - ps * args.scale)
        input_patch = input_images[ind][:, yy : yy + ps, xx : xx + ps, :]

        gt_patch = gt_images[ind][
            :, yy : yy + ps * args.scale, xx : xx + ps * args.scale, :
        ]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        in_img = torch.from_numpy(input_patch).permute(0, 3, 1, 2).to(device)
        gt_img = torch.from_numpy(gt_patch).permute(0, 3, 1, 2).to(device)

        model.zero_grad()
        out_img = model(in_img)

        loss = reduce_mean(out_img, gt_img)
        loss.backward()

        out_img = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
        gt_img = gt_img.permute(0, 2, 3, 1).cpu().data.numpy()

        opt.step()
        g_loss[ind] = loss.detach().cpu().numpy()

        psnr.append(compare_psnr(gt_img[:, :, :], out_img[:, :, :]))

    logging.info("---------------------------------")
    logging.info("%d mean psnr: %.4f", epoch, np.mean(psnr))

    scheduler.step(
        np.mean(g_loss[np.where(g_loss)])
    )  # Update learning rate based on the scheduler

    if epoch % save_freq == 0:
        if not os.path.isdir(model_dir + test_name):
            os.makedirs(model_dir + test_name)
        torch.save(
            model.state_dict(),
            model_dir + test_name + "denoise_e%04d.pth" % epoch,
        )
