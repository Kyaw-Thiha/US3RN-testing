from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as io
import os
import random
import time
import socket
import sys

from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from analytics.csv_logger import CSVLogger
from analytics.tee_logger import TeeLogger
from model import S3RNet
from data import get_patch_training_set, get_test_set
from torch.autograd import Variable
from psnr import MPSNR
from ssim import MSSIM
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


# Training settings
parser = argparse.ArgumentParser(description="PyTorch Super Res Example")
parser.add_argument("--upscale_factor", type=int, default=4, help="super resolution upscale factor")
parser.add_argument("--batchSize", type=int, default=8, help="training batch size")
parser.add_argument("--patch_size", type=int, default=64, help="training patch size")
parser.add_argument("--testBatchSize", type=int, default=1, help="testing batch size")
parser.add_argument("--ChDim", type=int, default=31, help="output channel number")
parser.add_argument("--alpha", type=float, default=0.2, help="alpha")
parser.add_argument("--nEpochs", type=int, default=0, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.01")
parser.add_argument("--threads", type=int, default=2, help="number of threads for data loader to use")
parser.add_argument("--seed", type=int, default=123, help="random seed to use. Default=123")
parser.add_argument("--save_folder", default="TrainedNet/", help="Directory to keep training outputs.")
parser.add_argument("--outputpath", type=str, default="result/", help="Path to output img")
parser.add_argument("--mode", default="test", help="Train or Test.")
opt = parser.parse_args()

print(opt)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed(opt.seed)

log_file_path = f"logs/train_logs/train_{opt.nEpochs}_.log"
sys.stdout = TeeLogger(log_file_path)


def load_train():
    print("===> Loading train datasets")
    train_set = get_patch_training_set(opt.upscale_factor, opt.patch_size)
    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=True,
        pin_memory=True,
    )
    return training_data_loader


def load_test():
    print("===> Loading test datasets")
    test_set = get_test_set(opt.upscale_factor)
    testing_data_loader = DataLoader(
        dataset=test_set,
        num_workers=opt.threads,
        batch_size=opt.testBatchSize,
        shuffle=False,
        pin_memory=True,
    )
    return testing_data_loader


print("===> Building model")


HSI_spectral_bands = 81
MSI_spectral_bands = 40
spatial_scale = 2
model = S3RNet(MSI_spectral_bands, HSI_spectral_bands, spatial_scale).cuda()
print("# network parameters: {}".format(sum(param.numel() for param in model.parameters())))
model = torch.nn.DataParallel(model).cuda()


optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = MultiStepLR(optimizer, milestones=[10, 30, 60, 120], gamma=0.5)

if opt.nEpochs != 0:
    load_dict = torch.load(opt.save_folder + "_epoch_{}.pth".format(opt.nEpochs))
    opt.lr = load_dict["lr"]
    epoch = load_dict["epoch"]
    model.load_state_dict(load_dict["param"])
    optimizer.load_state_dict(load_dict["adam"])

criterion = nn.L1Loss()


current_time = datetime.now().strftime("%b%d_%H-%M-%S")
CURRENT_DATETIME_HOSTNAME = "/" + current_time + "_" + socket.gethostname()
tb_logger = SummaryWriter(log_dir="./tb_logger/" + "unfolding2" + CURRENT_DATETIME_HOSTNAME)
current_step = 0


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")


mkdir(opt.save_folder)
mkdir(opt.outputpath)


batch_log_path = "analytics/batch_logs.csv"
epoch_log_path = "analytics/epoch_logs.csv"


def train(epoch, optimizer, scheduler):
    logger = CSVLogger(batch_log_path, epoch_log_path)
    training_data_loader = load_train()
    epoch_loss = 0
    global current_step

    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        # with torch.autograd.set_detect_anomaly(True):
        W, Y, Z, X = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
        optimizer.zero_grad()
        W = Variable(W).float()
        Y = Variable(Y).float()
        Z = Variable(Z).float()
        X = Variable(X).float()
        HX, HY, HZ, listX, listY, listZ = model(W)
        alpha = opt.alpha

        loss = criterion(HX, X) + alpha * criterion(HY, Y) + alpha * criterion(HZ, Z)
        for i in range(len(listX) - 1):
            loss = (
                loss
                + 0.5 * alpha * criterion(X, listX[i])
                + 0.5 * alpha * criterion(Y, listY[i])
                + 0.5 * alpha * criterion(Z, listZ[i])
            )
        epoch_loss += loss.item()

        tb_logger.add_scalar("total_loss", loss.item(), current_step)
        current_step += 1

        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            logger.log_batch(epoch, iteration, round(loss.item(), 4))
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    avg_loss = round(epoch_loss / len(training_data_loader), 4)
    logger.log_epoch(epoch, avg_loss)
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss))
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"===> Timestamp: [{timestamp}]")
    return avg_loss


def test():
    testing_data_loader = load_test()
    avg_psnr = 0
    avg_ssim = 0
    avg_time = 0
    model.eval()

    with torch.no_grad():
        for batch in testing_data_loader:
            W, X = batch[0].cuda(), batch[1].cuda()
            W = Variable(W).float()
            X = Variable(X).float()
            torch.cuda.synchronize()
            start_time = time.time()

            HX, HY, HZ, listX, listY, listZ = model(W)
            torch.cuda.synchronize()
            end_time = time.time()

            X = torch.squeeze(X).permute(1, 2, 0).cpu().numpy()
            HX = torch.squeeze(HX).permute(1, 2, 0).cpu().numpy()

            print(f"X min/max: {X.min()}, {X.max()}")
            print(f"HX min/max: {HX.min()}, {HX.max()}")
            psnr = MPSNR(HX, X)
            ssim = MSSIM(HX, X)
            im_name = batch[2][0]
            print(f"Analyzing {im_name}")

            avg_time += end_time - start_time
            (path, filename) = os.path.split(im_name)
            io.savemat(opt.outputpath + filename, {"HX": HX})
            avg_psnr += psnr
            avg_ssim += ssim
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    print("===> Avg. SSIM: {:.4f}".format(avg_ssim / len(testing_data_loader)))
    print("===> Avg. time: {:.4f} s".format(avg_time / len(testing_data_loader)))
    return avg_psnr / len(testing_data_loader)


def checkpoint(epoch):
    model_out_path = opt.save_folder + "_epoch_{}.pth".format(epoch)
    if epoch % 5 == 0:
        save_dict = dict(
            lr=optimizer.state_dict()["param_groups"][0]["lr"],
            param=model.state_dict(),
            adam=optimizer.state_dict(),
            epoch=epoch,
        )
        torch.save(save_dict, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))


if opt.mode == "train":
    for epoch in range(opt.nEpochs + 1, 161):
        avg_loss = train(epoch, optimizer, scheduler)
        checkpoint(epoch)
        scheduler.step()
else:
    test()
