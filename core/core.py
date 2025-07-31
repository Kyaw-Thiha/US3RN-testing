from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from model import S3RNet


@dataclass
class Options:
    upscale_factor: int
    batchSize: int
    patch_size: int
    testBatchSize: int
    ChDim: int
    alpha: float
    nEpochs: int
    endEpochs: int
    lr: float
    threads: int
    seed: int
    save_folder: str
    outputpath: str
    mode: str


def build_model(msi_spectral_bands: int, hsi_spectral_bands: int, opt: Options):
    """
    Build and initialize the model, optimizer, scheduler, and loss function.

    Parameters
    ----------
    msi_spectral_bands : int
        Number of spectral bands in the MSI input.
    hsi_spectral_bands : int
        Number of spectral bands in the HSI ground truth.
    opt : Options
        Configuration object containing training hyperparameters.

    Returns
    -------
    model : nn.Module
        The initialized and wrapped model.
    optimizer : optim.Optimizer
        The optimizer for training.
    scheduler : optim.lr_scheduler.LRScheduler
        The learning rate scheduler.
    criterion : nn.Module
        The loss function (criterion).
    """
    print("===> Building model")

    model = S3RNet(msi_spectral_bands, hsi_spectral_bands, opt.upscale_factor).cuda()
    print("# network parameters: {}".format(sum(param.numel() for param in model.parameters())))
    model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=[10, 30, 60, 120], gamma=0.5)
    criterion = nn.L1Loss()

    print("===> Model successfully built")

    return model, optimizer, scheduler, criterion


def load_model(model: torch.nn.Module, optimizer: optim.Optimizer, nEpochs: int, opt: Options):
    """
    Load a saved model and optimizer state from a checkpoint file.

    Parameters
    ----------
    model : nn.Module
        The model into which the parameters are to be loaded.
    optimizer : optim.Optimizer
        The optimizer to load the state for.
    nEpochs : int
        Epoch number of the checkpoint to load.
    opt : Options
        Configuration object with save folder path.

    Returns
    -------
    model : nn.Module
        The model with loaded parameters.
    optimizer : optim.Optimizer
        The optimizer with loaded state.
    """
    if nEpochs != 0:
        print("===> Loading existing model")

        load_dict = torch.load(opt.save_folder + "_epoch_{}.pth".format(nEpochs))
        opt.lr = load_dict["lr"]
        epoch = load_dict["epoch"]
        model.load_state_dict(load_dict["param"])
        optimizer.load_state_dict(load_dict["adam"])

    return model, optimizer
