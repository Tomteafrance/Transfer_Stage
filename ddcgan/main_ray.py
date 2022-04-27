import argparse
import logging
import time

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image

from networks import Generator, Discriminator
from utils import get_data_loader, generate_images, save_gif

import ray.train as train
from ray.train.trainer import Trainer
from ray.train.callbacks import JsonLoggerCallback


def train_func(config: Dict):
    batch_size = config["batch_size"]
    dlr = config["d-lr"]
    glr = config["g-lr"]
    epochs = config["epochs"]

    num_test_samples = config['num_test_samples']
    nc = config['nc']
    nz = config['nz']
    ngf = config['ngf']
    ndf = config['ndf']

    num_epochs = config['num_epochs']
    output_path =config['output_path']
    use_fixed = config['use_fixed']

    # Gather MNIST Dataset    
    train_loader = get_data_loader(batch_size)

    # Ray Train DataLoader 
    train_loader = train.torch.prepare_data_loader(train_loader)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)

    # Define Discriminator and Generator architectures
    netG = Generator(nc, nz, ngf).to(device)
    netD = Discriminator(nc, ndf).to(device)

    # Ray Train Prepare Model
    netG = train.torch.prepare_model(netG)
    netD = train.torch.prepare_model(netD)

    # loss function
    criterion = nn.BCELoss()

    # optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=dlr)
    optimizerG = optim.Adam(netG.parameters(), lr=glr)
    
    # initialize other variables
    real_label = 1
    fake_label = 0
    num_batches = len(train_loader)
    fixed_noise = torch.randn(num_test_samples, 100, 1, 1, device=device)

    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(train_loader):
            bs = real_images.shape[0]
            ##############################
            #   Training discriminator   #
            ##############################

            netD.zero_grad()
            real_images = real_images.to(device)
            label = torch.full((bs,), real_label, device=device)

            output = netD(real_images)
            
            #print('type output',output.dtype)
            #print('type label',label.dtype)
            label = label.type(torch.float)

            lossD_real = criterion(output, label)
            lossD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(bs, nz, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(fake_label)
            output = netD(fake_images.detach())
            lossD_fake = criterion(output, label)
            lossD_fake.backward()
            D_G_z1 = output.mean().item()
            lossD = lossD_real + lossD_fake
            optimizerD.step()

            ##########################
            #   Training generator   #
            ##########################

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake_images)
            lossG = criterion(output, label)
            lossG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if (i+1)%100 == 0:
                print('Epoch [{}/{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, Discriminator - D(G(x)): {:.2f}, Generator - D(G(x)): {:.2f}'.format(epoch+1, num_epochs, 
                                                            i+1, num_batches, lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))
        netG.eval()
        generate_images(epoch, output_path, fixed_noise, num_test_samples, netG, device, use_fixed=use_fixed)
        netG.train()

def train_mnist(num_workers=2,use_gpu=False):
    trainer = Trainer(backend="torch",num_workers=num_workers,use_gpu=use_gpu)
    trainer.start()
    result = trainer.run(
        train_func=train_func,
        config={"d-lr": 0.0002,"g-lr": 0.0002, "batch_size": 64, "epochs": 4, "nz":100,"ndf":32, "ngf":32,
        "nc":1,"num_test_samples":16,"num_epochs":5,"output_path":'./results/',"use_fixed":True},
        callbacks=[JsonLoggerCallback()],
    )
    trainer.shutdown()
    print(f"Loss results: {result}")


if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser(description='DCGANS MNIST')
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--ndf', type=int, default=32, help='Number of features to be used in Discriminator network')
    parser.add_argument('--ngf', type=int, default=32, help='Number of features to be used in Generator network')
    parser.add_argument('--nz', type=int, default=100, help='Size of the noise')
    parser.add_argument('--d-lr', type=float, default=0.0002, help='Learning rate for the discriminator')
    parser.add_argument('--g-lr', type=float, default=0.0002, help='Learning rate for the generator')
    parser.add_argument('--nc', type=int, default=1, help='Number of input channels. Ex: for grayscale images: 1 and RGB images: 3 ')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--num-test-samples', type=int, default=16, help='Number of samples to visualize')
    parser.add_argument('--output-path', type=str, default='./results/', help='Path to save the images')
    parser.add_argument('--fps', type=int, default=5, help='frames-per-second value for the gif')
    parser.add_argument('--use-fixed', action='store_true', help='Boolean to use fixed noise or not')

    opt = parser.parse_args()
    print(opt)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address", required=False, type=str, help="the address to use for Ray"
    )
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=2,
        help="Sets number of workers for training.",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", default=False, help="Enables GPU training"
    )

    args, _ = parser.parse_known_args()

    import ray

    ray.init(address=args.address)
    train_mnist(num_workers=args.num_workers, use_gpu=args.use_gpu)

