#!/usr/bin/env python
import argparse
import os
import pprint
import time
import pickle
import random
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils import spectral_norm
from torch.cuda.amp import autocast, GradScaler
import wandb
import matplotlib.pyplot as plt

torch.multiprocessing.set_sharing_strategy("file_system")

sys.path.append("..")
from src.training_manager import TrainingManager, get_current_time

# Additional Loss Functions
class MultiScaleSpectrogramLoss(nn.Module):
    def __init__(self, scales):
        super().__init__()
        self.scales = scales

    def forward(self, input, target):
        loss = 0.0
        for scale in self.scales:
            input_scaled = F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=False)
            target_scaled = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
            loss += F.l1_loss(input_scaled, target_scaled)
        return loss

def make_inf_iterator(data_iterator):
    while True:
        for data in data_iterator:
            yield data

def check_for_nans(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN or Inf detected in {name}")
        print(tensor)
        exit(1)

class VocDataset(Dataset):
    def __init__(self, ids, path):
        self.metadata = ids
        self.path = path

    def __getitem__(self, index):
        id = self.metadata[index]
        voc_fp = os.path.join(self.path, f"{id}.npy")
        voc = np.load(voc_fp)
        return voc

    def __len__(self):
        return len(self.metadata)

def get_voc_datasets(path, feat_type, batch_size, va_samples):
    dataset_fp = os.path.join(path, "dataset.pkl")
    in_dir = os.path.join(path, feat_type)
    with open(dataset_fp, "rb") as f:
        dataset = pickle.load(f)
    dataset_ids = [x[0] for x in dataset]
    random.seed(1234)
    random.shuffle(dataset_ids)
    va_ids = dataset_ids[-va_samples:]
    tr_ids = dataset_ids[:-va_samples]
    tr_dataset = VocDataset(tr_ids, in_dir)
    va_dataset = VocDataset(va_ids, in_dir)
    iterator_tr = DataLoader(
        tr_dataset,
        batch_size=batch_size,
        num_workers=5,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    iterator_va = DataLoader(
        va_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )
    return iterator_tr, len(tr_dataset), iterator_va, len(va_dataset)

def validate(epoch, step, netG, netD, netE, iterator_va, mean, std, z_dim, z_total_scale_factor, scales):
    sum_losses_va = OrderedDict([(loss_name, 0) for loss_name in ["G", "D", "RealD", "FakeD", "Convergence", "NoiseRecon", "RealRecon"]])
    count_all_va = 0
    netG.eval()
    netD.eval()
    netE.eval()
    num_batches_va = len(iterator_va)
    with torch.no_grad():
        for i_batch, batch in enumerate(iterator_va):
            voc = batch.cuda()
            check_for_nans(voc, "validation voc before normalization")
            voc = (voc - mean) / std
            check_for_nans(voc, "validation voc after normalization")
            bs, _, nf = voc.size()
            z = torch.zeros((bs, z_dim, int(np.ceil(nf / z_total_scale_factor)))).normal_(0, 1).float().cuda()
            fake_voc = netG(z)
            check_for_nans(fake_voc, "validation fake_voc")
            z_fake = netE(fake_voc)
            z_real = netE(voc)
            gloss = torch.mean(torch.abs(netD(fake_voc) - fake_voc))
            check_for_nans(gloss, "validation gloss")
            noise_rloss = torch.mean(torch.abs(z_fake - z))
            check_for_nans(noise_rloss, "validation noise_rloss")
            real_rloss = torch.mean(torch.abs(netG(z_real)[..., :nf] - voc[..., :nf]))
            check_for_nans(real_rloss, "validation real_rloss")
            real_dloss = torch.mean(torch.abs(netD(voc) - voc))
            check_for_nans(real_dloss, "validation real_dloss")
            fake_dloss = torch.mean(torch.abs(netD(fake_voc.detach()) - fake_voc.detach()))
            check_for_nans(fake_dloss, "validation fake_dloss")
            dloss = real_dloss - k * fake_dloss
            check_for_nans(dloss, "validation dloss")
            _, convergence = recorder(real_dloss, fake_dloss, update_k=False)
            check_for_nans(convergence, "validation convergence")
            losses = OrderedDict(
                [
                    ("G", gloss),
                    ("D", dloss),
                    ("RealD", real_dloss),
                    ("FakeD", fake_dloss),
                    ("Convergence", convergence),
                    ("NoiseRecon", noise_rloss),
                    ("RealRecon", real_rloss),
                ]
            )
            count_all_va += bs
            losses_va = OrderedDict([(loss_name, lo.item()) for loss_name, lo in losses.items()])
            for loss_name, lo in losses_va.items():
                sum_losses_va[loss_name] += lo * bs
            if i_batch % 10 == 0:
                print("{}/{}".format(i_batch, num_batches_va))
            wandb.log({"loss/eval": losses, "epoch": epoch}, step=step)
    mean_losses_va = OrderedDict([(loss_name, slo / count_all_va) for loss_name, slo in sum_losses_va.items()])
    return mean_losses_va

class BEGANRecorder(nn.Module):
    def __init__(self, lambda_k, init_k, gamma):
        super().__init__()
        self.lambda_k = lambda_k
        self.init_k = init_k
        self.gamma = gamma
        self.k = nn.Parameter(torch.tensor(init_k))

    def forward(self, real_dloss, fake_dloss, update_k=False):
        diff = self.gamma * real_dloss - fake_dloss
        convergence = real_dloss + torch.abs(diff)
        if update_k:
            self.k.data = torch.clamp(self.k + self.lambda_k * diff, 0, 1).data
        return self.k.item(), convergence

class RCBlock(nn.Module):
    def __init__(self, feat_dim, ks, dilation, num_groups):
        super().__init__()
        ksm1 = ks - 1
        mfd = feat_dim
        di = dilation
        self.num_groups = num_groups
        self.relu = nn.LeakyReLU()
        self.rec = nn.GRU(mfd, mfd, num_layers=1, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(mfd, mfd, ks, 1, ksm1 * di // 2, dilation=di, groups=num_groups)
        self.gn = nn.GroupNorm(num_groups, mfd)

    def init_hidden(self, batch_size, hidden_size):
        num_layers = 1
        num_directions = 2
        hidden = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
        hidden.normal_(0, 1)
        return hidden

    def forward(self, x):
        bs, mfd, nf = x.size()
        hidden = self.init_hidden(bs, mfd).to(x.device)
        r = x.transpose(1, 2)
        r, _ = self.rec(r, hidden)
        r = r.transpose(1, 2).view(bs, 2, mfd, nf).sum(1)
        c = self.relu(self.gn(self.conv(r)))
        x = x + r + c
        return x

class BodyGBlock(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim, num_groups):
        super().__init__()
        ks = 3  # kernel size
        mfd = middle_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mfd = mfd
        self.num_groups = num_groups
        block = [
            nn.Conv1d(input_dim, mfd, 3, 1, 1),
            nn.GroupNorm(num_groups, mfd),
            nn.LeakyReLU(),
            RCBlock(mfd, ks, dilation=1, num_groups=num_groups),
            nn.Conv1d(mfd, output_dim, 3, 1, 1),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        x = self.block(x)
        return x

class NetG(nn.Module):
    def __init__(self, feat_dim, z_dim, z_scale_factors):
        super().__init__()
        mfd = 512
        num_groups = 4
        self.num_groups = num_groups
        self.mfd = mfd
        self.feat_dim = feat_dim
        self.z_dim = z_dim
        self.z_scale_factors = z_scale_factors
        self.block0 = BodyGBlock(z_dim, mfd, mfd, num_groups)
        self.head0 = nn.Conv1d(mfd, feat_dim, 3, 1, 1)
        blocks = []
        heads = []
        for scale_factor in z_scale_factors:
            block = BodyGBlock(mfd, mfd, mfd, num_groups)
            blocks.append(block)
            head = nn.Conv1d(mfd, feat_dim, 3, 1, 1)
            heads.append(head)
        self.blocks = nn.ModuleList(blocks)
        self.heads = nn.ModuleList(heads)

    def forward(self, z):
        z_scale_factors = self.z_scale_factors
        x_body = self.block0(z)
        x_head = self.head0(x_body)
        for ii, (block, head, scale_factor) in enumerate(zip(self.blocks, self.heads, z_scale_factors)):
            x_body = F.interpolate(x_body, scale_factor=scale_factor, mode="nearest")
            x_head = F.interpolate(x_head, scale_factor=scale_factor, mode="nearest")
            x_body = x_body + block(x_body)
            x_head = x_head + head(x_body)
        return x_head

class BNSNConv2dDBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, frequency_stride, time_dilation):
        super().__init__()
        ks = kernel_size
        ksm1d2 = (ks - 1) // 2
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.time_dilation = time_dilation
        self.frequency_stride = frequency_stride
        block = [
            spectral_norm(
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    ks,
                    (frequency_stride, 1),
                    (1, time_dilation * ksm1d2),
                    dilation=(1, time_dilation),
                )
            ),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        x = self.block(x)
        return x

class BNSNConv1dDBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation):
        super().__init__()
        ks = kernel_size
        ksm1d2 = (ks - 1) // 2
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        block = [
            spectral_norm(
                nn.Conv1d(input_dim, output_dim, ks, 1, dilation * ksm1d2, dilation=dilation)
            ),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        x = self.block(x)
        return x

class StridedBNSNConv1dDBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride):
        super().__init__()
        ks = kernel_size
        ksm1d2 = (ks - 1) // 2
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        block = [
            spectral_norm(nn.Conv1d(input_dim, output_dim, ks, stride, ksm1d2)),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        x = self.block(x)
        return x

class NetD(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        ks = 3  # kernel size
        mfd = 512
        self.mfd = mfd
        self.input_size = input_size
        blocks2d = [
            BNSNConv2dDBlock(1, 4, ks, 2, 2),
            BNSNConv2dDBlock(4, 16, ks, 2, 4),
            BNSNConv2dDBlock(16, 64, ks, 2, 8),
        ]
        blocks1d = [
            BNSNConv1dDBlock(64 * 10, mfd, 3, 1),
            BNSNConv1dDBlock(mfd, mfd, ks, 16),
            BNSNConv1dDBlock(mfd, mfd, ks, 32),
            BNSNConv1dDBlock(mfd, mfd, ks, 64),
            BNSNConv1dDBlock(mfd, mfd, ks, 128),
        ]
        self.body2d = nn.Sequential(*blocks2d)
        self.body1d = nn.Sequential(*blocks1d)
        self.head = spectral_norm(nn.Conv1d(mfd, input_size, 3, 1, 1))

    def forward(self, x):
        bs, fd, nf = x.size()
        x = x.unsqueeze(1)
        x = self.body2d(x)
        x = x.view(bs, -1, x.size(3))
        x = self.body1d(x)
        out = self.head(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_size, z_dim, z_scale_factors):
        super().__init__()
        ks = 3  # kernel size
        mfd = 512
        self.mfd = mfd
        self.input_size = input_size
        self.z_dim = z_dim
        self.z_scale_factors = z_scale_factors
        blocks2d = [
            BNSNConv2dDBlock(1, 4, ks, 2, 2),
            BNSNConv2dDBlock(4, 16, ks, 2, 4),
            BNSNConv2dDBlock(16, 64, ks, 2, 8),
        ]
        blocks1d = [BNSNConv1dDBlock(64 * 10, mfd, 3, 1)]
        for sf in z_scale_factors:
            blocks1d.append(StridedBNSNConv1dDBlock(mfd, mfd, ks, sf))
        self.body2d = nn.Sequential(*blocks2d)
        self.body1d = nn.Sequential(*blocks1d)
        self.head = spectral_norm(nn.Conv1d(mfd, z_dim, 3, 1, 1))

    def forward(self, x):
        bs, fd, nf = x.size()
        x = x.unsqueeze(1)
        x = self.body2d(x)
        x = x.view(bs, -1, x.size(3))
        x = self.body1d(x)
        out = self.head(x)
        return out

# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def step(self, current_loss):
        if self.best_loss is None or current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            return False  # Do not stop
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
            return False

# Placeholder lists to store metrics
epochs = []
training_convergence = []
validation_convergence = []

def update_and_plot_metrics(epoch, mean_losses_tr, mean_losses_va):
    epochs.append(epoch)
    training_convergence.append(mean_losses_tr["Convergence"])
    validation_convergence.append(mean_losses_va["Convergence"])

    # Plot metrics
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, training_convergence, label='Training Convergence', marker='o')
    plt.plot(epochs, validation_convergence, label='Validation Convergence', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Convergence Loss')
    plt.legend()
    plt.title('Training vs Validation Convergence')
    
    # Save plot to file
    plot_path = 'metrics.png'
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved as {plot_path}. Please view it manually.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str)
    parser.add_argument("--batches-per-epoch", type=int, default=500)
    args = parser.parse_args()
    model_id = args.model_id
    if model_id is None:
        model_id = get_current_time()
        resume_training = False
    else:
        resume_training = True
    script_path = os.path.realpath(__file__)
    print(model_id)
    print(script_path)
    data_dir = "./training_data/exp_data"
    feat_dim = 80
    z_dim = 20
    z_scale_factors = [2, 2, 2, 2]
    z_total_scale_factor = np.prod(z_scale_factors)
    num_va = 200
    feat_type = "mel"
    gamma = 1.0
    lambda_k = 0.01
    init_k = 0.0
    eval_metrics = [("Convergence", "lower_better")]
    init_lr = 0.0001
    num_epochs = 200
    batches_per_epoch = args.batches_per_epoch
    lambda_cycle = 1
    max_grad_norm = 3
    save_rate = 20
    batch_size = 5
    config_keys = [
        "script_path",
        "data_dir",
        "feat_dim",
        "z_dim",
        "z_scale_factors",
        "z_total_scale_factor",
        "num_va",
        "feat_type",
        "gamma",
        "lambda_k",
        "init_k",
        "init_lr",
        "num_epochs",
        "lambda_cycle",
        "max_grad_norm",
        "save_rate",
        "batch_size",
    ]
    locs = locals()
    config = {**{k: locs[k] for k in config_keys}, **args.__dict__}
    pprint.pprint(config)
    wandb.init(entity="demiurge", project="unagan", config=config)
    base_out_dir = wandb.run.dir
    save_dir = os.path.join(base_out_dir, "save")
    output_dir = os.path.join(save_dir, model_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    iterator_tr, num_tr, iterator_va, _ = get_voc_datasets(data_dir, feat_type, batch_size, num_va)
    print("tr: {}, va: {}".format(num_tr, num_va))
    inf_iterator_tr = make_inf_iterator(iterator_tr)
    mean_fp = os.path.join(data_dir, f"mean.{feat_type}.npy")
    mean = torch.from_numpy(np.load(mean_fp)).float().cuda().view(1, feat_dim, 1)
    std_fp = os.path.join(data_dir, f"std.{feat_type}.npy")
    std = torch.from_numpy(np.load(std_fp)).float().cuda().view(1, feat_dim, 1)
    netG = NetG(feat_dim, z_dim, z_scale_factors).cuda()
    netD = NetD(feat_dim).cuda()
    netE = Encoder(feat_dim, z_dim, z_scale_factors).cuda()
    recorder = BEGANRecorder(lambda_k, init_k, gamma)
    optimizerG = optim.Adam(netG.parameters(), lr=init_lr)
    optimizerD = optim.Adam(netD.parameters(), lr=init_lr)
    optimizerE = optim.Adam(netE.parameters(), lr=init_lr)
    scaler = GradScaler()
    manager = TrainingManager(
        [netG, netD, netE, recorder], [optimizerG, optimizerD, optimizerE, None],
        ["Generator", "Discriminator", "Encoder", "BEGANRecorder"], output_dir, save_rate,
        script_path=script_path,
    )
    k = recorder.k.item()
    if resume_training:
        init_epoch = manager.resume_training(model_id, save_dir)
        print(f"Resumed k: {k}")
    else:
        init_epoch = 1
        manager.save_initial()

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=20, min_delta=0.0001)

    for epoch in range(init_epoch, 1 + num_epochs):
        print(model_id)
        t0 = time.time()
        print("Training...")
        sum_losses_tr = OrderedDict([(loss_name, 0) for loss_name in ["G", "D", "RealD", "FakeD", "Convergence", "NoiseRecon", "RealRecon"]])
        count_all_tr = 0
        num_batches_tr = batches_per_epoch
        tt0 = time.time()
        netG.train()
        netD.train()
        netE.train()
        for i_batch in range(batches_per_epoch):
            batch = next(inf_iterator_tr)
            step = epoch * batches_per_epoch + i_batch
            voc = batch.cuda()
            check_for_nans(voc, "raw voc before normalization")
            voc = (voc - mean) / std
            check_for_nans(voc, "normalized voc")
            bs, _, nf = voc.size()
            z = torch.zeros((bs, z_dim, int(np.ceil(nf / z_total_scale_factor)))).normal_(0, 1).float().cuda()
            fake_voc = netG(z)
            check_for_nans(fake_voc, "fake_voc")
            z_fake = netE(fake_voc)
            z_real = netE(voc)
            gloss = torch.mean(torch.abs(netD(fake_voc) - fake_voc))
            check_for_nans(gloss, "gloss")
            noise_rloss = torch.mean(torch.abs(z_fake - z))
            check_for_nans(noise_rloss, "noise_rloss")
            real_rloss = torch.mean(torch.abs(netG(z_real)[..., :nf] - voc[..., :nf]))
            check_for_nans(real_rloss, "real_rloss")
            netG.zero_grad()
            netE.zero_grad()
            with autocast():
                scaled_loss = scaler.scale(gloss + lambda_cycle * (noise_rloss + real_rloss))
                scaled_loss.backward(retain_graph=True)
            if max_grad_norm is not None:
                scaler.unscale_(optimizerG)
                scaler.unscale_(optimizerE)
                clip_grad_norm_(netG.parameters(), max_grad_norm)
                clip_grad_norm_(netE.parameters(), max_grad_norm)
            scaler.step(optimizerG)
            scaler.step(optimizerE)
            scaler.update()
            real_dloss = torch.mean(torch.abs(netD(voc) - voc))
            check_for_nans(real_dloss, "real_dloss")
            fake_dloss = torch.mean(torch.abs(netD(fake_voc.detach()) - fake_voc.detach()))
            check_for_nans(fake_dloss, "fake_dloss")
            dloss = real_dloss - k * fake_dloss
            check_for_nans(dloss, "dloss")
            netD.zero_grad()
            scaled_dloss = scaler.scale(dloss)
            scaled_dloss.backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizerD)
                clip_grad_norm_(netD.parameters(), max_grad_norm)
            scaler.step(optimizerD)
            scaler.update()
            k, convergence = recorder(real_dloss, fake_dloss, update_k=True)
            losses = OrderedDict(
                [
                    ("G", gloss),
                    ("D", dloss),
                    ("RealD", real_dloss),
                    ("FakeD", fake_dloss),
                    ("Convergence", convergence),
                    ("NoiseRecon", noise_rloss),
                    ("RealRecon", real_rloss),
                ]
            )
            wandb.log({"loss/train": losses, "epoch": epoch}, step=step)
            losses_tr = OrderedDict([(loss_name, lo.item()) for loss_name, lo in losses.items()])
            for loss_name, lo in losses_tr.items():
                sum_losses_tr[loss_name] += lo
            count_all_tr += 1
            if i_batch % 10 == 0:
                batch_info = "Epoch {}. Batch: {}/{}, T: {:.3f}, ".format(
                    epoch, i_batch, num_batches_tr, time.time() - tt0
                ) + "".join(
                    [
                        "(tr) {}: {:.5f}, ".format(loss_name, lo)
                        for loss_name, lo in losses_tr.items()
                    ]
                )
                print(batch_info, k)
            tt0 = time.time()
        mean_losses_tr = OrderedDict([(loss_name, slo / count_all_tr) for loss_name, slo in sum_losses_tr.items()])
        print("Validation...")
        mean_losses_va = validate(epoch, step, netG, netD, netE, iterator_va, mean, std, z_dim, z_total_scale_factor, z_scale_factors)
        va_metrics = [(metric_name, mean_losses_va[metric_name], higher_or_lower) for metric_name, higher_or_lower in eval_metrics]
        best_va_metrics = manager.check_best_va_metrics(va_metrics, epoch)
        record = {
            "mean_losses_tr": mean_losses_tr,
            "mean_losses_va": mean_losses_va,
            "best_va_metrics": best_va_metrics,
        }
        manager.save_middle(epoch, record, va_metrics)
        manager.print_record(record)
        manager.print_record_in_one_line(best_va_metrics)
        t1 = time.time()
        print("Epoch: {} finished. Time: {:.3f}. Model ID: {}".format(epoch, t1 - t0, model_id))

        # Update and plot metrics after each epoch
        update_and_plot_metrics(epoch, mean_losses_tr, mean_losses_va)

        # Early stopping check after each epoch
        if early_stopping.step(mean_losses_va["RealRecon"]):
            print(f"Early stopping at epoch {epoch}")
            break

    print(model_id)

