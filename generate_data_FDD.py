import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.fft as fft
from utils import *
import os

from torch.utils.tensorboard import SummaryWriter

model_zoo = {'deit_tiny': 'deit_tiny_patch16_224',
            'deit_small': 'deit_small_patch16_224',
            'deit_base': 'deit_base_patch16_224',
            'swin_tiny': 'swin_tiny_patch4_window7_224',
            'swin_small': 'swin_small_patch4_window7_224',
            }

class AttentionMap:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.feature = output

    def remove(self):
        self.hook.remove()

def fftmask(h, w, r):
    lmask = torch.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if ((i - (h - 1) / 2) ** 2 + (j - (w - 1) / 2) ** 2 < r ** 2):
                lmask[i, j] = 1
    hmask = 1 - lmask
    lmask, hmask = lmask.cuda(), hmask.cuda()
    return lmask, hmask

def imgfft(calibrate_data, lmask, hmask):
    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2).unsqueeze(0).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2).unsqueeze(0).cuda()
    denorm_data = (calibrate_data * std.expand_as(calibrate_data)) + mean.expand_as(calibrate_data)

    h, w = denorm_data.shape[-2], denorm_data.shape[-1]

    f = fft.fftn(denorm_data, dim=(2,3))
    f = torch.roll(f, (h // 2, w // 2), dims=(2, 3))
    data_l = f * lmask
    data_h = f * hmask
    data_l = torch.abs(fft.ifftn(data_l, dim=(2, 3)))
    data_l = (data_l - mean.expand_as(calibrate_data)) / std.expand_as(calibrate_data)
    data_h = torch.abs(fft.ifftn(data_h, dim=(2, 3)))
    data_h = (data_h - mean.expand_as(calibrate_data)) / std.expand_as(calibrate_data)
    return data_l, data_h

def generate_data(args):

    if not os.path.exists('logs'):
        os.mkdir('logs')
    writer = SummaryWriter(log_dir='logs/'+args.model)
    
    args.batch_size = args.calib_batchsize

    # Load pretrained model
    p_model = build_model(model_zoo[args.model], Pretrained=True)

    # Hook the feature map
    hooks = []
    if 'swin' in args.model:
        for m in p_model.layers:
            for n in range(len(m.blocks)):
                hooks.append(AttentionMap(m.blocks[n].attn.matmul2))
    else:
        for m in p_model.blocks:
            hooks.append(AttentionMap(m.attn.matmul2))

    # Init Gaussian noise
    img = torch.randn((args.batch_size, 3, 224, 224)).cuda()
    img.requires_grad = True

    # Init optimizer
    args.lr = 0.25 if 'swin' in args.model else 0.20
    optimizer = optim.Adam([img], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)

    # Set pseudo labels
    pred = torch.LongTensor([random.randint(0, 999) for _ in range(args.batch_size)]).to('cuda')
    var_pred = random.uniform(2500, 3000)  # for batch_size 32

    # Set criterion
    criterion = nn.CrossEntropyLoss()
    KL_Loss = nn.KLDivLoss(reduction = 'batchmean')

    # Get frequency domain mask
    lmask, _ = fftmask(img.shape[-2], img.shape[-1], 112 * 0.7)
    _, hmask = fftmask(img.shape[-2], img.shape[-1], 112 * 0.1)

    # AdaIteration
    if 'tiny' in args.model:
        iterations = 100
    elif 'swin_small' in args.model:
        iterations = 200
    else:
        iterations = 300

    # Train for two epochs
    for lr_it in range(3):
        if lr_it == 0:
            iterations_per_layer = iterations
            lim = int(112 * 0.3)
        elif lr_it == 1:
            iterations_per_layer = iterations
            lim = int(112 * 0.2)
        else:
            iterations_per_layer = iterations
            lim = int(112 * 0.1)

        lr_scheduler = lr_cosine_policy(args.lr, iterations // 5, iterations_per_layer)

        with tqdm(range(iterations_per_layer)) as pbar:
            for sss, itr in enumerate(pbar):
                pbar.set_description(f"Epochs {lr_it+1}/{3}")

                # Learning rate scheduling
                lr_scheduler(optimizer, itr, itr)

                # fft to get img_l, img_h or org img as input
                img_l, img_h = imgfft(img, lmask, hmask)
                if lr_it == 0:
                    img_input = img_l
                elif lr_it == 1:
                    img_input = img_h
                else:
                    img_input = img

                # Apply random jitter offsets (from DeepInversion[1])
                # [1] Yin, Hongxu, et al. "Dreaming to distill: Data-free knowledge transfer via deepinversion.", CVPR2020.
                off = random.randint(-lim, lim)
                img_jit = torch.roll(img_input, shifts=(off, off), dims=(2, 3))
                img_jit_l = torch.roll(img_l, shifts=(off, off), dims=(2, 3))
                img_jit_h = torch.roll(img_h, shifts=(off, off), dims=(2, 3))
                # Flipping
                flip = random.random() > 0.5
                if flip:
                    img_jit = torch.flip(img_jit, dims=(3,))
                    img_jit_l = torch.flip(img_jit_l, dims=(3,))
                    img_jit_h = torch.flip(img_jit_h, dims=(3,))

                # Forward pass
                optimizer.zero_grad()
                p_model.zero_grad()

                output = p_model(img_jit)

                if lr_it == 0:
                    loss_kl = torch.zeros(1).cuda()
                else:
                    teacher_output = p_model(img_jit_l).clone().softmax(dim=-1).detach()
                    student_output = output.log_softmax(dim=-1)
                    loss_kl = KL_Loss(student_output, teacher_output)

                loss_hard = criterion(output, pred)

                if lr_it == 0:
                    loss_oh = loss_hard
                else:
                    loss_oh = loss_hard * 0.5 + loss_kl * 0.5

                loss_tv = torch.norm(get_image_prior_losses(img_jit) - var_pred)

                loss_entropy = 0
                for itr_hook in range(len(hooks)):
                    # Hook attention feature
                    attention = hooks[itr_hook].feature
                    attention_p = attention.mean(dim=1)[:, 1:, :]
                    sims = torch.cosine_similarity(attention_p.unsqueeze(1), attention_p.unsqueeze(2), dim=3)

                    # Compute differential entropy
                    kde = KernelDensityEstimator(sims.view(args.batch_size, -1))
                    start_p = sims.min().item()
                    end_p = sims.max().item()
                    x_plot = torch.linspace(start_p, end_p, steps=10).repeat(args.batch_size, 1).cuda()
                    kde_estimate = kde(x_plot)
                    dif_entropy_estimated = differential_entropy(kde_estimate, x_plot)
                    loss_entropy -= dif_entropy_estimated

                # Combine loss
                total_loss = loss_entropy + loss_oh + 0.05 * loss_tv

                # Record loss
                writer.add_scalar('Total Loss', total_loss.item(), global_step=lr_it*100+sss)
                writer.add_scalar('OH Loss', loss_oh.item(), global_step=lr_it*100+sss)
                writer.add_scalar('Hard Loss', loss_hard.item(), global_step=lr_it*100+sss)
                writer.add_scalar('KL Loss', loss_kl.item(), global_step=lr_it*100+sss)
                writer.add_scalar('Entropy Loss', loss_entropy.item(), global_step=lr_it*100+sss)
                writer.add_scalar('TV Loss', loss_tv.item(), global_step=lr_it*100+sss)

                # Do image update
                total_loss.backward()

                optimizer.step()

                # Clip color outliers
                img.data = clip(img.data)

    return img.detach()

def differential_entropy(pdf, x_pdf):  
    # pdf is a vector because we want to perform a numerical integration
    pdf = pdf + 1e-4
    f = -1 * pdf * torch.log(pdf)
    # Integrate using the composite trapezoidal rule
    ans = torch.trapz(f, x_pdf, dim=-1).mean()  
    return ans

def get_image_prior_losses(inputs_jit):
    # Compute total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    return loss_var_l2

def clip(image_tensor, use_fp16=False):
    # Adjust the input based on mean and variance
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor

def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr

def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)
