import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import logging

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from util import lr_decay as lrd, misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models

from train_utils import *
import trimesh

import math
import sys

import util.misc as misc
import util.lr_sched as lr_sched


from PIL import Image



def get_inline_arg():
    parser = argparse.ArgumentParser('Train', add_help=False)
    parser.add_argument('--batch_size', default=1024*64*2, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    

    # Model parameters
    parser.add_argument('--model', default='EDMPrecond', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--depth', default=6, type=int, metavar='MODEL')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-7, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=5e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--target', default='Gaussian', type=str, )
    parser.add_argument('--data_path', default='shapes/Jellyfish_lamp_part_A__B_normalized.obj', type=str,
                        help='dataset path')

    parser.add_argument('--texture_path', default=None, type=str,
                        help='dataset path')

    parser.add_argument('--noise_mesh', default=None, type=str,
                        help='dataset path')
     
    parser.add_argument('--output_dir', default='./output/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    args = parser.parse_args()

    return args


# Constants
NEURAL_RENDERING_RESOLUTION = 128

def setup_data_loader(args):
    """Set up the data loader based on the input data path."""
    if args.data_path.endswith(('.obj', '.ply')):
        data_loader_train = {
            'obj_file': args.data_path,
            'batch_size': args.batch_size,
            'epoch_size': 512,
            'texture_path': args.texture_path,
            'noise_mesh': args.noise_mesh or None
        }
    elif any(primitive in args.data_path for primitive in ['sphere', 'plane', 'volume']):
        data_loader_train = {
            'obj_file': None,
            'primitive': args.data_path,
            'batch_size': args.batch_size,
            'epoch_size': 512,
            'texture_path': args.texture_path,
            'noise_mesh': args.noise_mesh or None
        }
    else:
        raise NotImplementedError(f"Unsupported data path: {args.data_path}")
    return data_loader_train



def initialize_model_and_optimizer(args,device):
    model = models.__dict__[args.model](channels=3 if args.texture_path is None else 6, depth=args.depth)
    model.to(device)
    
    
    """Initialize the model, optimizer, and loss scaler."""
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 128
    print("base lr: %.2e" % (args.lr * 128 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp =model.module
        optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    loss_scaler = NativeScaler()
    return model, optimizer, loss_scaler


def train_one_epoch(model: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    # Set the model to training mode
    model.train(True)

    # Initialize metric logger
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20

    # Gradient accumulation setup
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    # Log directory if log_writer is provided
    if log_writer is not None:
        print(f'log_dir: {log_writer.log_dir}')
    
    print(data_loader)

    # Initialize noise and samples
    noise = None

    # Handle data loader setup
    if isinstance(data_loader, dict):
        obj_file = data_loader['obj_file']
        batch_size = data_loader['batch_size']

        # Load samples from an object file
        if obj_file is not None:
            mesh = trimesh.load(obj_file)
            if len(mesh.faces)>0:
                if data_loader['texture_path'] is not None:
                    # Load texture and apply it to the mesh
                    img = Image.open(data_loader['texture_path'])
                    material = trimesh.visual.texture.SimpleMaterial(image=img)
                    assert mesh.visual.uv is not None
                    texture = trimesh.visual.TextureVisuals(mesh.visual.uv, image=img, material=material)
                    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, visual=texture, process=False)

                    # Sample points and colors from the mesh
                    samples, _, colors = trimesh.sample.sample_surface(mesh, 2048 * 64 * 4 * 64, sample_color=True)
                    colors = colors[:, :3]  # Remove alpha channel
                    colors = (colors.astype(np.float32) / 255.0 - 0.5) / np.sqrt(1 / 12)  # Normalize colors
                    samples = np.concatenate([samples, colors], axis=1)
                else:
                    # Sample points from the mesh without texture
                    samples, _ = trimesh.sample.sample_surface(mesh, 2048 * 64 * 4 * 64)
                    #samples = mesh.vertices

            else:
                samples = mesh.vertices
        # Generate primitive shapes if no object file is provided
        else:
            if data_loader['primitive'] == 'sphere':
                n = torch.randn(2048 * 64 * 4 * 64, 3)
                n = torch.nn.functional.normalize(n, dim=1)
                samples = n / np.sqrt(1 / 3)
                samples = samples.numpy()
            elif data_loader['primitive'] == 'plane':
                samples = torch.rand(2048 * 64 * 4 * 64, 3) - 0.5
                samples[:, 2] = 0
                samples = (samples - 0) / np.sqrt(2 / 9 * 2 * 0.5 ** 3)
                samples = samples.numpy()
            elif data_loader['primitive'] == 'volume':
                samples = (torch.rand(2048 * 64 * 4 * 64, 3) - 0.5) / np.sqrt(1 / 12)
                samples = samples.numpy()
            elif data_loader['primitive'] == 'gaussian':
                samples = np.random.randn(2048 * 64 * 4 * 64, 3).astype(np.float32)
            else:
                raise NotImplementedError

        # Load noise mesh if provided
        if data_loader['noise_mesh'] is not None:
            noise, _ = trimesh.sample.sample_surface(trimesh.load(data_loader['noise_mesh']), 2048 * 64 * 4 * 64)
        else:
            noise = None

        # Convert samples to float32
        samples = samples.astype(np.float32)
        data_loader = range(data_loader['epoch_size'])

    # Training loop
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # Adjust learning rate per iteration
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Prepare input data
        if isinstance(batch, int):
            ind = np.random.default_rng().choice(samples.shape[0], batch_size, replace=True)
            xyz = samples[ind]
            xyz = torch.from_numpy(xyz).float().to(device, non_blocking=True)
        else:
            xyz = batch.to(device, non_blocking=True)

        # Forward pass and loss computation
        with torch.amp.autocast('cuda',enabled=False):
            if noise is not None:
                ind = np.random.default_rng().choice(noise.shape[0], batch_size, replace=True)
                init_noise = noise[ind]
                init_noise = torch.from_numpy(init_noise).float().to(device, non_blocking=True)
            else:
                init_noise = None

            # Sample noise  ---> noysis step
            rnd_normal = torch.randn([xyz.shape[0],], device=xyz.device)

            sigma = (rnd_normal * 1.2 -1).exp()
            weight = (sigma ** 2 + 1) / (sigma) ** 2
            y = xyz

            #chose the kind of noise
            if args.target == 'Gaussian':
                n = torch.randn_like(y[:, :3]) * sigma[:, None]
                if y.shape[1] != 3:
                    c = (torch.rand_like(y[:, 3:]) - 0.5) / np.sqrt(1/12) * sigma[:, None]
                    n = torch.cat([n, c], dim=1)
            elif args.target == 'Uniform':
                n = (torch.rand_like(y) - 0.5) / np.sqrt(1/12) * sigma[:, None]
            elif args.target == 'Sphere':
                n = torch.randn_like(y[:, :3])
                n = torch.nn.functional.normalize(n, dim=1)
                n /= np.sqrt(1/3)
                n = n * sigma[:, None]

            elif args.target == "Mesh":
                assert init_noise is not None
                n = init_noise * sigma[:, None]
            else:
                raise NotImplementedError
            
            D_yn = model(y + n, sigma)

            loss = ( weight[:, None] * ((D_yn - y) ** 2) ).mean()
            
        loss_value = loss.item()

        # Handle invalid loss values
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # Backward pass and gradient update
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # Update metrics
        metric_logger.update(loss=loss_value)

        # Track learning rate
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        # Log metrics to TensorBoard
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # Synchronize metrics across processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # Return averaged metrics
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train(args, device):
    """Main training loop."""
    data_loader_train = setup_data_loader(args)
    model, optimizer, loss_scaler = initialize_model_and_optimizer(args, device)

    misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    logging.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=None, args=args
        )
        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    logging.info(f"Training completed in {str(datetime.timedelta(seconds=int(total_time)))}")




def main():
    args = get_inline_arg()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        setup_logging(args.output_dir)
    device = initialize_device_and_seed(args)
    train(args, device)

if __name__ == '__main__':
    main()
