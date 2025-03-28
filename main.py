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
from models import EDMLoss
from engine import train_one_epoch

from utils import *

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



def initialize_model_and_optimizer(args):
    device = torch.device(args.device)
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


def train(args, device):
    """Main training loop."""
    data_loader_train = setup_data_loader(args)
    model, optimizer, loss_scaler = initialize_model_and_optimizer(args, device)
    criterion = EDMLoss(dist=args.target)

    misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    logging.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, criterion, device, epoch, loss_scaler,
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
