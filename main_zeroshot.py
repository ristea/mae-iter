import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

assert timm.__version__ == "0.3.2"  # version check

import util.misc as misc
from util.datasets import build_dataset_zero_shot
from util.pos_embed import interpolate_pos_embed

import models_vit
from engine_zeroshot import compute_zeroshot_acc


def get_args_parser():
    parser = argparse.ArgumentParser('MAE zero-shot for image classification', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--knn_size', default=5, type=int, help='Neighbours size for KNN')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch8', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=64, type=int, help='images input size')

    # * Finetuning params
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=100, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--checkpoint_path', default='/Users/cristea/Downloads/pretrained.pth',
                        help='resume from checkpoint')

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset_zero_shot(is_train=True, args=args)
    dataset_val = build_dataset_zero_shot(is_train=False, args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=0.0,
        global_pool=args.global_pool,
    )

    # load pre-trained model
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % args.checkpoint_path)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    if args.global_pool:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    else:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    model.to(device)

    print(f"Start zero shot")
    start_time = time.time()
    test_stats = compute_zeroshot_acc(
        model, data_loader_train, data_loader_val, device, args.knn_size
    )

    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

    if log_writer is not None:
        log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], 0)
        log_writer.add_scalar('perf/test_acc_nn', test_stats['acc_nn'], 0)

    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}

    if args.output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
