# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, DINO, and BIOT code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# https://github.com/ycq091044/BIOT
# ---------------------------------------------------------

import io
import os
import time
import datetime
import pickle
import math
import numpy as np
import argparse
from collections import defaultdict, deque

import json
import glob
import h5py
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.stats
from timm.utils import get_state_dict

import torch
import torch.distributed as dist
from torch import inf
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter
from data_processor.dataset import ShockDataset, DistillatedSingleShockDataset
from scipy.signal import resample
from scipy.stats import pearsonr
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
from sklearn.metrics import mean_squared_error, r2_score


standard_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def performance_metrics(Y, pre_Y, method):
    assert Y.shape == pre_Y.shape

    if method == 'r2':
        score = list(r2_score(Y, pre_Y, multioutput='raw_values'))
    elif method == 'rmse':
        score = list(mean_squared_error(Y, pre_Y, multioutput='raw_values', squared=False))
    elif method == 'pearsonr':
        score = [pearsonr(Y[:, idx], pre_Y[:, idx])[0] for idx in range(Y.shape[1])]
    else:
        print('Error! {} is undefined.'.format(method))
    score = sum(score) / len(score)
    return score


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
            or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def update_image(self, head='images', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            self.writer.add_image(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print_(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print_


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False):
    world_size = get_world_size()

    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=op, async_op=async_op)

    return tensor


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


def _get_rank_env():
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])


def _get_local_rank_env():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])


def _get_world_size_env():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = _get_rank_env()
        args.world_size = _get_world_size_env()  # int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = _get_local_rank_env()
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,
                 layer_names=None):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters, layer_names=layer_names)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0, layer_names=None) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device

    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        # norm_type)
        layer_norm = torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters])
        total_norm = torch.norm(layer_norm, norm_type)
        # print(layer_norm.max(dim=0))

        if layer_names is not None:
            if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1.0:
                value_top, name_top = torch.topk(layer_norm, k=5)
                print(f"Top norm value: {value_top}")
                print(f"Top norm name: {[layer_names[i][7:] for i in name_top.tolist()]}")

    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    """
    Creates a learning rate schedule that follows a cosine decay with an optional warm-up phase.

    Parameters:
    -----------
    base_value : float
        The initial learning rate value after the warm-up period.
    final_value : float
        The final learning rate value at the end of the training schedule.
    epochs : int
        The total number of training epochs.
    niter_per_ep : int
        The number of iterations (steps) per epoch.
    warmup_epochs : int, optional
        Number of epochs to use for a warm-up phase, where the learning rate gradually increases
        from `start_warmup_value` to `base_value`. Default is 0 (no warm-up).
    start_warmup_value : float, optional
        The initial learning rate value at the start of the warm-up phase. Default is 0.
    warmup_steps : int, optional
        An alternative to `warmup_epochs` that sets the total number of steps for the warm-up phase.
        Overrides `warmup_epochs` if specified. Default is -1 (use `warmup_epochs`).

    Returns:
    --------
    schedule : numpy.ndarray
        An array of learning rate values for each iteration in the training schedule,
        with a warm-up phase followed by cosine decay from `base_value` to `final_value`.

    Notes:
    ------
    - The warm-up phase linearly interpolates from `start_warmup_value` to `base_value` over
      `warmup_steps` or `warmup_epochs * niter_per_ep` steps.
    - After the warm-up, the cosine decay schedule gradually reduces the learning rate from
      `base_value` to `final_value` over the remaining iterations.
    - The total length of the schedule array matches `epochs * niter_per_ep`.

    Example:
    --------
    schedule = cosine_scheduler(0.1, 0.001, epochs=100, niter_per_ep=500, warmup_epochs=10)

    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None,
               save_ckpt_freq=1):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    if not getattr(args, 'enable_deepspeed', False):
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        if epoch == 'best':
            checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name), ]
        elif (epoch + 1) % save_ckpt_freq == 0:
            checkpoint_paths.append(output_dir / ('checkpoint-%s.pth' % epoch_name))

        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                # 'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            if loss_scaler is not None:
                to_save['scaler'] = loss_scaler.state_dict()

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            if optimizer_disc is not None:
                to_save['optimizer_disc'] = optimizer_disc.state_dict()

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None):
    output_dir = Path(args.output_dir)

    if not getattr(args, 'enable_deepspeed', False):
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint.pth'))
            if len(all_checkpoints) > 0:
                args.resume = os.path.join(output_dir, 'checkpoint.pth')
            else:
                all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('-')[-1].split('.')[0]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])  # strict: bool=True, strict=False
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"Resume checkpoint at epoch {checkpoint['epoch']}")
                args.start_epoch = 1  # checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
            if 'optimizer_disc' in checkpoint:
                optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])


def create_ds_config(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))


def build_pretraining_dataset(datasets: list, time_window: list = None, stride_size=200, start_percentage=0,
                              end_percentage=1, sample_percentage: float = 1.0, distillated_path: Path = None,
                              is_return_subject_id: bool = False):
    """
    Builds a list of ShockDataset instances for pretraining and retrieves channel names.

    This function processes a list of datasets and their corresponding time windows, creating a `ShockDataset` instance
    for each dataset. The datasets are sampled with a specified stride size, and data extraction is restricted to a
    specified percentage range.

    Args:
        datasets (list):
            A list of lists, where each inner list contains file paths (as strings) to datasets
            to be processed for pretraining.
        time_window (list):
            A list of integers representing the time window (in seconds) for each dataset.
            The time window is used to determine the number of samples per patch.
            None means the list is automatically inferred from datasets.
        stride_size (int, optional):
            The interval (in samples) between two consecutive patches in the dataset. Default is 200.
        start_percentage (float, optional):
            The starting percentage of the dataset to include (0.0 <= start_percentage <= 1.0).
            Default is 0, meaning from the beginning of the dataset.
        end_percentage (float, optional):
            The ending percentage of the dataset to include (0.0 <= end_percentage <= 1.0).
            Default is 1, meaning up to the end of the dataset.

    Returns:
        tuple:
            - shock_dataset_list (list): A list of `ShockDataset` objects created for each input dataset.
            - ch_names_list (list): A list of channel names for each dataset, as returned by `get_ch_names()`.
    """
    if time_window is None:
        channel_num2dataset = defaultdict(list)
        for i, dataset in enumerate(datasets):
            try:
                f = h5py.File(dataset, 'r')
                k = list(f.keys())[0]
                eeg_shape = f[k]['eeg'].shape
                print('[DEBUG] dataset{}={}\tshape_num={}'.format(i, dataset, eeg_shape))
                channel_num = eeg_shape[0]
                channel_num2dataset[channel_num].append(dataset)
                f.close()
            except OSError as e:
                print('[WARNING] {} is skipped due to OSError {}'.format(dataset, e))
        channel_num2dataset = dict(channel_num2dataset)
        print(f'[INFO] {channel_num2dataset}')
    shock_dataset_list = []
    ch_names_list = []
    for channel_num, dataset_list in channel_num2dataset.items():
        window_size = 256 // channel_num
        try:
            dataset = ShockDataset([Path(file_path) for file_path in dataset_list], window_size * 200, stride_size,
                                   start_percentage, end_percentage, sample_percentage, distillated_path, is_return_subject_id)
            shock_dataset_list.append(dataset)
            ch_names_list.append(dataset.get_ch_names())
        except IsADirectoryError as e:
            print(f'[WARNING] Unable to open {dataset_list} due to the error {e}')
            continue
    return shock_dataset_list, ch_names_list


def get_input_chans(ch_names):
    input_chans = [0]  # for cls token
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name.upper()) + 1)
    return input_chans


class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y


class MOBILoader(torch.utils.data.Dataset):
    def __init__(self, filePath, target):
        self.root = filePath
        self.target = target
        self.data = np.load(self.root)
        self.EEG = self.data[f'{target}EEG']
        self.joints = self.data[f'{target}Joints']

    def __len__(self):
        return (self.EEG.shape[1] - 400 + 10) // 10

    def __getitem__(self, index):
        # 2秒一次预测，每次跨 1 个采样点
        X = self.EEG[:, index * 10:index * 10 + 400]
        Y = self.joints[:, index * 10 + 399]
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
        return X, Y


class DownstreamDataset(Dataset):
    """读取单个hdf5文件，仅使用范式中途采集的数据，标签内包含范式标签与被试性别，如有需要可以继续往字典中添加"""

    def __init__(self, file_path: Path, window_size: int = 200, stride_size: int = 1, start_percentage: float = 0,
                 end_percentage: float = 1, trial_start_percentage: float = 0, trial_end_percentage: float = 1,
                 subject_start_percentage: float = 0, subject_end_percentage: float = 1):
        """
        从路径file_path中提取数据集。

        :param Path file_path: 目标数据路径
        :param int window_size: 单个样本长度
        :param int stride_size: 两个相邻样本间隔
        :param float start_percentage: 数据集中，每个采纳的trial内首个样本在此trial的样本中的百分比索引（包括）。
        :param float end_percentage: 数据集中，每个采纳的trial内末尾样本在此trial的样本中的百分比索引（不包括）。
        :param float trial_start_percentage: 数据集中，采纳的首个trial在此被试的所有trial中的百分比索引（包括）。
        :param float trial_end_percentage: 数据集中，采纳的末个trial在此被试的所有trial中的百分比索引（不包括）。
        :param float subject_start_percentage: 数据集中，采纳的首个被试的百分比索引（包括）。
        :param float subject_end_percentage: 数据集中，采纳的末个被试的百分比索引（不包括）。

        比如，数据文件总共10个被试，每个被试有15个trial，每个trial提供100个样本时。取参数为0.2, 0.8, 0.34, 0.67, 0.2, 0.8时，数据集会包括下标为[2, 8)的被试，每个被试的下标为[5, 10)的trial中，每个trial下标为[20, 80)的样本。
        """
        self.__file_path = file_path
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage
        self.__trial_start_percentage = trial_start_percentage
        self.__trial_end_percentage = trial_end_percentage
        self.__subject_start_percentage = subject_start_percentage
        self.__subject_end_percentage = subject_end_percentage

        self.__file = None
        self.__length = None
        self.__feature_size = None

        self.__subjects = []
        self.__global_idxes = []  # 从第几个样本开始是哪个被试
        self.__local_idxess = []  # 从这个被试的第几个样本开始是哪个trial
        self.__trial_start_idxess = []  # trial开始索引
        self.__genders = []
        self.__labelss = []

        self.__rsFreq = None

        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.__subjects = [i for i in self.__file]

        global_idx = 0
        subject_start_id = int(len(self.__subjects) * self.__subject_start_percentage)  # 包括在数据集中的被试开始id
        subject_end_id = int(len(self.__subjects) * self.__subject_end_percentage - 1)  # 包括在数据集中的被试结束id
        for subject_id, subject in enumerate(self.__subjects):
            self.__global_idxes.append(global_idx)
            self.__genders.append(self.__file[subject].attrs['gender'])
            self.__labelss.append(self.__file[subject].attrs['label'])
            self.__rsFreq = self.__file[subject]['eeg'].attrs['rsFreq']

            local_idxes = []  # 当前trial的第一个样本在数据集中的样本索引
            trial_start_idxes = []  # 当前trial在原始数据中的开始位置索引
            trial_starts = self.__file[subject].attrs['trialStart']
            trial_ends = self.__file[subject].attrs['trialEnd']
            local_idx = 0
            if subject_start_id <= subject_id <= subject_end_id:
                trial_start_id = int(len(trial_starts) * self.__trial_start_percentage)  # 该被试包括在数据集中的trial开始id
                trial_end_id = int(len(trial_starts) * self.__trial_end_percentage - 1)  # 该被试包括在数据集中的trial结束id
                for trial_id, (trial_start, trial_end) in enumerate(zip(trial_starts, trial_ends)):
                    local_idxes.append(local_idx)

                    if trial_start_id <= trial_id <= trial_end_id:
                        trial_len = (trial_end - trial_start + 1) * self.__rsFreq
                        trial_sample_num = (trial_len - self.__window_size) // self.__stride_size + 1
                        start_idx = int(
                            trial_sample_num * self.__start_percentage) * self.__stride_size + trial_start * self.__rsFreq
                        end_idx = int(
                            trial_sample_num * self.__end_percentage - 1) * self.__stride_size + trial_start * self.__rsFreq

                        trial_start_idxes.append(start_idx)
                        local_idx += (end_idx - start_idx) // self.__stride_size + 1
                    else:
                        trial_start_idxes.append(0)

            self.__local_idxess.append(local_idxes)
            self.__trial_start_idxess.append(trial_start_idxes)

            global_idx += local_idx

        self.__length = global_idx

        self.__feature_size = [i for i in self.__file[self.__subjects[0]]['eeg'].shape]
        self.__feature_size[1] = self.__window_size

    @property
    def feature_size(self):
        return self.__feature_size

    @property
    def rsfreq(self):
        return self.__rsFreq

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        # 先确认样本属于哪个被试，再确认样本属于哪个trial
        subject_id = bisect.bisect(self.__global_idxes, idx) - 1
        trial_id = bisect.bisect(self.__local_idxess[subject_id], idx - self.__global_idxes[subject_id]) - 1
        item_start_idx = (idx - self.__global_idxes[subject_id] - self.__local_idxess[subject_id][
            trial_id]) * self.__stride_size + self.__trial_start_idxess[subject_id][trial_id]

        labels = {'gender': self.__genders[subject_id], 'label': self.__labelss[subject_id][trial_id]}

        return self.__file[self.__subjects[subject_id]]['eeg'][:,
               item_start_idx:item_start_idx + self.__window_size], labels

    def free(self) -> None:
        # TODO 临时方案，目标：减少文件打开次数。查一下flush
        if self.__file:
            self.__file.close()
            self.__file = None

    def get_ch_names(self):
        return self.__file[self.__subjects[0]]['eeg'].attrs['chOrder']


def prepare_TUEV_dataset(root):
    # set random seed
    seed = 4523
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "processed_train"))
    val_files = os.listdir(os.path.join(root, "processed_eval"))
    test_files = os.listdir(os.path.join(root, "processed_eval"))

    # prepare training and test data loader
    train_dataset = TUEVLoader(
        os.path.join(
            root, "processed_train"), train_files
    )
    test_dataset = TUEVLoader(
        os.path.join(
            root, "processed_eval"), test_files
    )
    val_dataset = TUEVLoader(
        os.path.join(
            root, "processed_eval"), val_files
    )
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_MoBI_dataset(npz_path):
    # prepare training and test data loader
    train_dataset = MOBILoader(npz_path, 'train')
    val_dataset = MOBILoader(npz_path, 'val')
    test_dataset = MOBILoader(npz_path, 'test')
    return train_dataset, test_dataset, val_dataset


def prepare_TUAB_dataset(root):
    # set random seed
    seed = 12345
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = TUABLoader(os.path.join(root, "train"), train_files)
    test_dataset = TUABLoader(os.path.join(root, "test"), test_files)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

        # Open the HDF5 file in read mode
        with h5py.File(self.file_path, "r") as f:
            f.keys()

    def __len__(self):
        return self.data_length


def prepare_HDF5_dataset(file_path):
    train_dataset = HDF5Dataset(file_path)

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    return train_dataset, val_dataset, test_dataset


def get_metrics(output, target, metrics, is_binary, is_regression: bool = False, threshold=0.5,
                trial_info: list = None):
    """
    returns PCC and RMSE when is_regression is True
    :return: dict
    """
    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (
                len(target) - sum(target)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                target,
                output,
                metrics=metrics,
                threshold=threshold,
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
    elif is_regression:
        results = {}
        # Compute R² (coefficient of determination)
        results["r2"] = r2_score(target, output)
        # Compute RMSE (Root Mean Squared Error)
        results["rmse"] = np.sqrt(mean_squared_error(target, output))
        # Compute Pearson correlation coefficient
        # Since Pearson's r works per channel, we compute it for each channel separately
        pearson_corrs = [
            pearsonr(target[:, i], output[:, i])[0] if np.std(target[:, i]) > 0 else 0.0
            for i in range(target.shape[1])
        ]
        results["pearsonr"] = np.mean(pearson_corrs)  # Taking mean correlation across channels
    else:
        results = multiclass_metrics_fn(
            target, output, metrics=metrics
        )
    return results


if __name__ == '__main__':
    prepare_HDF5_dataset("/home/yuting/seed-5.hdf5")
