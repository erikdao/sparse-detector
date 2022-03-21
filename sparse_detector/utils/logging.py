"""
Logging utilities
"""
import time
import logging
import datetime
from collections import deque, defaultdict

import torch
import torch.distributed as dist

from sparse_detector.utils.distributed import is_dist_avail_and_initialized, rank_zero_only


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU friendly python command line logger"""
    logger = logging.getLogger(name)
    # Set the default level to debug
    logger.addHandler(logging.StreamHandler())

    # This ensures all logging levels get marked with rank_zero_only,
    # otherwise logs would get multiplied for each GPU process in multiple-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical"
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    setattr(logger, "debug", rank_zero_only(getattr(logger, "debug")))

    return logger

log = get_logger(__name__)


@rank_zero_only
def log_to_wandb(run, data, extra_data=None, global_step=None, epoch=None, prefix="train"):
    # Remove AP metrics as they require separate logging logics
    data_copy = {**data}
    data_copy.pop('coco_eval_bbox', None)

    main_metrics = ("loss", "loss_ce", "loss_bbox", "loss_giou", "class_error", "cardinality_error_unscaled")
    log_dict = dict()
    for key, value in data_copy.items():
        if isinstance(value, SmoothedValue):
            data_copy[key] = value.global_avg
    # train-epoch-main-metrics, val-epoch-main-metrics
    for key in main_metrics:
        log_dict[f"{prefix}-main-metrics/{key}"] = data_copy.pop(key)

    # train-metrics, val-metrics
    for key, value in data_copy.items():
        log_dict[f"{prefix}-metrics/{key}"] = value
    
    # train-extra-metrics, val-extra-metrics
    if extra_data:
        for key, value in extra_data.items():
            if isinstance(value, SmoothedValue):
                value = value.global_avg
            log_dict[f"{prefix}-extra-metrics/{key}"] = value
        if global_step is not None:
            log_dict[f"{prefix}-extra-metrics/global_step"] = global_step

    if epoch is not None:
        log_dict[f"{prefix}-main-metrics/epoch"] = epoch
        log_dict[f"{prefix}-metrics/epoch"] = epoch

    if global_step is not None:
        log_dict[f"{prefix}-metrics/global_step"] = global_step

    run.log(log_dict)


@rank_zero_only
def log_ap_to_wandb(run, stats, epoch=None, prefix=None):
    """Log AP metrics to WandB"""
    metrics = dict(
       mAP=stats[0],
       AP_50=stats[1],
       AP_75=stats[2],
       AP_S=stats[3],
       AP_M=stats[4],
       AP_L=stats[5],
    )

    log_dict = dict()
    if prefix is not None:
        for key, value in metrics.items():
            log_dict[f"{prefix}/{key}"] = value
    
    if epoch is not None:
        log_dict[f"{prefix}/epoch"] = epoch

    run.log(log_dict)


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a window or the global series average.
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
            value=self.value
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t", wandb_run=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.wandb_run = wandb_run

    def update(self, **kwargs):
        for k, v in kwargs.items():
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

    def log_every(self, iterable, log_freq=50, global_step=None, header=None, prefix="train", epoch=None):
        """
        Log stats every `log_freq` steps.
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'global step: {global_step}',
            'eta: {eta}',
            '{meters}',
            'time: {iter_time}',
            'data: {data_time}',
            'max mem: {memory:.0f}'
        ])
        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            if global_step is not None:  # for training
                yield obj, global_step
            else:  # for testing
                yield obj
            iter_time.update(time.time() - end)

            log_cond = False  # logging condition
            if global_step is not None:
                log_cond = global_step == 0 or (global_step + 1) % log_freq == 0
            else:
                log_cond = i % log_freq == 0 or i == len(iterable) - 1

            if log_cond:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                # Log to console
                extra_data = dict(
                    iter_time=float(str(iter_time)),
                    data_time=float(str(data_time)),
                    memory=torch.cuda.max_memory_allocated() / MB
                )
                # Log to wandb
                if self.wandb_run is not None:
                    log_to_wandb(
                        self.wandb_run,
                        data=self.meters,
                        extra_data=extra_data,
                        global_step=global_step,
                        epoch=epoch,
                        prefix=prefix
                    )
                else:  # Otherwise, log to console
                    print(log_msg.format(
                        i, len(iterable), global_step=global_step, eta=eta_string,
                        meters=str(self), iter_time=str(iter_time), data_time=str(data_time), memory=extra_data["memory"]
                    ))
            i += 1
            if global_step is not None:
                global_step += 1  # Increase the global step
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / len(iterable)))
