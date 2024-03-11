import os
import random
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from pathnorm.utils.evaluate_accuracy import (
    test_after_training,
    validate,
    accuracy,
)
from pathnorm.path_norm.compute_margin import get_quantile_margins
from pathnorm.path_norm.compute_path_norm import compute_path_norms
from pathnorm.IMP.imp import do_rewind
from pathnorm.utils.log_and_checkpoint import (
    get_logger,
    resume,
    save_init_checkpoint,
    log_and_save_checkpoint,
)
from pathnorm.utils.meters import AverageMeter, ProgressMeter
from pathnorm.dataset.imagenet import get_dataloaders
from pathnorm.optim.scheduler import get_scheduler
from pathnorm.utils.parse_arguments import get_args

best_val_top1 = 0


def set_distributed(args, ngpus_per_node, gpu):
    """
    Set up distributed training if specified in the arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        ngpus_per_node (int): Number of GPUs per node.
        gpu (int): GPU index.

    Raises:
        ValueError: If the distributed training setup is incorrect.
    """
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )


def get_model(args):
    """
    Load the specified model, either pre-trained or a new one.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        torch.nn.Module: Loaded model.
    """
    if args.pretrained:
        print("=> using pre-trained models '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating models '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            print("Using DDP")
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                assert args.batch_size % args.world_size == 0
                args.batch_size = args.batch_size // args.world_size
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu]
                )
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.no_data_parallel:
        model.cuda()
    else:
        if args.device_ids is not None:
            model = torch.nn.DataParallel(
                model, device_ids=args.device_ids
            ).cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # NOTA BENE: channels_last incompatible with pruning atm so we cannot use:
    # model.to(memory_format=torch.channels_last)

    return model


def get_device(args):
    """
    Get the device (CPU or GPU) based on the availability of GPUs.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device("cuda:{}".format(args.gpu))
        else:
            device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def get_optimizer(args, model):
    """
    Initialize the optimizer based on the specified arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model (torch.nn.Module): Model to be optimized.

    Returns:
        torch.optim.Optimizer: Initialized optimizer.
    """
    parameters = model.parameters()
    optimizer = torch.optim.SGD(
        parameters,
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    return optimizer


def main():
    """
    Main function to initiate training.
    """
    args = get_args()

    if args.jean_zay:
        import idr_torch

        args.rank = idr_torch.rank
        args.world_size = idr_torch.size
        args.gpu = idr_torch.local_rank

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args)
        )
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def train_one_epoch(
    train_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    epoch,
    device,
    scaler,
    args,
):
    """
    Train the model for one epoch.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        model (torch.nn.Module): Model to be trained.
        criterion (torch.nn.Module): Loss criterion.
        optimizer (torch.optim.Optimizer): Model optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        epoch (int): Current epoch number.
        device (torch.device): Training device (CPU or GPU).
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed-precision training.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        float: Top-1 accuracy.
        float: Top-5 accuracy.
        float: Average loss.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    lr = AverageMeter("Lr", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    # switch to train mode
    model.train()

    end = time.time()

    new_train_loader = train_loader

    progress = ProgressMeter(
        len(new_train_loader),
        [batch_time, losses, lr, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )
    for i, (images, target) in enumerate(new_train_loader):

        # move data to the same device as models
        # NOTA BENE: channels_last incompatible with pruning so we cannot use:
        # images = images.to(
        #     device, non_blocking=True, memory_format=torch.channels_last
        # )
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        if args.mixup_alpha is None:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

        losses.update(loss.detach().item(), images.size(0))

        optimizer.zero_grad()

        scaler.scale(loss).backward()

        if args.clip_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip_grad_norm
            )
        scaler.step(optimizer)
        scaler.update()

        # Scheduler step at each training iterations
        if scheduler is not None:
            lr.update(scheduler.get_last_lr()[0])
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    return top1.avg, top5.avg, losses.avg


def main_worker(gpu, ngpus_per_node, args):
    """
    Main worker function for distributed training.

    Args:
        gpu (int): GPU index.
        ngpus_per_node (int): Number of GPUs per node.
        args (argparse.Namespace): Parsed command-line arguments.
    """
    global best_val_top1

    args.gpu = gpu

    # OPTIONALLY SET DISTRIBUTED TRAINING
    set_distributed(args, ngpus_per_node, gpu)

    # GET MODEL
    model = get_model(args)

    # GET DEVICE
    device = get_device(args)

    # GET LOSS
    criterion = nn.CrossEntropyLoss().to(device)

    # GET DATALOADERS
    train_loader, val_loader, test_loader, train_sampler = get_dataloaders(
        args
    )

    # GET OPTIMIZER
    optimizer = get_optimizer(args, model)

    # GET LR SCHEDULER
    scheduler = get_scheduler(args, optimizer, len(train_loader))

    # OPTIONALLY RESUME FROM A CHECKPOINT
    if args.resume:
        model, optimizer, scheduler, corresponding_epoch, best_val_top1 = (
            resume(
                model,
                optimizer,
                scheduler,
                args.resume,
                args.gpu,
                args.start_IMP_iter,
            )
        )
        args.start_epoch = corresponding_epoch

    # OPTIONALLY EVALUATE MODEL BEFORE TRAINING
    if args.evaluate_before_train:
        validate(
            test_loader,
            model,
            criterion,
            device,
            args.gpu,
            args.print_freq,
            args.distributed,
            args.world_size,
            args.batch_size,
            args.workers,
            prefix_print="Test",
        )
        exponents = [1, 2, 4, 8, 16]
        pathnorms = compute_path_norms(
            model, args.arch, exponents, device, not args.no_data_parallel
        )
        print(
            "Path-norms before training (L1, L2, L4, L8, L16)= \n\t", pathnorms
        )

    scaler = torch.cuda.amp.GradScaler()

    # GET LOGGER
    logger = get_logger(args.saving_dir, args.rank, args.tensorboard)

    # SAVE INITIAL CHECKPOINT
    if args.start_epoch == 0 and args.start_IMP_iter == 0:
        save_init_checkpoint(
            model,
            optimizer,
            scheduler,
            ngpus_per_node,
            args.multiprocessing_distributed,
            args.rank,
            best_val_top1,
            args.arch,
            args.saving_dir,
        )

    # TRAINING LOOP
    print("=> Begin training")
    for imp_iter in range(args.start_IMP_iter, args.IMP_iters + 1):
        print(f"=> Starting IMP iteration {imp_iter}")
        # RESET BEST VAL TOP1 AT EACH NEW IMP ITER
        best_val_top1 = 0
        # REWINDS at epoch 5
        if imp_iter > 0:
            model, new_starting_epoch = do_rewind(
                model,
                optimizer,
                scheduler,
                imp_iter,
                args.saving_dir,
                args.gpu,
                args.percentage_pruning,
            )
            args.start_epoch = new_starting_epoch
        # args.start_epoch = 5 if rewinded, 0 for first iteration
        for epoch in range(args.start_epoch, args.epochs):
            # TIMER
            begin = time.time()

            if args.distributed:
                train_sampler.set_epoch(epoch)

            # TRAIN ONE EPOCH
            train_top1, train_top5, train_loss = train_one_epoch(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                device,
                scaler,
                args,
            )

            epoch_time = time.time() - begin

            # VAL AND TEST ACC

            val_top1, val_top5, val_loss = validate(
                val_loader,
                model,
                criterion,
                device,
                args.gpu,
                args.print_freq,
                args.distributed,
                args.world_size,
                args.batch_size,
                args.workers,
                "Val",
            )

            test_top1, test_top5, test_loss = validate(
                test_loader,
                model,
                criterion,
                device,
                args.gpu,
                args.print_freq,
                args.distributed,
                args.world_size,
                args.batch_size,
                args.workers,
                "Test",
            )

            # Do not call scheduler at each epoch, but at each iteration (for warmup)
            # scheduler.step()

            # COMPUTE PATH-NORMS
            print(
                f"=> Computing path-norms at the end of current epoch ({epoch})"
            )
            exponents = [1, 2, 4, 8, 16]
            pathnorms = compute_path_norms(
                model, args.arch, exponents, device, not args.no_data_parallel
            )

            print(
                f"=> Computing margins at the end of current epoch ({epoch})"
            )

            quantile_margins_train = get_quantile_margins(
                model, train_loader, device
            )

            print(
                f"=> Logging and saving checkpoint at the end of current epoch ({epoch})"
            )
            # LOG, AND SAVE CHECKPOINT IF BEST OR IF EPOCH 5 AND IMP_ITER=0 FOR REWIND
            log_and_save_checkpoint(
                val_top1,
                ngpus_per_node,
                epoch,
                model,
                optimizer,
                scheduler,
                train_loss,
                train_top1,
                train_top5,
                val_loss,
                val_top5,
                test_loss,
                test_top5,
                test_top1,
                pathnorms,
                quantile_margins_train,
                best_val_top1,
                epoch_time,
                logger,
                imp_iter,
                args.multiprocessing_distributed,
                args.rank,
                args.arch,
                args.saving_dir,
                args.lr,
                args.weight_decay,
                args.batch_size,
                args.seed,
            )

            # TIMER
            end = time.time()
            print(f"time for epoch + logging {epoch}: {end - begin}")

        # OPTIONALLY TEST AFTER EACH IMP ROUND

        if args.test_after_train:
            test_after_training(
                model,
                test_loader,
                train_loader,
                criterion,
                logger,
                imp_iter,
                device,
                args.saving_dir,
                args.gpu,
                args.epochs,
                args.arch,
                not args.no_data_parallel,
                args.print_freq,
                args.distributed,
                args.world_size,
                args.batch_size,
                args.workers,
            )


if __name__ == "__main__":
    main()
