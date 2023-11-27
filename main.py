from __future__ import print_function

import os
import gc
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from datetime import datetime
import argparse

from models.loss import SemiLoss, NegEntropy
from util import save_args, save_checkpoint, make_logdir, \
    load_args, seed_setting, adjust_learningrate
from models.base_train import warmup, test, eval_train, create_model, val


def build_meta_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--src', type=str)
    parser.add_argument('--dst', type=str)
    return parser


def build_parser(meta_parser):
    parser = argparse.ArgumentParser(parents=[meta_parser],
                                     description='PyTorch Training')
    parser.add_argument('-r', '--resume', type=str,
                        help='path to latent checkpoint')
    parser.add_argument('-m', '--method', type=str, default='confidentmix',
                        choices=['dividemix', 'confidentmix'],
                        help='method name')
    parser.add_argument('-n', '--name', default='_Proposed_')
    parser.add_argument('-s', '--save_dir', default='./saved/Proposed',
                        type=str, help='save directory path')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                        help='initial learning rate')
    parser.add_argument('--decay', '--weight_decay', default=5e-4, type=float)
    parser.add_argument('--milestones', default=50, type=int)
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='parameter for Beta')
    parser.add_argument('--lambda_u', default=0, type=float,
                        help='weight for unsupervised loss')
    parser.add_argument('--p_threshold', default=0.5, type=float,
                        help='clean probability threshold')
    parser.add_argument('--T', default=0.5, type=float,
                        help='sharpening temperature')
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--seed', default=123)
    parser.add_argument('--gpuid1', default=0, type=int)
    parser.add_argument('--gpuid2', default=1, type=int)
    parser.add_argument('--num_class', default=50, type=int)
    parser.add_argument('--num_batches', default=1000, type=int)
    parser.add_argument('--data_path', default='../datasets',
                        type=str, help='path to dataset')
    parser.add_argument('--dataset', default='webvision', type=str)
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['resnet50', 'InceptionResNetV2'],
                        default='InceptionResNetV2',
                        help='CNN architecture (default: PreActResNet18)')
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--warmup', default=1, type=int,
                        help='num warmup epochs')
    parser.add_argument('--tol', type=float, default=1e-2,
                        help='tol for gaussian mixture model')
    parser.add_argument('--reg_covar', type=float, default=1e-3,
                        help="reg_covar for gaussian mixture model")
    parser.add_argument('--tau', default=0.7, type=int,
                        help='threshold of label confidence')
    parser.add_argument('--m', default=0.01, type=float)
    parser.add_argument('--strong', default=False, action='store_true',
                        help='use strong augmentation')

    parser.add_argument('--save_freq', default=0, type=int,
                        help='epoch to save models parameter')
    parser.add_argument('--date_time', default=None, type=str,
                        help='checkpoint')
    parser.add_argument('--multiprocess', default=False, action='store_true')
    return parser


def main(args):
    # year month date _ hours minits
    date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    date_time = args.date_time if args.date_time else date_time
    args.date_time = date_time
    f_name = '{}{}{}'.format(args.dataset, args.name, args.arch)
    dir_path = os.path.join(args.save_dir, 'log', f_name, date_time)
    model_pth = os.path.join(args.save_dir, 'models', f_name, date_time)
    os.makedirs(dir_path, exist_ok=True), os.makedirs(model_pth, exist_ok=True)

    log_path = os.path.join(dir_path, "{}_".format(args.dataset))
    state_log1 = log_path + "states1.txt"
    state_log2 = log_path + "states2.txt"
    test_log = log_path + "acc.txt"
    state_txt = 'Epoch Labeled_sample_ratio\n'
    if args.dataset == 'webvision':
        test_txt = 'Epoch Accuracy top-5 val_acc val_acc5\n'
    else:
        test_txt = 'Epoch Accuracy top-5 val_acc1 val_acc2\n'

    # make directory
    make_logdir(state_log1, state_txt), make_logdir(state_log2, state_txt)
    make_logdir(test_log, test_txt)

    if args.dataset == "webvision":
        from dataloader.webvision import WebvisionDataloader as Loader
    else:
        from dataloader.clothing1m import Clothing1mDataloader as Loader
    loader = Loader(
        args.data_path, args.batch_size, args.num_workers,
        num_class=args.num_class, num_batches=args.num_batches,
        strong=args.strong)

    print('\n| Building net')
    if args.multiprocess:
        os.environ["CUDA_VISIBLE_DEVICES"] = '{},{}'.format(
            args.gpuid1, args.gpuid2)
        mp.set_start_method('spawn')
        device1 = torch.device('cuda:{}'.format(args.gpuid1))
        device2 = torch.device('cuda:{}'.format(args.gpuid2))
        net1_clone = create_model(args, device=device2)
        net2_clone = create_model(args, device=device1)
    else:
        torch.cuda.set_device(args.gpuid1)
        device1, device2 = args.gpuid1, args.gpuid1
    net1 = create_model(args, device=device1)
    net2 = create_model(args, device=device2)

    # epoch = 1
    epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net1.load_state_dict(checkpoint['state_dict1'])
        net2.load_state_dict(checkpoint['state_dict2'])
        args.warmup = checkpoint['epoch']
        epoch = args.warmup + 1

    criterion = SemiLoss()
    optimizer1 = optim.SGD(
        net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    optimizer2 = optim.SGD(
        net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

    eval_loss = nn.CrossEntropyLoss(reduction='none')
    ce_loss = nn.CrossEntropyLoss()

    conf_penalty = NegEntropy() if args.dataset == 'clothing1m' else None

    test_loader = loader.run('test')
    val_loader = loader.run('val')
    eval_loader = loader.run('eval_train')

    best_acc = 0.
    val_best = [0., 0.]

    if args.method == 'confidentmix':
        from models.confidentmix import train
    else:
        from models.dividemix import train

    # start time count
    start_time = time.perf_counter()

    if args.resume:
        if args.multiprocess:
            eval_loader1 = loader.run('eval_train')
            eval_loader2 = loader.run('eval_train')
            q1, q2 = mp.Queue(), mp.Queue()
            p1 = mp.Process(
                    target=eval_train,
                    args=(args, net1, eval_loss, eval_loader1, args.tol,
                          args.reg_covar, device1, q1))
            p2 = mp.Process(
                    target=eval_train,
                    args=(args, net2, eval_loss, eval_loader2, args.tol,
                          args.reg_covar, device2, q2))
            p1.start(), p2.start()
            out1, out2 = q1.get(), q2.get()
            prob1, path1 = out1['prob'], out1['path']
            prob2, path2 = out2['prob'], out2['path']
            p1.join(), p2.join()
        else:
            prob1, path1 = eval_train(
                args, net1, eval_loss, eval_loader,
                tol=args.tol, reg_covar=args.reg_covar, device=device1)
            prob2, path2 = eval_train(
                args, net2, eval_loss, eval_loader,
                tol=args.tol, reg_covar=args.reg_covar, device=device2)

    # for epoch in range(args.num_epochs+1):
    while epoch < args.num_epochs + 1:
        lr = args.lr
        adjust_learningrate(
            epoch, lr, optimizer1, optimizer2, milestones=args.milestones)

        if args.multiprocess:
            if epoch < args.warmup:
                warmup_loader1 = loader.run('warmup')
                warmup_loader2 = loader.run('warmup')
                p1 = mp.Process(
                    target=warmup,
                    args=(args, epoch, net1, optimizer1, warmup_loader1,
                          ce_loss, device1, conf_penalty))
                p2 = mp.Process(
                    target=warmup,
                    args=(args, epoch, net2, optimizer2, warmup_loader2,
                          ce_loss, device2, conf_penalty))
                p1.start(), p2.start()
            else:
                pred1 = (prob1 > args.p_threshold)
                pred2 = (prob2 > args.p_threshold)

                loader_x1, loader_u1 = loader.run(
                    'train', pred2, prob2, epoch, state_log1, path2)
                loader_x2, loader_u2 = loader.run(
                    'train', pred1, prob1, epoch, state_log2, path1)

                p1 = mp.Process(
                    target=train,
                    args=(args, epoch, net1, net2_clone, optimizer1,
                          loader_x1, loader_u1, criterion, device1))
                p2 = mp.Process(
                    target=train,
                    args=(args, epoch, net2, net1_clone, optimizer2,
                          loader_x2, loader_u2, criterion, device2))
                p1.start(), p2.start()
            p1.join(), p2.join()

            net1_clone.load_state_dict(net1.state_dict())
            net2_clone.load_state_dict(net2.state_dict())

            q1, q2 = mp.Queue(), mp.Queue()
            p1 = mp.Process(
                target=test,
                args=(args, epoch, net1, net2_clone, test_loader, device1, q1))
            p2 = mp.Process(
                target=test,
                args=(args, epoch, net1_clone, net2, val_loader, device2, q2))
            p1.start(), p2.start()

            test_acc, val_acc = q1.get(), q2.get()
            p1.join(), p2.join()
            if args.dataset == 'webvision':
                print("\n| Test Epoch #{} WebVision Acc: ".format(epoch) +
                      "{:.2f}%({:.2f}%)".format(test_acc[0], test_acc[1]) +
                      " ImageNet Acc: {:.2f}({:.2f}%)\n".format(
                        val_acc[0], val_acc[1]))
            else:
                print("\n| Test Epoch #{} Test Acc: {:.2f}%({:.2f}%)".format(
                        epoch, test_acc[0], test_acc[1]) +
                      " Val Acc: {:.2f}%({:.2f}%)\n".format(
                        val_acc[0], val_acc[1]))

            eval_loader1 = loader.run('eval_train')
            eval_loader2 = loader.run('eval_train')
            q1, q2 = mp.Queue(), mp.Queue()
            p1 = mp.Process(
                    target=eval_train,
                    args=(args, net1, eval_loss, eval_loader1, args.tol,
                          args.reg_covar, device1, q1))
            p2 = mp.Process(
                    target=eval_train,
                    args=(args, net2, eval_loss, eval_loader2, args.tol,
                          args.reg_covar, device2, q2))
            p1.start(), p2.start()
            out1, out2 = q1.get(), q2.get()
            prob1, path1 = out1['prob'], out1['path']
            prob2, path2 = out2['prob'], out2['path']
            p1.join(), p2.join()
        else:
            if epoch < args.warmup:
                warmup_loader = loader.run('warmup')
                print('Warmup Net1')
                warmup(args, epoch, net1, optimizer1, warmup_loader, ce_loss,
                       device=device1, conf_penalty=conf_penalty)
                print('\nWarmup Net2')
                warmup_loader = loader.run('warmup')
                warmup(args, epoch, net2, optimizer2, warmup_loader, ce_loss,
                       device=device2, conf_penalty=conf_penalty)
                if epoch == args.warmup:
                    del warmup_loader
                    gc.collect()

            else:
                pred1 = (prob1 > args.p_threshold)
                pred2 = (prob2 > args.p_threshold)

                print('\nTrain Net1')
                loader_x, loader_u = loader.run(
                    'train', pred2, prob2, epoch, log=state_log1, path=path2)
                print("Labeled/Unlabeled data: [{}/{}]".format(
                    len(loader_x.dataset), len(loader_u.dataset)))
                train(args, epoch, net1, net2, optimizer1,
                      loader_x, loader_u, criterion, device=device1)

                print('\nTrain Net2')
                loader_x, loader_u = loader.run(
                    'train', pred1, prob1, epoch, log=state_log2, path=path1)
                print("Labeled/Unlabeled data: [{}/{}]".format(
                    len(loader_x.dataset), len(loader_u.dataset)))
                train(args, epoch, net2, net1, optimizer2,
                      loader_x, loader_u, criterion, device=device2)

            test_acc = test(
                args, epoch, net1, net2, test_loader, device=device1)
            # val_acc = test(args, epoch, net1, net2, val_loader, device=device2)
            acc1, val_best[0] = val(
                net1, val_loader, val_best[0], model_pth, k=1, device=device1)
            acc2, val_best[1] = val(
                net2, val_loader, val_best[1], model_pth, k=2, device=device2)
            val_acc = [acc1, acc2]
            if args.dataset == 'webvision':
                print("\n| Test Epoch #{} WebVision Acc: ".format(epoch) +
                      "{:.2f}%({:.2f}%)".format(test_acc[0], test_acc[1]) +
                      " ImageNet Acc: {:.2f}({:.2f}%)\n".format(
                        val_acc[0], val_acc[1]))
            else:
                print("\n| Test Epoch #{} Test Acc: {:.2f}%({:.2f}%)".format(
                        epoch, test_acc[0], test_acc[1]) +
                      " Val Acc: {:.2f}%({:.2f}%)\n".format(
                        val_acc[0], val_acc[1]))

            prob1, path1 = eval_train(
                args, net1, eval_loss, eval_loader,
                tol=args.tol, reg_covar=args.reg_covar, device=device1)
            prob2, path2 = eval_train(
                args, net2, eval_loss, eval_loader,
                tol=args.tol, reg_covar=args.reg_covar, device=device2)

        if os.path.isfile(test_log):
            with open(test_log, 'a') as f:
                f.write('{} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
                    epoch, test_acc[0], test_acc[1], val_acc[0], val_acc[1]))
        save_checkpoint(args, epoch, net1, net2, optimizer1, optimizer2,
                        test_acc[0], save_dir=model_pth)
        if test_acc[0] > best_acc:
            best_acc = test_acc[0]
            save_checkpoint(args, epoch, net1, net2, optimizer1, optimizer2,
                            best_acc, save_best=True, save_dir=model_pth)
        if epoch == args.warmup - 1 or\
                (args.save_freq != 0 and epoch % args.save_freq == 0):
            save_checkpoint(args, epoch, net1, net2, optimizer1, optimizer2,
                            test_acc[0], save_ckpt=True, save_dir=model_pth)
        epoch += 1

    end_time = time.perf_counter()
    args.total_time = "{:.2f}".format((end_time - start_time)/3600.)
    print(str(args.total_time) + "h")

    if args.dataset == 'clothing1m':
        net1.load_state_dict(
            torch.load(os.path.join(model_pth, 'model1_best.pth')))
        net2.load_state_dict(
            torch.load(os.path.join(model_pth, 'model2_best.pth')))
        acc = test(args, epoch, net1, net2, test_loader, device=device1)
        with open(test_log, 'a') as f:
            f.write('\n  {:.2f} {:.2f}\n'.format(acc[0], acc[1]))

    save_args(args, os.path.join(dir_path, 'config.json'))


if __name__ == '__main__':
    meta_parser = build_meta_parser()
    parser = build_parser(meta_parser)

    # parser --src and --dst
    margs, rest = meta_parser.parse_known_args()
    namespace = load_args(margs.src)

    args = parser.parse_args(args=rest, namespace=namespace)
    if margs.dst is None:
        margs.dst = args.save_dir

    seed_setting(args.seed)
    main(args)
