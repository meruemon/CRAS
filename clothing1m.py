from __future__ import print_function
import os
import sys
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import argparse
import numpy as np
import dataloader.clothing1m as dataloader
from sklearn.mixture import GaussianMixture
from util import seed_setting, make_logdir, save_args, \
    load_args, save_checkpoint


def build_meta_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--src', type=str)
    parser.add_argument('--dst', type=str)
    return parser


meta_parser = build_meta_parser()
parser = argparse.ArgumentParser(parents=[meta_parser],
                                 description='PyTorch Clothing1M Training')
parser.add_argument('-r', '--resume', type=str,
                    help='path to latent checkpoint')
parser.add_argument('--method', default='dividemix', type=str)
parser.add_argument('-n', '--name', default='_DivideMix_')
parser.add_argument('-s', '--save_dir', default='./saved/DivideMix')
parser.add_argument('--batch_size', default=32, type=int,
                    help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float,
                    help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float,
                    help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float,
                    help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float,
                    help='sharpening temperature')
parser.add_argument('--m', default=0.01, type=float,
                    help='adaptive weight parameter')
parser.add_argument('-a', '--arch', default='resnet50', type=str)
parser.add_argument('--num_epochs', default=80, type=int)
parser.add_argument('--num_workers', default=24, type=int)
parser.add_argument('--warmup', default=1, type=int)
parser.add_argument('--id', default='clothing1m')
parser.add_argument('--data_path', default='../datasets/clothing1m', type=str,
                    help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--num_batches', default=1000, type=int)
parser.add_argument('--date_time', default=None, type=str)

# parser --src and --dst
margs, rest = meta_parser.parse_known_args()
namespace = load_args(margs.src)
args = parser.parse_args(args=rest, namespace=namespace)
if margs.dst is None:
    margs.dst = args.save_dir

torch.cuda.set_device(args.gpuid)
seed_setting(args.seed)


# Training
def train(epoch, net, net2, optimizer,
          labeled_trainloader, unlabeled_trainloader):
    net.train()
    net2.eval()  # fix one network and train the other

    r_lb = float(len(labeled_trainloader)) / \
        (len(labeled_trainloader)+len(unlabeled_trainloader))
    n = np.log10(args.num_class)
    ada_w = args.num_class*args.m*pow(
        r_lb, -(2. - (n/2) + pow(args.m, 2)*np.log10(r_lb)))
    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in \
            enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(
            batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, labels_x, w_x = \
            inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)

            pu = (torch.softmax(outputs_u11, dim=1) +
                  torch.softmax(outputs_u12, dim=1) +
                  torch.softmax(outputs_u21, dim=1) +
                  torch.softmax(outputs_u22, dim=1)) / 4
            ptu = pu**(1/args.T)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)

            px = (torch.softmax(outputs_x, dim=1) +
                  torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px
            ptx = px**(1/args.T)  # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)

        all_inputs = torch.cat(
            [inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat(
            [targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        # mixed_input = l*input_a[:batch_size*2] + (1 - l)*input_b[:batch_size*2]
        # mixed_target = l*target_a[:batch_size*2] +\
        #     (1 - l)*target_b[:batch_size*2]
        mixed_input = l*input_a + (1 - l)*input_b
        mixed_target = l*target_a + (1 - l)*target_b

        logits = net(mixed_input)

        # Lx = -torch.mean(
        #     torch.sum(F.log_softmax(logits, dim=1)*mixed_target, dim=1))
        Lx, Lu, lamb = criterion(
            logits[:batch_size*2], mixed_target[:batch_size*2],
            logits[batch_size*2:], mixed_target[batch_size*2:],
            epoch, args.warmup)

        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        if args.method == 'dividemix':
            loss = Lx + penalty
        else:
            loss = Lx + ada_w*lamb*Lu + penalty

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write(
            '\rClothing1M | Epoch[%3d/%3d] Iter[%3d/%3d] Labeled loss:%.4f'
            % (epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
        sys.stdout.flush()


def warmup(net, optimizer, dataloader):
    net.train()
    for batch_idx, (inputs, labels, _, _) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)

        penalty = conf_penalty(outputs)
        L = loss + penalty
        L.backward()
        optimizer.step()

        sys.stdout.write(
            '\r|Warm-up: Iter[%3d/%3d] CE-loss:%.4f Conf-Penalty:%.4f'
            % (batch_idx+1, args.num_batches, loss.item(), penalty.item()))
        sys.stdout.flush()


def val(net, val_loader, k):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100.*correct/total
    print("\n| Validation\t Net%d  Acc: %.2f%%" % (k, acc))
    if acc > best_acc[k-1]:
        best_acc[k-1] = acc
        print('| Saving Best Net%d ...' % k)
        save_point = './checkpoint/%s_net%d.pth.tar' % (args.id, k)
        torch.save(net.state_dict(), save_point)
    return acc


def test(net1, net2, test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100.*correct/total
    print("\n| Test Acc: %.2f%%\n" % (acc))
    return acc


def eval_train(epoch, model):
    model.eval()
    num_samples = args.num_batches*args.batch_size
    losses = torch.zeros(num_samples)
    paths = []
    n = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _, path) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[n] = loss[b]
                paths.append(path[b])
                n += 1
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter %3d\t' % (batch_idx))
            sys.stdout.flush()

    losses = (losses-losses.min())/(losses.max()-losses.min())
    losses = losses.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=2, max_iter=10, reg_covar=5e-4, tol=1e-2)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses)
    prob = prob[:, gmm.means_.argmin()]
    return prob, paths


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up)/rampup_length, 0.0, 1.0)
    return float(current)


class SemiLoss(object):
    def __call__(self, logits_x, targets_x, outputs_u, targets_u,
                 epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        sup_loss = -torch.mean(
            torch.sum(F.log_softmax(logits_x, dim=1)*targets_x, dim=1))
        unsup_loss = torch.mean((probs_u - targets_u)**2)

        return sup_loss, unsup_loss, linear_rampup(epoch, warm_up)


def create_model():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, args.num_class)
    model = model.cuda()
    return model


# year month date _ hours minits
date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
date_time = args.date_time if args.date_time else date_time
args.date_time = date_time
f_name = '{}{}{}'.format(args.id, args.name, args.arch)
dir_path = os.path.join(args.save_dir, 'log', f_name, date_time)
model_pth = os.path.join(args.save_dir, 'models', f_name, date_time)
os.makedirs(dir_path, exist_ok=True), os.makedirs(model_pth, exist_ok=True)

log_path = os.path.join(dir_path, "{}_".format(args.id))
state_log1 = log_path + "states1.txt"
state_log2 = log_path + "states2.txt"
test_log = log_path + "acc.txt"
state_txt = 'Epoch Labeled_sample_ratio\n'
test_txt = 'Epoch val_acc1 val_acc2\n'

# make directory
make_logdir(state_log1, state_txt), make_logdir(state_log2, state_txt)
make_logdir(test_log, test_txt)

loader = dataloader.Clothing1mDataloader(
    args.data_path, args.batch_size, args.num_workers,
    num_batches=args.num_batches)

print('| Building net')
net1 = create_model()
net2 = create_model()

epoch = 0
if args.resume:
    checkpoint = torch.load(args.resume, map_location='cpu')
    net1.load_state_dict(checkpoint['state_dict1'])
    net2.load_state_dict(checkpoint['state_dict2'])
    args.warmup = checkpoint['epoch']
    epoch = args.warmup + 1

optimizer1 = optim.SGD(
    net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(
    net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()
criterion = SemiLoss()

# start time count
start_time = time.perf_counter()

if epoch > args.warmup - 1:
    print('\n==== net 1 evaluate next epoch training data loss ====')
    eval_loader = loader.run('eval_train')
    prob1, paths1 = eval_train(epoch, net1)
    print('\n==== net 2 evaluate next epoch training data loss ====')
    eval_loader = loader.run('eval_train')
    prob2, paths2 = eval_train(epoch, net2)

best_acc = [0, 0]
while epoch < args.num_epochs + 1:
    lr = args.lr
    if epoch >= 40:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr

    if epoch < args.warmup:  # warm up
        train_loader = loader.run('warmup')
        print('Warmup Net1')
        warmup(net1, optimizer1, train_loader)
        train_loader = loader.run('warmup')
        print('\nWarmup Net2')
        warmup(net2, optimizer2, train_loader)
    else:
        pred1 = (prob1 > args.p_threshold)  # divide dataset
        pred2 = (prob2 > args.p_threshold)

        print('\n\nTrain Net1')
        labeled_trainloader, unlabeled_trainloader =\
            loader.run(
                'train', pred2, prob2, epoch, log=state_log1, path=paths2)
        train(epoch, net1, net2, optimizer1,
              labeled_trainloader, unlabeled_trainloader)  # train net1
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader =\
            loader.run(
                'train', pred1, prob1, epoch, log=state_log2, path=paths1)
        train(epoch, net2, net1, optimizer2,
              labeled_trainloader, unlabeled_trainloader)  # train net2

    val_loader = loader.run('val')  # validation
    acc1 = val(net1, val_loader, 1)
    acc2 = val(net2, val_loader, 2)
    if os.path.isfile(test_log):
        with open(test_log, 'a') as f:
            f.write('{} {:.2f} {:.2f}\n'.format(epoch, acc1, acc2))
    print('\n==== net 1 evaluate next epoch training data loss ====')
    # evaluate training data loss for next epoch
    eval_loader = loader.run('eval_train')
    prob1, paths1 = eval_train(epoch, net1)
    print('\n==== net 2 evaluate next epoch training data loss ====')
    eval_loader = loader.run('eval_train')
    prob2, paths2 = eval_train(epoch, net2)

    save_checkpoint(args, epoch, net1, net2, optimizer1, optimizer2,
                    best_acc, save_dir=model_pth)
    epoch += 1

test_loader = loader.run('test')
net1.load_state_dict(torch.load('./checkpoint/%s_net1.pth.tar' % args.id))
net2.load_state_dict(torch.load('./checkpoint/%s_net2.pth.tar' % args.id))
acc = test(net1, net2, test_loader)
if os.path.isfile(test_log):
    with open(test_log, 'a') as f:
        f.write('Test_acc {:.2f}\n'.format(acc))

end_time = time.perf_counter()
args.total_time = "{:.2f}".format((end_time - start_time)/3600.)
print(str(args.total_time) + 'h')

save_args(args, os.path.join(dir_path, 'config.json'))
