import os
import gc
import sys
import torch
import torch.nn as nn
import torchnet
import torchvision.models as models
from networks.InceptionResNetV2 import InceptionResNetV2
from sklearn.mixture import GaussianMixture


def warmup(args, epoch, net, optimizer, dataloader, criterion,
           device='cpu', conf_penalty=None):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, _, _) in enumerate(dataloader):
        inputs, labels = inputs.to(device), \
            labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = net(inputs)
        # del inputs
        # torch.cuda.empty_cache()

        loss = criterion(outputs, labels)
        if conf_penalty is not None:
            penalty = conf_penalty(outputs)
            L = loss + penalty
        else:
            L = loss
        L.backward()
        optimizer.step()

        sys.stdout.write('\r%s: | Epoch [%3d/%3d] Iter[%4d/%4d] loss: %.4f' % (
            args.dataset, epoch, args.num_epochs, batch_idx + 1,
            num_iter, L.item()))
        sys.stdout.flush()


def val(net, val_loader, best, path, k=1, device='cpu'):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = net(inputs)
            _, pred = torch.max(outputs, 1)

            total += targets.size(0)
            correct += pred.eq(targets).cpu().sum().item()
    acc = 100.*correct/total
    if acc > best:
        best = acc
        save_pth = os.path.join(path, "model{}_best.pth".format(k))
        torch.save(net.state_dict(), save_pth)
    return acc, best


def test(args, epoch, net1, net2, data_loader, device='cpu', queue=None):
    acc_meter = torchnet.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    acc_meter.reset()
    net1.eval(), net2.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), \
                targets.to(device, non_blocking=True)
            outputs1, outputs2 = net1(inputs), net2(inputs)
            outputs = outputs1 + outputs2
            acc_meter.add(outputs, targets)
    acc = acc_meter.value()

    if queue is None:
        return acc
    else:
        queue.put(acc)


def eval_train(args, model, criterion, data_loader,
               tol=1e-2, reg_covar=5e-4, device='cpu', queue=None):
    model.eval()
    if args.dataset == 'webvision':
        losses = torch.zeros(len(data_loader.dataset))
    else:
        num_samples = args.num_batches*args.batch_size
        losses = torch.zeros(num_samples)
    paths, n = [], 0
    with torch.no_grad():
        for inputs, targets, index, path in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            del inputs
            gc.collect(), torch.cuda.empty_cache()

            loss = criterion(outputs, targets)
            if args.dataset == 'webvision':
                for b in range(outputs.size(0)):
                    losses[index[b]] = loss[b]
            else:
                for b in range(outputs.size(0)):
                    losses[n] = loss[b]
                    paths.append(path[b])
                    n += 1
    losses = (losses-losses.min())/(losses.max()-losses.min())
    losses = losses.reshape(-1, 1)

    # fit a two-component GMM to the loss
    gmm = GaussianMixture(
        n_components=2, max_iter=10, tol=tol, reg_covar=reg_covar).fit(losses)
    prob = gmm.predict_proba(losses)
    prob = prob[:, gmm.means_.argmin()]
    if queue is None:
        return prob, paths
    else:
        queue.put({"prob": prob, "path": paths})


def create_model(args, device='cpu'):
    if args.dataset == 'webvision':
        model = InceptionResNetV2(num_classes=args.num_class)
    else:
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, args.num_class)
    model = model.to(device)
    return model
