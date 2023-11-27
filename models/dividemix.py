import gc
import sys
import torch
import torch.nn.functional as F
import numpy as np


# Training
def train(args, epoch, net, net2, optimizer, loader, loader_u, criterion,
          device="cpu"):
    net.train()
    net2.eval()  # fix one network and train the other

    r_lb = float(len(loader))/(len(loader)+len(loader_u))
    n = np.log10(args.num_class)
    ada_w = args.num_class*args.m*pow(
        r_lb, -(2. - (n/2) + pow(args.m, 2)*np.log10(r_lb)))
    loader_u_iter = iter(loader_u)
    num_iter = (len(loader.dataset)//args.batch_size) + 1

    for batch_idx, (im_x, im_x2, labels_x, w_x) in enumerate(loader):
        try:
            im_w, im_w2 = loader_u_iter.next()
        except StopIteration:
            loader_u_iter = iter(loader_u)
            im_w, im_w2 = loader_u_iter.next()
        batch_size = im_x.size(0)

        labels_x = torch.zeros(
            batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        im_x = im_x.to(device, non_blocking=True)
        im_x2 = im_x2.to(device, non_blocking=True)
        labels_x = labels_x.to(device, non_blocking=True)
        w_x = w_x.to(device, non_blocking=True)
        im_w = im_w.to(device, non_blocking=True)
        im_w2 = im_w2.to(device, non_blocking=True)

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            logits_u11, logits_u12 = net(im_w), net(im_w2)
            logits_u21, logits_u22 = net2(im_w), net2(im_w2)

            pu = (torch.softmax(logits_u11, dim=1) +
                  torch.softmax(logits_u12, dim=1) +
                  torch.softmax(logits_u21, dim=1) +
                  torch.softmax(logits_u22, dim=1)) / 4
            # del logits_u11, logits_u12, logits_u21, logits_u22
            # torch.cuda.empty_cache()

            # temparature sharpening and normalize
            ptu = pu**(1/args.T)
            targets_u = (ptu/ptu.sum(dim=1, keepdim=True)).detach()

            # label refinement of labeled samples
            logits_x, logits_x2 = net(im_x), net(im_x2)

            px = (torch.softmax(logits_x, dim=1) +
                  torch.softmax(logits_x2, dim=1)) / 2
            px = w_x*labels_x + (1 - w_x)*px

            # temparature sharpening and normalize
            ptx = px**(1/args.T)
            targets_x = (ptx/ptx.sum(dim=1, keepdim=True)).detach()

        # mixmatch
        lam = np.random.beta(args.alpha, args.alpha)
        lam = max(lam, 1 - lam)

        all_inputs = torch.cat([im_x, im_x2, im_w, im_w2], dim=0)
        all_targets = torch.cat(
            [targets_x, targets_x, targets_u, targets_u], dim=0)

        del im_x, im_x2, im_w, im_w2, targets_x, targets_u
        torch.cuda.empty_cache()

        idx = torch.randperm(all_inputs.size(0))
        # all_inputs = lam*all_inputs + (1 - lam)*all_inputs[idx, :]
        # all_targets = lam*all_targets + (1 - lam)*all_targets[idx, :]
        # all_inputs, all_targets = \
        #     all_inputs[:batch_size*2], all_targets[:batch_size*2]

        # logits = net(all_inputs)
        # del all_inputs
        # torch.cuda.empty_cache()

        inputs_a, inputs_b = all_inputs, all_inputs[idx]
        del all_inputs
        torch.cuda.empty_cache()
        targets_a, targets_b = all_targets, all_targets[idx]
        del all_targets
        torch.cuda.empty_cache()

        # mixed_input = lam*inputs_a[:batch_size*2] + \
        #     (1 - lam)*inputs_b[:batch_size*2]
        # mixed_target = lam*targets_a[:batch_size*2] + \
        #     (1 - lam)*targets_b[:batch_size*2]
        mixed_input = lam*inputs_a + (1 - lam)*inputs_b
        mixed_target = lam*targets_a + (1 - lam)*targets_b
        del inputs_a, inputs_b, targets_a, targets_b
        torch.cuda.empty_cache()

        logits = net(mixed_input)
        del mixed_input
        torch.cuda.empty_cache()

        # sup_loss = -torch.mean(
        #     torch.sum(F.log_softmax(logits, dim=1)*mixed_target, dim=1))
        sup_loss, unsup_loss, lamb = criterion(
            logits[:batch_size*2], mixed_target[:batch_size*2],
            logits[batch_size*2:], mixed_target[batch_size*2:],
            epoch, args.warmup
        )

        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.to(device)
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        if args.method == 'dividemix':
            loss = sup_loss + penalty
        else:
            loss = sup_loss + lamb*ada_w*unsup_loss + penalty

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write(
            '\r%s: | Epoch [%3d/%3d] Iter[%4d/%4d] | sup_loss:%.2f'
            % (args.dataset, epoch, args.num_epochs,
               batch_idx+1, num_iter, sup_loss.item()))
        sys.stdout.flush()

        del sup_loss, penalty, loss
        gc.collect(), torch.cuda.empty_cache()
