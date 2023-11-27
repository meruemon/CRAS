import torch.nn.functional as F
import torch
import torch.nn as nn
from utils import sigmoid_rampup


def cross_entropy(output, target, M=3):
    return F.cross_entropy(output, target)


class elr_plus_loss(nn.Module):
    def __init__(self, num_examp, config, device, num_classes=10, beta=0.3):
        super(elr_plus_loss, self).__init__()
        self.config = config
        self.pred_hist = (torch.zeros(num_examp, num_classes)).to(device)
        self.q = 0
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, iteration, output, y_labeled, class_acc, p_cutoff):
        y_pred = F.softmax(output, dim=1)

        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)

        if self.num_classes == 100:
            y_labeled = y_labeled*self.q
            y_labeled = y_labeled/(y_labeled).sum(dim=1, keepdim=True)

        ce_loss = torch.mean(
            -torch.sum(y_labeled * F.log_softmax(output, dim=1), dim=-1))
        reg = ((1-(self.q * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + \
            sigmoid_rampup(iteration, self.config['coef_step']) *\
            (self.config['train_loss']['args']['lambda']*reg)

        return final_loss, y_pred.cpu().detach(), None, None, None

    def update_hist(self, epoch, out, index=None, mix_index=..., mixup_l=1):
        y_pred_ = F.softmax(out, dim=1)
        self.pred_hist[index] = \
            self.beta*self.pred_hist[index] + \
            (1-self.beta)*y_pred_/(y_pred_).sum(dim=1, keepdim=True)
        self.q = mixup_l*self.pred_hist[index] + \
            (1-mixup_l)*self.pred_hist[index][mix_index]


class CurriculumRegularizationLoss(nn.Module):
    def __init__(self, num_examp, config, device, num_classes=10, beta=0.3):
        super(CurriculumRegularizationLoss, self).__init__()
        self.config = config
        self.pred_hist = (torch.zeros(num_examp, num_classes)).to(device)
        self.q = 0
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, iteration, output, y_labeled, class_acc, p_cutoff):
        y_pred = F.softmax(output, dim=1)

        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)

        if self.num_classes == 100:
            y_labeled = y_labeled * self.q
            y_labeled = y_labeled / (y_labeled).sum(dim=1, keepdim=True)

        with torch.no_grad():
            ptu = self.t / self.t.sum(dim=-1, keepdim=True)
            max_probs, max_idx = torch.max(ptu, dim=-1)

            # convex
            maxavg_cutoff = torch.mean(max_probs)
            mask = max_probs.ge(
                p_cutoff*(class_acc[max_idx]/(2. - class_acc[max_idx]))
                ).float()
            select = max_probs.ge(maxavg_cutoff).long()

        ce_loss = -torch.sum(y_labeled * F.log_softmax(output, dim=1), dim=-1)
        reg = (1 - (self.q * y_pred).sum(dim=1)).log()
        masked_reg = (mask * reg).mean()

        final_loss = torch.mean(ce_loss) + \
            sigmoid_rampup(iteration, self.config['coef_step']) * \
            (self.config['train_loss']['args']['lambda'] * masked_reg)

        return final_loss, y_pred.cpu().detach(), mask.mean(), \
            select, max_idx.long()

    def update_hist(self, epoch, out, index=None, mix_index=..., mixup_l=1):
        y_pred_ = F.softmax(out, dim=1)
        self.pred_hist[index] = \
            self.beta*self.pred_hist[index] + \
            (1-self.beta)*y_pred_/(y_pred_).sum(dim=1, keepdim=True)
        self.t = self.pred_hist[index]
        self.q = mixup_l*self.pred_hist[index] + \
            (1-mixup_l)*self.pred_hist[index][mix_index]
