import numpy as np
import torch
import torch.nn.functional as F


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


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
