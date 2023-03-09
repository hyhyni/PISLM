# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F

def own_cross_loss(ref_scores, pseudo_labels):

    logsoftmax_output = F.log_softmax(ref_scores, dim=1)
    mask1 = torch.zeros(logsoftmax_output.size()).cuda()
    for i in range(logsoftmax_output.size()[0]):
        mask1[i][pseudo_labels[i]] = 1
    p = torch.masked_select(logsoftmax_output, mask1.bool())
    return p
