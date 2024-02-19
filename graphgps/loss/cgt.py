import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss
import numpy as np


@register_loss('cgt_losses')
def cgt_losses(pred, true, loss_reg):
    if cfg.model.loss_fun == 'cgt+l1':
        print('here')
        l1 = nn.L1Loss()
        loss = l1(pred, true) + cfg.cgt.lamb * loss_reg
        return loss, pred
    elif cfg.model.loss_fun == 'cgt+smoothl1':
        smoothl1 = nn.SmoothL1Loss()
        loss = smoothl1(pred, true) + cfg.cgt.lamb * loss_reg
        return loss, pred
    
    elif cfg.model.loss_fun == 'cgt+cross_entropy':
        bce_loss = nn.BCEWithLogitsLoss()
        is_labeled = true == true  # Filter our nans.
        loss = bce_loss(pred[is_labeled], true[is_labeled].float()) + cfg.cgt.lamb * loss_reg
        return loss, pred
    
    elif cfg.model.loss_fun == 'cgt+weighted_cross_entropy':
        # calculating label weights for weighted loss computation
        V = true.size(0)
        n_classes = pred.shape[1] if pred.ndim > 1 else 2
        label_count = torch.bincount(true)
        label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
        cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
        cluster_sizes[torch.unique(true)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        # multiclass
        if pred.ndim > 1:
            pred = F.log_softmax(pred, dim=-1)
            loss = F.nll_loss(pred, true, weight=weight) + cfg.cgt.lamb * loss_reg
            return loss, pred
        # binary
        else:
            loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                      weight=weight[true]) + cfg.cgt.lamb * loss_reg
            return loss, torch.sigmoid(pred)
        
        
def soft_cgt_loss(self ,adj, A):
        margin = cfg.cgt.margin
        n = adj.shape[2]
        adj = adj.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        deg = adj.sum(-1)
        deg = deg.unsqueeze(3).repeat(1,1,1,n)
        # flatten the tensor
        deg = deg.view(-1)
        adj = adj.view(-1)
        A = A.view(-1)
        # get the indices of the non-zero elements of the adj matrix
        indices = torch.nonzero(adj)
        tgt_attn = A[indices].squeeze()
        # get the constraint vector
        cstr = torch.ones_like(deg) / deg
        cstr[cstr==np.inf] = 0
        cstr = (cstr - margin)[indices].squeeze()
        # compute loss
        loss_reg = nn.functional.relu(cstr - tgt_attn)
        return loss_reg
    
