import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
import numpy as np



def cgt_loss(adj, attn):
        if cfg.cgt.mode == 'soft':
                return soft_cgt_loss(adj, attn)
        elif cfg.cgt.mode == 'quantity':
                return quantity_loss(adj, attn)
        else:
                raise ValueError(f"Unexpected mode: {cfg.cgt.mode}")
        
def soft_cgt_loss(adj, attn):
        margin = cfg.cgt.margin
        n = adj.shape[2]
        adj = adj.unsqueeze(1).repeat(1, cfg.gt.n_heads, 1, 1)
        deg = adj.sum(-1)
        deg = deg.unsqueeze(3).repeat(1,1,1,n)
        # flatten the tensor
        deg = deg.view(-1)
        adj = adj.view(-1)
        attn = attn.view(-1)
        # get the indices of the non-zero elements of the adj matrix
        indices = torch.nonzero(adj)
        tgt_attn = attn[indices].squeeze()
        # get the constraint vector
        cstr = torch.ones_like(deg) / deg
        cstr[cstr==np.inf] = 0
        cstr = (cstr - margin)[indices].squeeze()
        # compute loss
        loss_reg = nn.functional.relu(cstr - tgt_attn)
        # Sum ? Mean ? 
        loss_reg = loss_reg.mean()
        return loss_reg

def quantity_loss(adj,attn):
        # Il faut faire la somme par tête et moyenné
        margin = cfg.cgt.margin
        n = adj.shape[2]
        adj = adj.unsqueeze(1).repeat(1, cfg.gt.n_heads, 1, 1)
        deg = adj.sum(-1)
        deg = deg.unsqueeze(3).repeat(1,1,1,n)
        # flatten the tensor
        deg = deg.view(-1)
        adj = adj.view(-1)
        attn = attn.view(-1)
        # get the indices of the non-zero elements of the adj matrix
        indices = torch.nonzero(adj)
        tgt_attn = attn[indices].squeeze()
        # get the constraint vector
        cstr = torch.ones_like(deg) / deg
        cstr[cstr==np.inf] = 0
        cstr = (cstr - margin)[indices].squeeze()
        # compute loss
        loss_reg = nn.functional.relu(cstr.sum() - tgt_attn.sum())
        return loss_reg
    
