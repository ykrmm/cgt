import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
import numpy as np



def cgt_loss(adj, attn):
        if cfg.cgt.mode == 'soft':
                return soft_cgt_loss(adj, attn)
        elif cfg.cgt.mode == 'quantity':
                return quantity_loss(adj, attn)
        elif cfg.cgt.mode == 'min_attn':
                return min_attn(adj, attn)
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

def min_attn(adj,attn):
        adj = adj.unsqueeze(1).repeat(1, cfg.gt.n_heads, 1, 1)
        # flatten the tensor
        adj = adj.view(-1)
        attn = attn.view(-1)
        # get the indices of the non-zero elements of the adj matrix
        indices = torch.nonzero(adj)
        tgt_attn = attn[indices].squeeze()
        # compute loss
        loss_reg = - tgt_attn.mean()
        return loss_reg

def quantity_loss(adj,attn):
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
        deg = deg[indices].squeeze()
        tgt_attn = tgt_attn.reshape((cfg.gt.n_heads,int(len(indices)/cfg.gt.n_heads)))
        deg = deg.reshape((cfg.gt.n_heads,int(len(indices)/cfg.gt.n_heads)))
        # get the constraint vector
        
        cstr = torch.ones_like(tgt_attn) / deg
        cstr[cstr==np.inf] = 0
        
        cstr = (cstr - margin)
        # compute loss
        loss_reg = nn.functional.relu(cstr.sum(dim=1) - tgt_attn.sum(dim=1)).mean()
        return loss_reg
    
