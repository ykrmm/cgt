from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_cgt')
def set_cfg_wandb(cfg):
    """
    CGT configuration
    """

    # CGT group
    cfg.cgt = CN()

    # Use CGT regularisation or not
    cfg.cgt.use = False

    # Use the soft reg (i.e only on non-zero elements of the adj matrix)
    cfg.cgt.mode = 'soft'

    # Margin constraint for the loss
    cfg.cgt.margin = 0.1

    # weight factor between the task loss and the regularisation loss
    cfg.cgt.lamb = 1.
    
    # Batch aggregation methos
    cfg.cgt.agg = 'mean'
    
    # Constraint only the first layer
    cfg.cgt.first = False