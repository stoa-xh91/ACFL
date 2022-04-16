import torch
import torch.nn.functional as F
import torch.nn as nn


class CFMLloss(nn.Module):
    def __init__(self, num_dims=256, num_heads=4):
        super(CFMLloss, self).__init__()

        self.job_assignment = nn.MultiheadAttention(num_dims, num_heads)
        self.linear         = nn.Linear(num_dims, num_dims)
        self.distill_loss   = nn.MSELoss(reduction='mean')
        self.alpha          = 0.5

    def forward(self, feature_t, feature_s, logits_t, logits_s, beta):
        """
        Args:
            - features from various models
            - predicted logists from various model

        Shape:
            - Inputs:
            - feature_t: :math:`(L, N, E)` where L is the number of source models, N is the batch size, E is the feature dimension.
            - feature_s: :math:`(S, N, E)` where S is the number of target models, N is the batch size, E is the feature dimension.
            - logits_t:  :math:`(L, N, C)` where L is the number of source models, N is the batch size, C is the number of classes.
            - logits_s:  :math:`(S, N, C)` where S is the number of target models, N is the batch size, C is the number of classes.
            - beta: :math:`(1, 1, L)`.
        """
        # which model should be considered
        _, ori_assigns = self.job_assignment(feature_s, feature_t, feature_t)
        
        assigns = ori_assigns * beta

        # what contents needed to be mimicked 
        attens = torch.sigmoid(self.linear(feature_t - feature_s))
        attens = torch.matmul(assigns, (attens).permute(1, 0, 2).contiguous())
        
        feat_selected = torch.matmul(assigns, (feature_t).permute(1, 0, 2).contiguous())
        d_f = self.distill_loss(attens * feature_s.permute(1, 0, 2).contiguous(), attens * feat_selected)
        logit_selected = torch.matmul(assigns, logits_t.permute(1, 0, 2).contiguous())   
        d_l = self.distill_loss(logits_s, logit_selected.permute(1, 0, 2).contiguous())
        d_loss   = self.alpha * d_f + (1-self.alpha) * d_l
        return {'d_loss':d_loss,'d_feat_loss':d_f,'d_logit_loss':d_l, 'attens':ori_assigns}


