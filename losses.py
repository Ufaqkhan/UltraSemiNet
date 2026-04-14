import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, features, labels):
        # Deprecated: SAT uses custom loss loop
        pass

class PCMLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, feats, probs, proto_0, proto_1, mask):
        """
        feats: (B, D, H, W)
        probs: (B, 1, H, W)
        proto_0/1: (1, D)
        mask: (B, 1, H, W)
        """
        if mask.sum() == 0:
            return torch.tensor(0.0).to(feats.device)
            
        feats = F.normalize(feats, p=2, dim=1)
        
        sim_0 = torch.sum(feats * proto_0.view(1, -1, 1, 1), dim=1, keepdim=True)
        sim_1 = torch.sum(feats * proto_1.view(1, -1, 1, 1), dim=1, keepdim=True)
        
        pi_1 = probs
        pi_0 = 1 - probs
        
        loss_0 = pi_0 * (F.relu(self.margin - sim_0 + sim_1))
        loss_1 = pi_1 * (F.relu(self.margin - sim_1 + sim_0))
        
        loss = loss_0 + loss_1
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        
        return loss
