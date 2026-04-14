import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels):
        self.temperature.requires_grad = True
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        logits = logits.reshape(-1)
        labels = labels.reshape(-1)

        def closure():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            self.temperature.clamp_(min=0.1, max=5.0)
        print(f"Calibrated Temperature: {self.temperature.item():.4f}")

class CPSGate:
    @staticmethod
    def check_stability_and_confidence(teacher_logits, tta_preds, tau_t, delta=0.15, temp=1.0):
        prob_calib = torch.sigmoid(teacher_logits / temp)
        conf = torch.max(prob_calib, 1 - prob_calib)
        conf_mask = (conf >= tau_t)
        
        p = prob_calib
        q = tta_preds
        eps = 1e-7
        p = torch.clamp(p, eps, 1-eps)
        q = torch.clamp(q, eps, 1-eps)
        kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
        stab_mask = (kl <= delta)
        
        accepted = conf_mask & stab_mask
        pseudo_label = (prob_calib >= 0.5).float()
        return accepted.float(), pseudo_label, prob_calib

class PCM(nn.Module):
    def __init__(self, dim=64, rho=0.99):
        super().__init__()
        self.rho = rho
        self.register_buffer("proto_0", torch.randn(1, dim))
        self.register_buffer("proto_1", torch.randn(1, dim))
        self.proto_0 = F.normalize(self.proto_0, p=2, dim=1)
        self.proto_1 = F.normalize(self.proto_1, p=2, dim=1)

    def update_prototypes(self, feats, pseudo_labels, acceptance_mask, entropy):
        B, D, H, W = feats.shape
        weights = acceptance_mask * (1 - entropy)
        feats = feats.permute(0, 2, 3, 1).reshape(-1, D)
        labels = pseudo_labels.reshape(-1)
        weights = weights.reshape(-1, 1)
        
        for c, proto in zip([0, 1], [self.proto_0, self.proto_1]):
            mask_c = (labels == c)
            if mask_c.sum() > 0:
                feats_c = feats[mask_c]
                weights_c = weights[mask_c]
                weighted_sum = (feats_c * weights_c).sum(dim=0, keepdim=True)
                total_weight = weights_c.sum() + 1e-8
                new_proto = weighted_sum / total_weight
                new_proto = F.normalize(new_proto, p=2, dim=1)
                updated_proto = self.rho * proto + (1 - self.rho) * new_proto
                updated_proto = F.normalize(updated_proto, p=2, dim=1)
                
                if c == 0:
                    self.proto_0.copy_(updated_proto)
                else:
                    self.proto_1.copy_(updated_proto)

    def compute_hardness(self, feats, probs):
        B, D, H, W = feats.shape
        feats = F.normalize(feats, p=2, dim=1)
        sim_0 = torch.sum(feats * self.proto_0.view(1, D, 1, 1), dim=1, keepdim=True)
        sim_1 = torch.sum(feats * self.proto_1.view(1, D, 1, 1), dim=1, keepdim=True)
        pi_1 = probs
        pi_0 = 1 - probs
        term_1 = pi_1 * (1 - sim_1)
        term_0 = pi_0 * (1 - sim_0)
        kappa = torch.max(term_1, term_0)
        return kappa

class SATModule:
    @staticmethod
    def compute_entropy_and_belt(prob_map, e1=0.40, e2=0.95):
        eps = 1e-7
        p = torch.clamp(prob_map, eps, 1-eps)
        entropy = -p * torch.log(p) - (1-p) * torch.log(1-p)
        max_ent = -0.5 * np.log(0.5) - 0.5 * np.log(0.5)
        norm_entropy = entropy / max_ent
        norm_entropy = torch.clamp(norm_entropy, 0, 1)
        belt = (norm_entropy >= e1) & (norm_entropy <= e2)
        return norm_entropy, belt.float()

def sat_loss_batch(student_feats, teacher_prob, teacher_label_raw, acc_mask, norm_entropy, belt, q_b, tau_biou=0.6, device='cuda'):
    B, D, H, W = student_feats.shape
    with torch.no_grad():
        avg_u = F.avg_pool2d(norm_entropy, kernel_size=11, stride=1, padding=5)
        
    w_map = (0.7 * (1 - norm_entropy) + 0.3 * avg_u) * (1 + 0.5 * belt)
    loss = torch.tensor(0.0, device=device)
    num_valid_anchors = torch.tensor(0.0, device=device)
    num_anchors = 256
    
    for b in range(B):
        h_idx, w_idx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
        h_idx, w_idx = h_idx.reshape(-1), w_idx.reshape(-1)
        
        acc_flat = acc_mask[b, 0].reshape(-1)
        belt_flat = belt[b, 0].reshape(-1)
        label_flat = teacher_label_raw[b, 0].reshape(-1)
        weight_flat = w_map[b, 0].reshape(-1)
        prob_flat = teacher_prob[b, 0].reshape(-1)
        
        perm = torch.randperm(H*W, device=device)[:num_anchors]
        anch_h = h_idx[perm]
        anch_w = w_idx[perm]
        anch_lbl = label_flat[perm]
        anch_belt = belt_flat[perm]
        anch_acc = acc_flat[perm]
        anch_w_val = weight_flat[perm]
        anch_feats = student_feats[b].view(D, -1)[:, perm].t()
        anch_feats = F.normalize(anch_feats, p=2, dim=1)
        
        max_attempts = 5
        found_pos = torch.zeros(num_anchors, dtype=torch.bool, device=device)
        pos_feats = torch.zeros_like(anch_feats)
        
        for _ in range(max_attempts):
            dy = torch.randint(-5, 6, (num_anchors,), device=device)
            dx = torch.randint(-5, 6, (num_anchors,), device=device)
            dist_sq = dy**2 + dx**2
            mask_disk = dist_sq <= 25
            cand_h = torch.clamp(anch_h + dy, 0, H-1)
            cand_w = torch.clamp(anch_w + dx, 0, W-1)
            cand_idx = cand_h * W + cand_w
            cand_lbl = label_flat[cand_idx]
            cand_belt = belt_flat[cand_idx]
            cand_acc = acc_flat[cand_idx]
            same_lbl = (cand_lbl == anch_lbl)
            
            cond1 = (anch_belt == 0) & (anch_acc == 1) & (cand_acc == 1)
            is_boundary = (anch_belt == 1)
            pass_prob = (torch.rand(num_anchors, device=device) < q_b)
            pass_or_acc = (anch_acc == 1) | (cand_acc == 1)
            cond2_candidate = is_boundary & pass_prob & pass_or_acc & same_lbl & mask_disk
            
            mask_valid = (cond1 & same_lbl & mask_disk) | cond2_candidate
            check_siou = cond2_candidate & (~found_pos)
            
            if check_siou.any():
                pad = 7
                padded_prob = F.pad(prob_flat.view(1, H, W), (pad, pad, pad, pad), mode='replicate')
                oy, ox = torch.meshgrid(torch.arange(-7, 8, device=device), torch.arange(-7, 8, device=device))
                oy, ox = oy.reshape(1, -1), ox.reshape(1, -1)
                active_idx = torch.nonzero(check_siou).squeeze(1)
                c_ah = anch_h[active_idx] + pad
                c_aw = anch_w[active_idx] + pad
                c_ch = cand_h[active_idx] + pad
                c_cw = cand_w[active_idx] + pad
                patch_ah = c_ah.unsqueeze(1) + oy
                patch_aw = c_aw.unsqueeze(1) + ox
                patch_a_idx = patch_ah * (W + 2*pad) + patch_aw
                patch_ch = c_ch.unsqueeze(1) + oy
                patch_cw = c_cw.unsqueeze(1) + ox
                patch_c_idx = patch_ch * (W + 2*pad) + patch_cw
                padded_flat = padded_prob.view(-1)
                p_a = padded_flat[patch_a_idx]
                p_c = padded_flat[patch_c_idx]
                mins = torch.min(p_a, p_c).sum(dim=1)
                maxs = torch.max(p_a, p_c).sum(dim=1)
                siou = mins / (maxs + 1e-6)
                siou_pass = (siou > tau_biou)
                mask_valid[active_idx[~siou_pass]] = False
            
            newly_found = mask_valid & (~found_pos)
            if newly_found.any():
                idx_found = torch.nonzero(newly_found).squeeze(1)
                cand_feats = student_feats[b].view(D, -1)[:, cand_idx[idx_found]].t()
                cand_feats = F.normalize(cand_feats, p=2, dim=1)
                pos_feats[idx_found] = cand_feats
                found_pos[idx_found] = True
            if found_pos.all():
                break
                
        if found_pos.any():
            logits_pos = (anch_feats[found_pos] * pos_feats[found_pos]).sum(dim=1, keepdim=True)
            logits_pos /= 0.07
            
            valid_neg_offsets = []
            for dy in range(-11, 12):
                for dx in range(-11, 12):
                    d2 = dy*dy + dx*dx
                    if 49 <= d2 <= 121:
                        valid_neg_offsets.append((dy, dx))
            valid_neg_offsets = torch.tensor(valid_neg_offsets, device=device)
            rand_off_idx = torch.randint(0, len(valid_neg_offsets), (found_pos.sum(), 64), device=device)
            sel_offsets = valid_neg_offsets[rand_off_idx]
            
            a_h = anch_h[found_pos].unsqueeze(1)
            a_w = anch_w[found_pos].unsqueeze(1)
            n_h = torch.clamp(a_h + sel_offsets[:, :, 0], 0, H-1)
            n_w = torch.clamp(a_w + sel_offsets[:, :, 1], 0, W-1)
            n_idx = n_h * W + n_w
            
            a_lbl = anch_lbl[found_pos].unsqueeze(1)
            n_lbl = label_flat[n_idx]
            neg_mask = (n_lbl != a_lbl)
            
            flat_feats = student_feats[b].view(D, -1)
            n_feats = flat_feats[:, n_idx.reshape(-1)].t()
            n_feats = n_feats.reshape(found_pos.sum(), 64, D)
            n_feats = F.normalize(n_feats, p=2, dim=2)
            
            logits_neg = (anch_feats[found_pos].unsqueeze(1) * n_feats).sum(dim=2)
            logits_neg /= 0.07
            logits_neg[~neg_mask] = -float('inf')
            
            max_val = torch.max(logits_pos, torch.max(logits_neg, dim=1, keepdim=True)[0])
            sum_exp = torch.exp(logits_pos - max_val) + torch.sum(torch.exp(logits_neg - max_val), dim=1, keepdim=True)
            log_sum_exp = max_val + torch.log(sum_exp + 1e-8)
            loss_anchor = -logits_pos + log_sum_exp
            
            weights = anch_w_val[found_pos]
            loss += (loss_anchor.squeeze() * weights).sum()
            num_valid_anchors += weights.sum()

    if num_valid_anchors > 0:
        return loss / num_valid_anchors
    return torch.tensor(0.0, device=device, requires_grad=True)
