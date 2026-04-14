import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
import copy

from dataset import FBUIDataset
from model import UNet
from transform_utils import get_tta_views
from components import TemperatureScaler, CPSGate, PCM, SATModule, sat_loss_batch
from losses import DiceLoss, PCMLoss
from metrics import compute_surface_metrics

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/FBUI')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30) # Reverted to 30 for stability
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    return parser.parse_args()

def weight_decay_step(student, teacher, alpha=0.996):
    for sp, tp in zip(student.parameters(), teacher.parameters()):
        tp.data.mul_(alpha).add_(sp.data, alpha=1-alpha)

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ds_l = FBUIDataset(args.root, split='train', fold=args.fold, mode='labeled', augment=True)
    ds_u = FBUIDataset(args.root, split='train', fold=args.fold, mode='unlabeled', augment=True)
    ds_val = FBUIDataset(args.root, split='val', fold=args.fold, augment=False)
    ds_test = FBUIDataset(args.root, split='test', fold=args.fold, augment=False) # Separate Test Set
    
    # Calib: augment=False to ensure stable predictions
    ds_calib = FBUIDataset(args.root, split='train', fold=args.fold, mode='labeled', augment=False)
    calib_indices = np.random.choice(len(ds_calib), min(len(ds_calib), 200), replace=False)
    ds_calib = torch.utils.data.Subset(ds_calib, calib_indices)
    
    loader_l = DataLoader(ds_l, batch_size=args.batch_size//2, shuffle=True, num_workers=4, drop_last=True)
    loader_u = DataLoader(ds_u, batch_size=args.batch_size//2, shuffle=True, num_workers=4, drop_last=True)
    loader_calib = DataLoader(ds_calib, batch_size=args.batch_size, shuffle=False)
    loader_val = DataLoader(ds_val, batch_size=1, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=1, shuffle=False)
    
    student = UNet(n_channels=1, n_classes=1).to(device)
    teacher = UNet(n_channels=1, n_classes=1).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
        
    scaler = TemperatureScaler().to(device)
    pcm_mod = PCM(dim=64).to(device)
    
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    criterion_ce = nn.BCEWithLogitsLoss(reduction='none')
    criterion_dice = DiceLoss()
    criterion_pcm = PCMLoss()
    
    iter_count = 0
    max_iters = len(loader_u) * args.epochs
    
    lambda_cps = 1.0
    lambda_sat = 0.3
    lambda_pcm = 0.2
    
    best_dice = 0.0
    
    # Algorithm 1: Warm-start prototypes from labeled set
    print("Warm-starting prototypes from labeled set...")
    with torch.no_grad():
        all_feats_0, all_feats_1 = [], []
        for batch in loader_l:
            img = batch['image'].to(device)
            lbl = batch['label'].to(device)
            _, feats = teacher(img, return_features=True)
            B, D, H, W = feats.shape
            feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, D)
            lbl_flat = lbl.reshape(-1)
            
            mask_0 = (lbl_flat == 0)
            mask_1 = (lbl_flat == 1)
            if mask_0.sum() > 0:
                all_feats_0.append(feats_flat[mask_0])
            if mask_1.sum() > 0:
                all_feats_1.append(feats_flat[mask_1])
        
        if all_feats_0:
            proto_0_init = torch.cat(all_feats_0, dim=0).mean(dim=0, keepdim=True)
            pcm_mod.proto_0.copy_(torch.nn.functional.normalize(proto_0_init, p=2, dim=1))
        if all_feats_1:
            proto_1_init = torch.cat(all_feats_1, dim=0).mean(dim=0, keepdim=True)
            pcm_mod.proto_1.copy_(torch.nn.functional.normalize(proto_1_init, p=2, dim=1))
    print("Prototypes initialized.")
    
    for epoch in range(args.epochs):
        student.train()
        teacher.eval() 
        
        if epoch % 3 == 0:
            print("Calibrating Teacher...")
            logits_list = []
            labels_list = []
            with torch.no_grad():
                for batch in loader_calib:
                    img = batch['image'].to(device)
                    lbl = batch['label'].to(device)
                    logits = teacher(img)
                    logits_list.append(logits)
                    labels_list.append(lbl)
            scaler.fit(torch.cat(logits_list), torch.cat(labels_list))
            
        pbar = tqdm(zip(loader_l, loader_u), total=min(len(loader_l), len(loader_u)))
        
        meter_cps_acc = 0
        total_batches = 0
        
        for batch_l, batch_u in pbar:
            progress = iter_count / max_iters
            # Baseline: tau adapted to calibrated confidence (0.60->0.80)
            tau_t = 0.60 - (0.60 - 0.80) * progress
            gamma = 0.3 + (0.7 - 0.3) * progress
            q_b = 0.4 * min(1.0, progress * 2)
            
            img_l = batch_l['image_strong'].to(device)
            lbl_l = batch_l['label'].to(device)
            
            s_logits_l, s_feats_l = student(img_l, return_features=True)
            loss_sup = criterion_ce(s_logits_l, lbl_l).mean() + criterion_dice(torch.sigmoid(s_logits_l), lbl_l)
            
            # Unlabeled
            img_u_w = batch_u['image'].to(device)
            img_u_s = batch_u['image_strong'].to(device)
            
            tta_views = get_tta_views(img_u_w)
            
            teacher_probs_sum = 0
            with torch.no_grad():
                for img_v, inv_fn in tta_views:
                    img_v = img_v.to(device)
                    if img_v.dim() == 3: img_v = img_v.unsqueeze(0)
                    logits_v = teacher(img_v)
                    probs_v = torch.sigmoid(scaler(logits_v))
                    probs_v_orig = inv_fn(probs_v)
                    teacher_probs_sum += probs_v_orig
            
            tta_avg = teacher_probs_sum / len(tta_views)
            
            with torch.no_grad():
                t_logits, t_feats_u = teacher(img_u_w, return_features=True)
            
            # Algorithm 1: delta=0.5 (Baseline), tau from schedule
            accepted_mask, pseudo_label, prob_calib = CPSGate.check_stability_and_confidence(
                t_logits, tta_avg, tau_t=tau_t, delta=0.5, temp=scaler.temperature
            )
            
            meter_cps_acc += accepted_mask.float().mean().item()
            total_batches += 1
            
            s_logits_u, s_feats_u = student(img_u_s, return_features=True)
            
            # Bidirectional CPS (Algorithm 1: symmetrical)
            loss_cps_ts = (criterion_ce(s_logits_u, pseudo_label) * accepted_mask).sum() / (accepted_mask.sum() + 1e-6)
            loss_cps = loss_cps_ts
            
            norm_entropy, belt = SATModule.compute_entropy_and_belt(prob_calib)
            
            loss_sat = sat_loss_batch(
                s_feats_u, prob_calib, pseudo_label, accepted_mask, 
                norm_entropy, belt, q_b=q_b, device=device
            )
            
            # Hybrid: Use STUDENT embeddings for more discriminative prototypes
            with torch.no_grad():
                pcm_mod.update_prototypes(s_feats_u.detach(), pseudo_label, accepted_mask, norm_entropy)
                
            hardness = pcm_mod.compute_hardness(s_feats_u, torch.sigmoid(s_logits_u))
            flat_hard = hardness.view(-1)
            k = int(len(flat_hard) * gamma)
            if k > 0:
                _, topk_idx = torch.topk(flat_hard, k)
                mask_curric = torch.zeros_like(flat_hard, dtype=torch.bool)
                mask_curric[topk_idx] = True
                mask_curric = mask_curric.view_as(hardness)
                loss_pcm = criterion_pcm(s_feats_u, torch.sigmoid(s_logits_u), pcm_mod.proto_0, pcm_mod.proto_1, mask_curric)
            else:
                loss_pcm = torch.tensor(0.0, device=device)
            
            loss = loss_sup + lambda_cps * loss_cps + lambda_sat * loss_sat + lambda_pcm * loss_pcm
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            weight_decay_step(student, teacher)
            iter_count += 1
            
            pbar.set_description(f"L:{loss.item():.2f} CPS:{loss_cps.item():.4f} SAT:{loss_sat.item():.4f} PCM:{loss_pcm.item():.3f} Acc:{meter_cps_acc/total_batches:.1%}")
        
        scheduler.step()
        
        # Validation
        student.eval()
        dice_scores = []
        with torch.no_grad():
            for batch in loader_val:
                img = batch['image'].to(device)
                lbl = batch['label'].to(device)
                logits = student(img)
                pred = (torch.sigmoid(logits) > 0.5).float()
                
                inter = (pred * lbl).sum()
                union = pred.sum() + lbl.sum()
                dice = (2 * inter) / (union + 1e-6)
                dice_scores.append(dice.item())
                
        mean_dice = np.mean(dice_scores)
        print(f"Epoch {epoch+1} Val DSC: {mean_dice:.4f}")
        
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(student.state_dict(), f"student_fold{args.fold}_best.pth")
    
    # Final Test Evaluation
    print("\nTraining Finished. Loading Best Model for Final Test Evaluation...")
    student.load_state_dict(torch.load(f"student_fold{args.fold}_best.pth"))
    student.eval()
    
    test_dice, test_asd, test_hd95 = [], [], []
    with torch.no_grad():
        for batch in loader_test:
            img = batch['image'].to(device)
            lbl = batch['label'].to(device)
            logits = student(img)
            pred = (torch.sigmoid(logits) > 0.5).float()
            
            inter = (pred * lbl).sum()
            union = pred.sum() + lbl.sum()
            dice = (2 * inter) / (union + 1e-6)
            test_dice.append(dice.item())
            
            if lbl.sum() > 0 and pred.sum() > 0:
                metrics = compute_surface_metrics(pred, lbl)
                test_asd.append(metrics['asd'])
                test_hd95.append(metrics['hd95'])
    
    print(f"Final Test DSC: {np.mean(test_dice):.4f}")
    print(f"Final Test ASD: {np.mean(test_asd):.4f}")
    print(f"Final Test HD95: {np.mean(test_hd95):.4f}")

if __name__ == '__main__':
    main()
