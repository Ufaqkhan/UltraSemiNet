import numpy as np
import cv2

def compute_surface_metrics(pred, gt, spacing=(1.0, 1.0)):
    """
    Computes ASD and HD95 using OpenCV fast distance transform.
    pred, gt: boolean numpy arrays or tensors (H, W).
    """
    if hasattr(pred, 'cpu'): pred = pred.cpu().numpy()
    if hasattr(gt, 'cpu'): gt = gt.cpu().numpy()
    
    pred = np.squeeze(pred).astype(np.uint8)
    gt = np.squeeze(gt).astype(np.uint8)
    
    if pred.sum() == 0 or gt.sum() == 0:
        return {'asd': 0.0, 'hd95': 0.0}
        
    # Get boundaries
    # boundary = mask - eroded_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    p_border = pred ^ cv2.erode(pred, kernel)
    g_border = gt ^ cv2.erode(gt, kernel)
    
    if p_border.sum() == 0 or g_border.sum() == 0:
         return {'asd': 0.0, 'hd95': 0.0}
         
    # Current grid is 1x1 mm assumed if spacing is not used in dist transform, 
    # but we can scale result.
    
    # Distance from P border pixels to G border
    # Invert G border: 0 is border, 1 is background? 
    # distTransform calculates distance to nearest ZERO pixel.
    # So we want G border to be 0, everything else 1.
    g_border_inv = 1 - g_border
    dmap_g = cv2.distanceTransform(g_border_inv, cv2.DIST_L2, 5) # Float32 map
    
    # Values at P border
    dists_p2g = dmap_g[p_border == 1]
    
    # Distance from G border pixels to P border
    p_border_inv = 1 - p_border
    dmap_p = cv2.distanceTransform(p_border_inv, cv2.DIST_L2, 5)
    
    dists_g2p = dmap_p[g_border == 1]
    
    # Combine
    all_dists = np.concatenate([dists_p2g, dists_g2p])
    
    # Scale by pixel spacing (assume isotropic for simplicity provided single value)
    scale = spacing[0]
    
    asd = np.mean(all_dists) * scale
    hd95 = np.percentile(all_dists, 95) * scale
    
    return {'asd': asd, 'hd95': hd95}
