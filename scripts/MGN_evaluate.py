import torch
import os
from torch_geometric.loader import DataLoader
from src.data_formatter import DataFormatter
from src.model import MeshGraphNet

# =========================================================
# IMPORT SINGLE SOURCE OF TRUTH FOR PHYSICS
# =========================================================
from src.physics_loss import (
    compute_element_physics_loss,
    _deformation_gradient,
    _principal_stretches,
    _ogden_2d,
    _triangle_areas
)

# Constants for the 1D Edge-Based comparison
MU = [11.5138e6, -0.33239e6]
ALPHA = [1.457, -7.8825]

# =========================================================
# EVALUATION
# =========================================================
def evaluate_epoch(model, loader, config, device):
    """
    Validation loop with element-based 2D Ogden physics diagnostic.
    Computes full deformation gradient F per triangle, extracts principal
    stretches, and evaluates proper plane-stress incompressible Ogden.
    Falls back to edge-based stretch if elements not available.
    """
    model.eval()
    total_loss = 0
    total_physics_node_loss = 0
    num_batches = 0
    
    # Element-based tracking
    all_energy_errors = []
    all_lam1_errors = []
    all_lam2_errors = []
    all_areas = []
    
    # Edge-based tracking (always computed for comparison)
    all_edge_stretch_errors = []
    all_edge_energy_errors = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Forward
            pred = model(batch, accumulate_stats=False)
            
            # 1. Standard Kinematic Loss
            loss = model.loss(pred, batch.y, batch, mask=batch.mask, accumulate_stats=False)
            total_loss += loss.item()

            spatial_dim = batch.pos.shape[1]
            curr_vel = batch.velocities[:, spatial_dim:]
            is_steel = batch.node_attr[:, 0].bool()
            is_tpu = ~is_steel
            is_fixed = batch.mask.squeeze().bool()
            
            # 2. Raw Node-Distributed Physics Loss (using weight 1.0 to see raw magnitude)
            phys_loss_tensor = compute_element_physics_loss(pred, batch.y, batch, model, physics_weight=1.0)
            
            if is_tpu.sum() > 0:
                active_loss = torch.abs(phys_loss_tensor[is_tpu])
                total_physics_node_loss += active_loss.mean().item()
                
            num_batches += 1
            
            # =========================================================
            # SETUP: Denormalize + integrate (matches rollout exactly)
            # =========================================================
            
            pred_raw = torch.zeros_like(pred)
            if is_tpu.sum() > 0:
                pred_raw[is_tpu] = model.absorber_target_normalizer.inverse(pred[is_tpu])
            pred_raw[is_steel] = 0.0
            pred_raw[is_fixed] = 0.0
            
            pred_next_vel = curr_vel + pred_raw
            pred_next_vel[is_fixed] = 0.0
            pred_next_vel[is_steel] = 0.0
            pred_next_pos = batch.pos + pred_next_vel * model.dt
            
            gt_next_vel = curr_vel + batch.y
            gt_next_vel[is_fixed] = 0.0
            gt_next_pos = batch.pos + gt_next_vel * model.dt
            
            # =========================================================
            # ELEMENT-BASED 2D OGDEN (using src.physics_loss functions)
            # =========================================================
            if hasattr(batch, 'elements') and batch.elements is not None:
                elements = batch.elements
                n0, n1, n2 = elements[:, 0], elements[:, 1], elements[:, 2]
                
                elem_is_tpu = is_tpu[n0] & is_tpu[n1] & is_tpu[n2]
                
                if elem_is_tpu.sum() > 0:
                    tpu_elems = elements[elem_is_tpu]
                    i0, i1, i2 = tpu_elems[:, 0], tpu_elems[:, 1], tpu_elems[:, 2]
                    
                    F_pred = _deformation_gradient(batch.mesh_pos, pred_next_pos, i0, i1, i2)
                    F_gt = _deformation_gradient(batch.mesh_pos, gt_next_pos, i0, i1, i2)
                    
                    lam1_pred, lam2_pred = _principal_stretches(F_pred)
                    lam1_gt, lam2_gt = _principal_stretches(F_gt)
                    
                    W_pred = _ogden_2d(lam1_pred, lam2_pred, MU, ALPHA)
                    W_gt = _ogden_2d(lam1_gt, lam2_gt, MU, ALPHA)
                    
                    areas = _triangle_areas(batch.mesh_pos, i0, i1, i2)
                    
                    all_energy_errors.append(torch.abs(W_pred - W_gt).cpu())
                    all_lam1_errors.append(torch.abs(lam1_pred - lam1_gt).cpu())
                    all_lam2_errors.append(torch.abs(lam2_pred - lam2_gt).cpu())
                    all_areas.append(areas.cpu())
            
            # =========================================================
            # EDGE-BASED STRETCH (always computed)
            # =========================================================
            src, dst = batch.edge_index
            edge_is_tpu = is_tpu[src] & is_tpu[dst] & (src < dst)
            
            if edge_is_tpu.sum() > 0:
                src_t, dst_t = src[edge_is_tpu], dst[edge_is_tpu]
                L0 = (batch.mesh_pos[src_t] - batch.mesh_pos[dst_t]).norm(dim=1).clamp(min=1e-9)
                L_pred = (pred_next_pos[src_t] - pred_next_pos[dst_t]).norm(dim=1).clamp(min=1e-9)
                L_gt = (gt_next_pos[src_t] - gt_next_pos[dst_t]).norm(dim=1).clamp(min=1e-9)
                
                lam_pred = (L_pred / L0).clamp(min=0.1, max=10.0)
                lam_gt = (L_gt / L0).clamp(min=0.1, max=10.0)
                
                all_edge_stretch_errors.append(torch.abs(lam_pred - lam_gt).cpu())
                
                # 1D Ogden for comparison
                W_edge_pred = torch.zeros_like(lam_pred)
                W_edge_gt = torch.zeros_like(lam_gt)
                for mu, alpha in zip(MU, ALPHA):
                    W_edge_pred += (mu / alpha) * (lam_pred**alpha + lam_pred**(-alpha) - 2.0)
                    W_edge_gt += (mu / alpha) * (lam_gt**alpha + lam_gt**(-alpha) - 2.0)
                all_edge_energy_errors.append(torch.abs(W_edge_pred - W_edge_gt).cpu())

    # =========================================================
    # PRINT DIAGNOSTICS
    # =========================================================
    avg_kinematic_loss = total_loss / num_batches
    avg_physics_loss = total_physics_node_loss / num_batches
    
    # Element-based
    if len(all_energy_errors) > 0:
        cat_W = torch.cat(all_energy_errors)
        cat_l1 = torch.cat(all_lam1_errors)
        cat_l2 = torch.cat(all_lam2_errors)
        cat_areas = torch.cat(all_areas)
        
        total_area = cat_areas.sum()
        weighted_W = (cat_W * cat_areas).sum() / total_area
        
        print(f"\n --- ELEMENT-BASED 2D OGDEN (Validation) ---")
        print(f"   TPU elements evaluated: {len(cat_W):,}")
        print(f"   Node Loss Output (Raw):  {avg_physics_loss:.6f}")
        print(f"   Energy |ΔW|:")
        print(f"     Mean:            {cat_W.mean().item():.6f}")
        print(f"     Area-weighted:   {weighted_W.item():.6f}")
        print(f"     Median:          {cat_W.median().item():.6f}")
        print(f"     Std:             {cat_W.std().item():.6f}")
        print(f"     P90:             {cat_W.quantile(0.90).item():.6f}")
        print(f"     P99:             {cat_W.quantile(0.99).item():.6f}")
        print(f"     Max:             {cat_W.max().item():.6f}")
        print(f"   Principal stretch |Δλ1| (major):")
        print(f"     Mean:   {cat_l1.mean().item():.8f}")
        print(f"     Median: {cat_l1.median().item():.8f}")
        print(f"     P99:    {cat_l1.quantile(0.99).item():.8f}")
        print(f"     Max:    {cat_l1.max().item():.8f}")
        print(f"   Principal stretch |Δλ2| (minor):")
        print(f"     Mean:   {cat_l2.mean().item():.8f}")
        print(f"     Median: {cat_l2.median().item():.8f}")
        print(f"     P99:    {cat_l2.quantile(0.99).item():.8f}")
        print(f"     Max:    {cat_l2.max().item():.8f}")
        print(f"   Element areas (reference):")
        print(f"     Mean:   {cat_areas.mean().item():.3e}")
        print(f"     Min:    {cat_areas.min().item():.3e}")
        print(f"     Max:    {cat_areas.max().item():.3e}")
        print(f"------------------------------------------------\n")
    else:
        print("\nNo 'elements' field in data — skipping element-based physics.")
    
    # Edge-based (always available)
    if len(all_edge_stretch_errors) > 0:
        cat_edge_lam = torch.cat(all_edge_stretch_errors)
        cat_edge_W = torch.cat(all_edge_energy_errors)
        
        print(f"   --- EDGE-BASED 1D OGDEN (Comparison) ---")
        print(f"   TPU edges evaluated: {len(cat_edge_lam):,}")
        print(f"   |Δλ_edge|:")
        print(f"     Mean:   {cat_edge_lam.mean().item():.8f}")
        print(f"     Median: {cat_edge_lam.median().item():.8f}")
        print(f"     P99:    {cat_edge_lam.quantile(0.99).item():.8f}")
        print(f"     Max:    {cat_edge_lam.max().item():.8f}")
        print(f"   |ΔW_edge| (1D Ogden):")
        print(f"     Mean:   {cat_edge_W.mean().item():.6f}")
        print(f"     Median: {cat_edge_W.median().item():.6f}")
        print(f"     P99:    {cat_edge_W.quantile(0.99).item():.6f}")
        print(f"     Max:    {cat_edge_W.max().item():.6f}")
        print(f"--------------------------------------------\n")
            
    return avg_kinematic_loss


def run_evaluation(config, device):
    """
    Standalone evaluation routine.
    """
    test_set = DataFormatter(config['data']['val_path'], augment=False)
    test_loader = DataLoader(
        test_set, 
        batch_size=config['train']['batch_size'], 
        shuffle=False,
    )

    model = MeshGraphNet(config).to(device)
    
    ckpt_path = config['rollout']['checkpoint_path'] 
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    print(f"Loading weights from: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    clean_checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(clean_checkpoint, strict=True)
    
    avg_loss = evaluate_epoch(model, test_loader, config, device)
    
    print(f"Final Validation Loss (Kinematic MSE): {avg_loss:.8f}")
    return avg_loss