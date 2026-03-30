"""
Standard Element-Based Ogden Physics Loss with Kinematic Error Routing
for 2D MeshGraphNets.

This computes the deformation gradient and strain energy strictly inside 
each 3-node element, but distributes the resulting physics penalty back to 
the nodes proportionally based on their individual kinematic tracking errors.
"""

import torch


def compute_element_physics_loss(pred, target, batch, model, physics_weight=0.005):
    """
    Computes Ogden strain energy error per element, distributes to vertices 
    based on their kinematic error ("The Blame Game").
    """
    if physics_weight <= 0.0:
        return torch.zeros_like(pred)
    
    pos = batch.pos
    mesh_pos = batch.mesh_pos
    elements = batch.elements           # [num_tris, 3]
    spatial_dim = model.spatial_dim
    
    is_steel = batch.node_attr[:, 0].bool()
    is_tpu = ~is_steel
    is_fixed = batch.mask.squeeze().bool()
    
    curr_vel = batch.velocities[:, spatial_dim:]
    
    # --- Denormalize predictions (differentiable) ---
    pred_raw = torch.zeros_like(pred)
    if is_tpu.sum() > 0:
        pred_raw[is_tpu] = model.absorber_target_normalizer.inverse(pred[is_tpu])
    
    # --- Integrate to next positions ---
    pred_next_vel = curr_vel + pred_raw
    pred_next_vel[is_fixed] = 0.0
    pred_next_vel[is_steel] = 0.0
    pred_next_pos = pos + pred_next_vel * model.dt
    
    gt_next_vel = curr_vel + target
    gt_next_vel[is_fixed] = 0.0
    gt_next_pos = pos + gt_next_vel * model.dt
    
    # --- Filter to TPU-only elements ---
    n0, n1, n2 = elements[:, 0], elements[:, 1], elements[:, 2]
    elem_is_tpu = is_tpu[n0] & is_tpu[n1] & is_tpu[n2]
    
    if elem_is_tpu.sum() == 0:
        return torch.zeros_like(pred)
    
    tpu_elems = elements[elem_is_tpu]   # [num_tpu_tris, 3]
    i0, i1, i2 = tpu_elems[:, 0], tpu_elems[:, 1], tpu_elems[:, 2]
    
    # --- Reference element areas (for weighting, detached) ---
    ref_areas = _triangle_areas(mesh_pos, i0, i1, i2).detach()
    
    # --- Deformation gradient F per element ---
    F_pred = _deformation_gradient(mesh_pos, pred_next_pos, i0, i1, i2)
    F_gt = _deformation_gradient(mesh_pos, gt_next_pos, i0, i1, i2)
    
    # --- Principal stretches via C = F^T F ---
    lam1_pred, lam2_pred = _principal_stretches(F_pred)
    lam1_gt, lam2_gt = _principal_stretches(F_gt)
    
    # --- 2D Ogden energy density (plane stress, incompressible) ---
    W_pred = _ogden_2d(lam1_pred, lam2_pred, model.OGDEN_MU, model.OGDEN_ALPHA)
    W_gt = _ogden_2d(lam1_gt, lam2_gt, model.OGDEN_MU, model.OGDEN_ALPHA)
    
    # --- Log-compressed error per element ---
    elem_error = torch.log1p(torch.abs(W_pred - W_gt))
    
    # --- Area-weighted: larger elements contribute more ---
    total_area = ref_areas.sum().clamp(min=1e-12)
    elem_error_weighted = elem_error * (ref_areas / total_area) * ref_areas.size(0)
    
    # =========================================================
    # AXIS-WISE KINEMATIC ERROR DISTRIBUTION ("The 2D Blame Game")
    # =========================================================
    
    # 1. Calculate absolute error per node, PER AXIS. 
    # Shape is [N, 2]. We do NOT take the mean. Detach is still required.
    axis_err = torch.abs(pred - target).detach()
    
    # 2. Extract the axis-specific errors for the 3 vertices of each TPU element
    # Shapes are [E, 2]
    err_0 = axis_err[i0]
    err_1 = axis_err[i1]
    err_2 = axis_err[i2]
    
    # 3. Calculate relative blame PER AXIS (weights sum to 1.0 per element, per axis)
    # Shape of sum_err is [E, 2]. 
    sum_err = err_0 + err_1 + err_2 + 1e-12
    
    # Shapes of w0, w1, w2 are [E, 2]. No unsqueeze needed anymore!
    w0 = err_0 / sum_err  
    w1 = err_1 / sum_err
    w2 = err_2 / sum_err
    
    # 4. Expand the scalar element energy error to match spatial dimensions [E, 2]
    err_expanded = elem_error_weighted.unsqueeze(1).expand(-1, spatial_dim)
    
    # 5. Distribute the Ogden penalty perfectly proportionally per node AND per axis
    physics_node_loss = torch.zeros_like(pred)
    physics_node_loss.index_add_(0, i0, err_expanded * w0)
    physics_node_loss.index_add_(0, i1, err_expanded * w1)
    physics_node_loss.index_add_(0, i2, err_expanded * w2)
    
    return physics_node_loss * physics_weight


def _deformation_gradient(ref_pos, curr_pos, i0, i1, i2):
    """Compute 2D deformation gradient F for each triangle."""
    dX1 = ref_pos[i1] - ref_pos[i0]
    dX2 = ref_pos[i2] - ref_pos[i0]
    
    dx1 = curr_pos[i1] - curr_pos[i0]
    dx2 = curr_pos[i2] - curr_pos[i0]
    
    dX = torch.stack([dX1, dX2], dim=2)
    dx = torch.stack([dx1, dx2], dim=2)
    
    dX_inv = _batch_inv_2x2(dX)
    F = torch.bmm(dx, dX_inv)
    
    return F


def _batch_inv_2x2(M):
    """Analytical inverse of batched 2x2 matrices."""
    a = M[:, 0, 0]
    b = M[:, 0, 1]
    c = M[:, 1, 0]
    d = M[:, 1, 1]
    
    det = (a * d - b * c).clamp(min=1e-12)
    
    inv = torch.zeros_like(M)
    inv[:, 0, 0] = d / det
    inv[:, 0, 1] = -b / det
    inv[:, 1, 0] = -c / det
    inv[:, 1, 1] = a / det
    
    return inv


def _principal_stretches(F):
    """Compute principal stretches from deformation gradient F."""
    C = torch.bmm(F.transpose(1, 2), F)
    
    a = C[:, 0, 0]
    b = C[:, 0, 1]
    d = C[:, 1, 1]
    
    trace = a + d
    discriminant = ((a - d) / 2.0) ** 2 + b ** 2
    sqrt_disc = torch.sqrt(discriminant.clamp(min=0.0))
    
    lam_sq_1 = (trace / 2.0 + sqrt_disc).clamp(min=1e-12)
    lam_sq_2 = (trace / 2.0 - sqrt_disc).clamp(min=1e-12)
    
    lam1 = torch.sqrt(lam_sq_1).clamp(min=0.1, max=5.0)
    lam2 = torch.sqrt(lam_sq_2).clamp(min=0.1, max=5.0)
    
    return lam1, lam2


def _ogden_2d(lam1, lam2, mu_list, alpha_list):
    """2D Ogden strain energy density (plane stress, incompressible)."""
    W = torch.zeros_like(lam1)
    
    for mu, alpha in zip(mu_list, alpha_list):
        lam3_alpha = (lam1 * lam2).clamp(min=0.01) ** (-alpha)
        W = W + (mu / alpha) * (lam1 ** alpha + lam2 ** alpha + lam3_alpha - 3.0)
    
    return W


def _triangle_areas(pos, i0, i1, i2):
    """Compute areas of triangles from vertex positions."""
    v1 = pos[i1] - pos[i0]
    v2 = pos[i2] - pos[i0]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return (0.5 * torch.abs(cross)).clamp(min=1e-16)