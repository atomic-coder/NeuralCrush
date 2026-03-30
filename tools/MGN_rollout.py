import torch
import numpy as np
import sys
import os
from torch_geometric.loader import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import MeshGraphNet
from src.data_formatter import DataFormatter

try:
    from tools.plot import plot_results
    HAS_PLOTTER = True
except ImportError:
    HAS_PLOTTER = False

# =========================================================
# MATERIAL MODEL (Ogden Hyperelastic)
# =========================================================
MU = [11.5138e6, -0.33239e6]
ALPHA = [1.457, -7.8825]
RHO_TPU = 1210.0


def precompute_energy_context(graph, device):
    """
    Precompute TPU edges, reference lengths, and exact volume
    recovered from inverse mass. Single-graph version (no batching).
    """
    edge_index = graph.edge_index
    is_tpu = ~graph.node_attr[:, 0].bool()
    is_steel = graph.node_attr[:, 0].bool()
    ref_pos = graph.mesh_pos
    inv_mass = graph.inv_mass.squeeze()

    # TPU-only edges (deduplicated)
    tpu_mask = is_tpu[edge_index[0]] & is_tpu[edge_index[1]]
    src, dst = edge_index[:, tpu_mask]
    uniq = src < dst
    src, dst = src[uniq], dst[uniq]

    L0 = (ref_pos[src] - ref_pos[dst]).norm(dim=1).clamp(min=1e-9)

    # Exact volume from inverse mass
    tpu_mass = torch.zeros_like(inv_mass)
    tpu_active = is_tpu & (inv_mass > 1e-6)
    tpu_mass[tpu_active] = 1.0 / inv_mass[tpu_active]
    total_volume = tpu_mass.sum() / RHO_TPU

    # Length-weighted volume per edge
    L0_sum = L0.sum().clamp(min=1e-9)
    V_per_edge = total_volume * (L0 / L0_sum)

    # Reference steel Y for displacement tracking
    steel_y = ref_pos[is_steel, 1]
    ref_steel_y = steel_y.mean() if steel_y.numel() > 0 else torch.tensor(0.0, device=device)

    mu_t = torch.tensor(MU, dtype=torch.float32, device=device)
    alpha_t = torch.tensor(ALPHA, dtype=torch.float32, device=device)

    return {
        'src': src, 'dst': dst, 'L0': L0,
        'V_per_edge': V_per_edge,
        'ref_steel_y': ref_steel_y,
        'is_steel': is_steel,
        'mu': mu_t, 'alpha': alpha_t,
    }


def compute_strain_energy(pos, ctx):
    """Total strain energy U from current positions."""
    src, dst, L0 = ctx['src'], ctx['dst'], ctx['L0']
    L = (pos[src] - pos[dst]).norm(dim=1).clamp(min=1e-9)
    lam = L / L0

    lam_e = lam.unsqueeze(0)
    a = ctx['alpha'].unsqueeze(1)
    m = ctx['mu'].unsqueeze(1)
    W = ((m / a) * (lam_e ** a + lam_e ** (-a) - 2.0)).sum(dim=0)

    return (W * ctx['V_per_edge']).sum()


def compute_impactor_disp(pos, ctx):
    """Mean Y displacement of steel from reference."""
    steel_y = pos[ctx['is_steel'], 1]
    if steel_y.numel() == 0:
        return torch.tensor(0.0, device=pos.device)
    return steel_y.mean() - ctx['ref_steel_y']


def derive_forces(energy_hist, disp_hist):
    """F = -dU/dy at each sampled step."""
    energy = np.array(energy_hist)
    disp = np.array(disp_hist)
    forces = np.zeros_like(energy)
    for t in range(1, len(energy)):
        dU = energy[t] - energy[t - 1]
        dy = disp[t] - disp[t - 1]
        safe_dy = dy if abs(dy) > 1e-10 else 1e-10
        forces[t] = -dU / safe_dy
    forces[0] = forces[1] if len(forces) > 1 else 0.0
    return forces


def run_rollout(config, device):
    print("Mode: ROLLOUT (Pure Deterministic Physics)")

    # 1. Setup
    config['mode'] = 'rollout'
    model = MeshGraphNet(config).to(device)
    ckpt_path = config['rollout']['checkpoint_path']
    print(f"Loading checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    clean_checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}

    try:
        model.load_state_dict(clean_checkpoint, strict=True)
        print("✅ Weights loaded perfectly.")
    except RuntimeError as e:
        print("\n❌ ARCHITECTURE MISMATCH DETECTED!")
        raise e

    model.eval()

    # 2. Initial State
    test_set = DataFormatter(config['data']['test_path'], augment=False)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    initial_data = next(iter(loader)).to(device)

    spatial_dim = initial_data.pos.shape[1]
    print(f"Starting simulation for {config['rollout']['num_steps']} steps...")
    print(f"Spatial Dimension: {spatial_dim}D")

    # 3. Storage
    trajectory = [initial_data.pos.cpu().numpy()]
    world_edges_history = []

    # 4. State Variables
    current_graph = initial_data.clone()
    curr_pos = initial_data.pos.clone()
    curr_prev_vel = initial_data.velocities[:, :spatial_dim].clone()
    curr_vel = initial_data.velocities[:, spatial_dim:spatial_dim*2].clone()

    dt = config['rollout']['time_step']
    static_mask = initial_data.mask.bool().squeeze()

    # 5. Strain Energy Context
    energy_ctx = precompute_energy_context(current_graph, device)
    energy_history = []
    disp_history = []
    sample_interval = 3

    print(f"📦 TPU Volume: {energy_ctx['V_per_edge'].sum().item():.6e} m³")

    with torch.no_grad():
        for step in range(config['rollout']['num_steps']):

            current_graph.pos = curr_pos
            current_graph.velocities = torch.cat([curr_prev_vel, curr_vel], dim=-1)

            # Forward
            outputs = model(current_graph, accumulate_stats=False)

            step_world_edges = torch.empty((2, 0), dtype=torch.long, device=device)
            if len(outputs) == 2:
                pred_accel_norm, step_world_edges = outputs
            else:
                pred_accel_norm = outputs
                if step == 0: print("⚠️ WARNING: Model returned only 1 item.")

            # Denormalize
            is_steel = current_graph.node_attr[:, 0].bool()
            is_tpu = ~is_steel

            pred_accel = torch.zeros_like(pred_accel_norm)
            if is_tpu.sum() > 0:
                pred_accel[is_tpu] = model.absorber_target_normalizer.inverse(
                    pred_accel_norm[is_tpu].detach())

            pred_accel[static_mask] = 0.0
            pred_accel[is_steel] = 0.0
            curr_vel[static_mask] = 0.0

            # Integrate
            next_vel = curr_vel + pred_accel
            next_pos = curr_pos + next_vel * dt

            # Store trajectory
            trajectory.append(next_pos.cpu().numpy())

            if step_world_edges.numel() > 0:
                edges_np = step_world_edges.t().cpu().numpy()
            else:
                edges_np = np.empty((0, 2), dtype=int)
            world_edges_history.append(edges_np)

            # Sample strain energy
            if step % sample_interval == 0 or step == config['rollout']['num_steps'] - 1:
                U = compute_strain_energy(next_pos, energy_ctx).item()
                d = compute_impactor_disp(next_pos, energy_ctx).item()
                energy_history.append(U)
                disp_history.append(d)

            # Update state
            curr_pos = next_pos
            curr_prev_vel = curr_vel
            curr_vel = next_vel

            if step % 10 == 0:
                max_accel = pred_accel.abs().max().item()
                num_edges = len(edges_np)
                print(f"Step {step}: Collisions={num_edges} | Max Accel={max_accel:.6f}")

    # 6. Derive Forces
    forces = derive_forces(energy_history, disp_history)
    disp_mm = np.abs(np.array(disp_history) - disp_history[0]) * 1000.0

    # CFE
    skip = max(1, len(forces) // 10)
    steady = forces[skip:]
    peak = np.max(np.abs(steady)) if len(steady) > 0 else 1e-6
    mean = np.mean(np.abs(steady)) if len(steady) > 0 else 0.0
    cfe = mean / peak if peak > 1e-6 else 0.0

    print(f"\n--- CRUSH FORCE ANALYSIS ---")
    print(f"   Peak Force: {peak:.2f} N")
    print(f"   Mean Force: {mean:.2f} N")
    print(f"   CFE:        {cfe:.4f}")
    print(f"-------------------------------")

    # 7. Save
    rollout_dir = config['data']['rollout_dir']
    os.makedirs(rollout_dir, exist_ok=True)

    traj_path = os.path.join(rollout_dir, "rollout.npy")
    np.save(traj_path, np.stack(trajectory))

    edges_path = os.path.join(rollout_dir, "world_edges.npy")
    np.save(edges_path, np.array(world_edges_history, dtype=object), allow_pickle=True)

    # Save force data for plot.py
    force_data = {
        'forces': forces,
        'disp_mm': disp_mm,
        'energy': np.array(energy_history),
        'cfe': cfe,
        'sample_interval': sample_interval,
    }
    np.save(os.path.join(rollout_dir, "force_data.npy"), force_data, allow_pickle=True)

    print(f"Saved rollout + force data to {rollout_dir}")

    # 8. Visualize
    if HAS_PLOTTER:
        plot_results(config)