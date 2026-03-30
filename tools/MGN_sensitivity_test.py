import torch
import numpy as np
import time
import multiprocessing as mp
import os
import sys
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from torch_geometric.data import Batch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.model import MeshGraphNet
from src.data_formatter import MeshData as Data
from tools.utils import build_world_edges
from rl_environment import generate_rl_environment, BOX_SIZE

'''
This scripts conducts a sensitivity test on your trained MGN weights by applying unit gaussian noise
to the seed point coordinates of a simple structure with a strut down the middle.
'''

# =========================================================
# MATERIAL MODEL (Ogden Hyperelastic)
# =========================================================
MU = [11.5138e6, -0.33239e6]
ALPHA = [1.457, -7.8825]
RHO_TPU = 1210.0
ACTUAL_THICKNESS = 0.05  # meters


# =========================================================
# STRAIN ENERGY & FORCE COMPUTATION (Batched, exact volume)
# =========================================================
def precompute_energy_context(batched_graph, valid_data_list, device):
    """
    Precompute TPU edges, reference lengths, and EXACT volume per env
    recovered from inverse mass.
    """
    edge_index = batched_graph.edge_index
    is_tpu_node = batched_graph.node_attr[:, 1].bool()
    is_steel_node = batched_graph.node_attr[:, 0].bool()
    batch = batched_graph.batch
    ref_pos = batched_graph.mesh_pos
    inv_mass = batched_graph.inv_mass.squeeze()
    num_envs = len(valid_data_list)

    # --- TPU-only edges (deduplicated: src < dst) ---
    tpu_edge_mask = is_tpu_node[edge_index[0]] & is_tpu_node[edge_index[1]]
    src, dst = edge_index[:, tpu_edge_mask]
    unique_mask = src < dst
    src, dst = src[unique_mask], dst[unique_mask]

    L0 = (ref_pos[src] - ref_pos[dst]).norm(dim=1).clamp(min=1e-9)
    edge_batch = batch[src]

    # --- Exact volume per env from inverse mass ---
    tpu_mass = torch.zeros_like(inv_mass)
    tpu_active = is_tpu_node & (inv_mass > 1e-6)
    tpu_mass[tpu_active] = 1.0 / inv_mass[tpu_active]

    env_volume = torch.zeros(num_envs, device=device)
    for i in range(num_envs):
        env_mass = tpu_mass[is_tpu_node & (batch == i)].sum()
        env_volume[i] = env_mass / RHO_TPU

    # --- Length-weighted volume per edge ---
    L0_sum_per_env = torch.zeros(num_envs, device=device)
    L0_sum_per_env.index_add_(0, edge_batch, L0)
    V_per_edge = env_volume[edge_batch] * (L0 / L0_sum_per_env[edge_batch].clamp(min=1e-9))

    # --- Reference steel Y per env ---
    ref_steel_y = torch.zeros(num_envs, device=device)
    steel_counts = torch.zeros(num_envs, device=device)
    ref_steel_y.index_add_(0, batch[is_steel_node], ref_pos[is_steel_node, 1])
    steel_counts.index_add_(0, batch[is_steel_node], torch.ones(is_steel_node.sum(), device=device))
    ref_steel_y = ref_steel_y / steel_counts.clamp(min=1)

    # --- Material tensors for vectorized Ogden ---
    mu_t = torch.tensor(MU, dtype=torch.float32, device=device)
    alpha_t = torch.tensor(ALPHA, dtype=torch.float32, device=device)

    return {
        'src': src, 'dst': dst, 'L0': L0,
        'edge_batch': edge_batch,
        'V_per_edge': V_per_edge,
        'env_volume': env_volume,
        'ref_steel_y': ref_steel_y,
        'num_envs': num_envs,
        'mu': mu_t, 'alpha': alpha_t,
    }


def compute_batched_strain_energy(pos, ctx):
    """U_total = sum(W_i * V_per_edge_i) per environment."""
    src, dst, L0 = ctx['src'], ctx['dst'], ctx['L0']

    L_curr = (pos[src] - pos[dst]).norm(dim=1).clamp(min=1e-9)
    lambdas = L_curr / L0

    lam_e = lambdas.unsqueeze(0)
    a = ctx['alpha'].unsqueeze(1)
    m = ctx['mu'].unsqueeze(1)
    W = ((m / a) * (lam_e ** a + lam_e ** (-a) - 2.0)).sum(dim=0)

    n = ctx['num_envs']
    U = torch.zeros(n, device=pos.device)
    U.index_add_(0, ctx['edge_batch'], W * ctx['V_per_edge'])
    return U


def compute_impactor_displacement(pos, batched_graph, ctx):
    """Displacement of steel from reference config per env."""
    is_steel = batched_graph.node_attr[:, 0].bool()
    batch = batched_graph.batch
    num_envs = ctx['num_envs']

    s = torch.zeros(num_envs, device=pos.device)
    c = torch.zeros(num_envs, device=pos.device)
    s.index_add_(0, batch[is_steel], pos[is_steel, 1])
    c.index_add_(0, batch[is_steel], torch.ones(is_steel.sum(), device=pos.device))
    curr_mean_y = s / c.clamp(min=1)
    return curr_mean_y - ctx['ref_steel_y']


def derive_force_from_energy(energy_history, disp_history):
    """F = -dU/dy per environment per step."""
    energy = np.array(energy_history)
    disp = np.array(disp_history)

    forces = np.zeros_like(energy)
    for t in range(1, len(energy)):
        dU = energy[t] - energy[t - 1]
        dy = disp[t] - disp[t - 1]
        safe_dy = np.where(np.abs(dy) > 1e-10, dy, 1e-10)
        forces[t] = -dU / safe_dy
    forces[0] = forces[1] if len(forces) > 1 else 0.0
    return forces


def compute_cfe(force_history):
    """CFE = mean/peak over steady state (skip first 10%)."""
    skip = max(1, len(force_history) // 10)
    steady = force_history[skip:]
    peak = np.max(np.abs(steady), axis=0)
    mean = np.mean(np.abs(steady), axis=0)
    return np.where(peak > 1e-6, mean / peak, 0.0)


# =========================================================
# CPU WORKER
# =========================================================
def worker_wrapper(seed_config):
    data_dict = generate_rl_environment(seed_config)
    if data_dict is None:
        return None

    node_type = data_dict['node_type']
    is_fixed = data_dict['is_constraint']
    node_attributes = torch.cat([
        node_type.float().unsqueeze(1),
        (1 - node_type).float().unsqueeze(1),
        is_fixed.float().unsqueeze(1)
    ], dim=1)

    return Data(
        pos=data_dict['pos'],
        mesh_pos=data_dict['mesh_pos'],
        inv_mass=data_dict['inv_mass'].float().unsqueeze(1),
        velocities=torch.cat([data_dict['prev_velocity'], data_dict['velocity']], dim=1),
        node_attr=node_attributes,
        mask=data_dict['is_constraint'],
        num_impactors=data_dict['num_impactors'],
        face_index=data_dict['face_index'],
        edge_index=data_dict['edge_index'],
        world_edge_index=data_dict['world_edge_index']
    )


# =========================================================
# VISUALIZATION
# =========================================================
def plot_batch_grid(final_pos, batch_vector, node_type_vector, num_envs):
    print("\nRendering final environment grid...")
    cols = min(8, num_envs)
    rows = math.ceil(num_envs / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if num_envs == 1: axes = [axes]
    else: axes = axes.flatten()

    pos_np = final_pos.cpu().numpy() * 1000.0
    batch_np = batch_vector.cpu().numpy()
    types_np = node_type_vector.cpu().numpy()

    for i in range(num_envs):
        ax = axes[i]
        m = batch_np == i
        if not m.any():
            ax.set_title(f"Env {i} (Failed)", color='red'); ax.axis('off'); continue
        p, t = pos_np[m], types_np[m]
        ax.scatter(p[t[:,1]==1, 0], p[t[:,1]==1, 1], c='#00aaff', s=0.5)
        ax.scatter(p[t[:,0]==1, 0], p[t[:,0]==1, 1], c='#ff3333', s=1)
        ax.scatter(p[t[:,2]==1, 0], p[t[:,2]==1, 1], c='#33ff33', s=1)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Env {i}", fontsize=10)
        ax.set_xlim(0, BOX_SIZE); ax.set_ylim(-5, BOX_SIZE + 35)

    for j in range(num_envs, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig("batch_rollout_results.png", dpi=300)
    print("Saved 'batch_rollout_results.png'")
    plt.close()


def plot_force_and_cfe(forces_energy, disp_history, cfe_energy, num_envs):
    """Plots dU/dy force curves and CFE bar chart."""
    disp_arr = np.array(disp_history)
    disp_0 = disp_arr[0]
    colors = plt.cm.tab10(np.linspace(0, 1, num_envs))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # LEFT: dU/dy Force
    ax1 = axes[0]
    for i in range(num_envs):
        cd = np.abs(disp_arr[:, i] - disp_0[i]) * 1000.0
        ax1.plot(cd, np.abs(forces_energy[:, i]), color=colors[i],
                 label=f"Env {i} (CFE={cfe_energy[i]:.3f})")
    ax1.set_xlabel("Crush Displacement (mm)")
    ax1.set_ylabel("Force (N)")
    ax1.set_title("Force from Strain Energy (F = −dU/dy)")
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

    # RIGHT: CFE Bar Chart
    ax2 = axes[1]
    x = np.arange(num_envs)
    ax2.bar(x, cfe_energy, color='steelblue', edgecolor='black')
    ax2.axhline(y=1.0, color='green', ls='--', alpha=0.5, label='Ideal')
    ax2.set_xlabel("Environment"); ax2.set_ylabel("CFE")
    ax2.set_title("CFE Analysis")
    ax2.set_ylim(0, 1.1); ax2.set_xticks(x)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("force_analysis.png", dpi=300)
    print("Saved 'force_analysis.png'")
    plt.close()


def create_batch_animation(pos_snapshots, batch_vector, node_type_vector, num_envs, 
                           forces_energy, disp_history, filename="batch_rollout.mp4"):
    print("\n🎬 Generating combined mesh & force animation...")
    
    cols = min(8, num_envs)
    rows = 2 
    
    fig = plt.figure(figsize=(4 * cols, 8))
    gs = GridSpec(rows, cols, figure=fig, height_ratios=[1, 1])

    batch_np = batch_vector.cpu().numpy()
    types_np = node_type_vector.cpu().numpy()
    
    disp_arr = np.abs(np.array(disp_history) - np.array(disp_history)[0]) * 1000.0

    scatters = []
    lines = []
    dots = []
    
    for i in range(cols):
        m = batch_np == i
        
        # --- 1. MESH PLOT (TOP ROW) ---
        ax_mesh = fig.add_subplot(gs[0, i])
        if not m.any():
            ax_mesh.set_title(f"Env {i} (Failed)"); ax_mesh.axis('off')
            scatters.append(None); lines.append(None); dots.append(None)
            continue
            
        ax_mesh.set_aspect('equal'); ax_mesh.set_xticks([]); ax_mesh.set_yticks([])
        ax_mesh.set_title(f"Env {i} Mesh")
        ax_mesh.set_xlim(0, BOX_SIZE)
        ax_mesh.set_ylim(-5, BOX_SIZE + 35)
        
        p0 = pos_snapshots[0][m] * 1000.0
        t = types_np[m]
        
        c = np.where(t[:, 1] == 1, '#00aaff', np.where(t[:, 0] == 1, '#ff3333', '#33ff33'))
        s = ax_mesh.scatter(p0[:, 0], p0[:, 1], c=c, s=1)
        scatters.append((s, m))

        # --- 2. FORCE PLOT (BOTTOM ROW) ---
        ax_force = fig.add_subplot(gs[1, i])
        ax_force.set_title("Force vs Disp")
        ax_force.set_xlabel("Disp (mm)")
        if i == 0: ax_force.set_ylabel("Force (N)")
        
        max_disp = np.max(disp_arr[:, i])
        max_force = np.max(np.abs(forces_energy[:, i]))
        
        ax_force.set_xlim(-0.1, max_disp * 1.1 + 1e-3)
        ax_force.set_ylim(-0.1, max_force * 1.1 + 1e-3)
        ax_force.grid(True, alpha=0.3)
        
        line, = ax_force.plot([], [], 'steelblue', linewidth=2)
        dot,  = ax_force.plot([], [], 'ro', markersize=6) # Red tracking dot
        
        lines.append(line)
        dots.append(dot)

    plt.tight_layout()

    # --- ANIMATION UPDATE LOGIC ---
    def update(frame):
        artists = []
        p_frame = pos_snapshots[frame] * 1000.0
        
        for i in range(cols):
            if scatters[i] is not None:

                s, m = scatters[i]
                s.set_offsets(p_frame[m])
                artists.append(s)
                
                x_data = disp_arr[:frame+1, i]
                y_data = np.abs(forces_energy[:frame+1, i])
                
                lines[i].set_data(x_data, y_data)
                dots[i].set_data([x_data[-1]], [y_data[-1]]) 
                
                artists.extend([lines[i], dots[i]])
                
        return artists

    anim = FuncAnimation(fig, update, frames=len(pos_snapshots), blit=False, interval=50)
    anim.save(filename, dpi=150, writer='ffmpeg')
    print(f"Saved '{filename}'")
    plt.close()


# =========================================================
# MAIN
# =========================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"🖥️  Device: {device}")

    config = {
        'mode': 'rollout',
        'train': {'time_step': 1.0e-2},
        'model': {
            'spatial_dim': 2, 'node_in_dim': 8,
            'mesh_edge_in_dim': 8, 'world_edge_in_dim': 4,
            'hidden_dim': 128, 'output_dim': 2,
            'num_layers': 15, 'radius': 3.0e-3
        }
    }

    ckpt_path = "checkpoints/latest_model.pth" # Ensure path is correct for your system

    model = MeshGraphNet(config).to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        clean = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(clean, strict=True)
    model.eval()

    BATCH_SIZE = 8
    SEEDS_PER_CONFIG = 6
    NUM_STEPS = 250
    DT = config['train']['time_step']
    IMPACTOR_VEL = -0.005
    SAMPLE_INTERVAL = 3
    
    # Safe boundary to prevent seeds from merging with the outer frame
    MARGIN = 5.0 

    print(f"\nGenerating Base Voronoi config: Perfectly Centered Vertical Strut...")
    
    # 1. Generate ONE base design (A Vertical Strut)
    '''
    base_seeds = np.array([
        [BOX_SIZE * 0.25, BOX_SIZE * 0.50],  # Seed 1: Left middle
        [BOX_SIZE * 0.75, BOX_SIZE * 0.50]   # Seed 2: Right middle
    ])

    '''
    # 1. Generate ONE base design (A Vertical Strut)
    base_seeds = np.random.uniform(
        low=MARGIN, 
        high=BOX_SIZE - MARGIN, 
        size=(SEEDS_PER_CONFIG, 2)
    )
    

    print(f"Generating {BATCH_SIZE} variations (Base + {BATCH_SIZE-1} Perturbed with Unit Gaussian Noise)...")
    # 2. Env 0 is the clean, unperturbed vertical strut
    seed_batch = [base_seeds] 
    
    # 3. Apply N(0, 1) Gaussian noise to the rest
    for _ in range(BATCH_SIZE - 1):

        noise = np.random.normal(loc=0.0, scale=2.0, size=(SEEDS_PER_CONFIG, 2))
        perturbed_seeds = base_seeds + noise
        
        perturbed_seeds = np.clip(perturbed_seeds, MARGIN, BOX_SIZE - MARGIN)
        seed_batch.append(perturbed_seeds)

    num_cores = min(mp.cpu_count(), BATCH_SIZE)
    with mp.Pool(processes=num_cores) as pool:
        pyg_data_list = pool.map(worker_wrapper, seed_batch)

    valid_data_list = [d for d in pyg_data_list if d is not None]
    if not valid_data_list:
        print("All meshes failed."); return

    batched_graph = Batch.from_data_list(valid_data_list).to(device)
    num_envs = len(valid_data_list)

    print(f"   Nodes: {batched_graph.pos.size(0)} | Edges: {batched_graph.edge_index.size(1)}")

    is_steel = batched_graph.node_attr[:, 0].bool()
    is_tpu = ~is_steel
    is_fixed = batched_graph.mask.squeeze()

    energy_ctx = precompute_energy_context(batched_graph, valid_data_list, device)
    print(f"Volume per env (m³): {energy_ctx['env_volume'].cpu().numpy()}")

    energy_history = []
    disp_history = []
    pos_snapshots = []

    curr_pos = batched_graph.pos.clone()
    curr_prev_vel = batched_graph.velocities[:, :2].clone()
    curr_vel = batched_graph.velocities[:, 2:].clone()

    print(f"\nRolling out {NUM_STEPS} steps...")
    t_start = time.time()

    with torch.no_grad():
        for step in range(NUM_STEPS):
            batched_graph.pos = curr_pos
            batched_graph.velocities = torch.cat([curr_prev_vel, curr_vel], dim=-1)

            outputs = model(batched_graph, accumulate_stats=False)
            pred_accel_norm = outputs[0] if isinstance(outputs, tuple) else outputs
            batched_graph.world_edge_index = outputs[1]

            pred_accel = torch.zeros_like(pred_accel_norm)
            if is_tpu.sum() > 0:
                pred_accel[is_tpu] = model.absorber_target_normalizer.inverse(
                    pred_accel_norm[is_tpu])
            pred_accel[is_steel] = 0.0
            pred_accel[is_fixed] = 0.0

            next_vel = curr_vel + pred_accel
            next_vel[is_steel, 0] = 0.0
            next_vel[is_steel, 1] = IMPACTOR_VEL
            next_vel[is_fixed] = 0.0
            next_pos = curr_pos + next_vel * DT

            curr_prev_vel = curr_vel
            curr_vel = next_vel
            curr_pos = next_pos

            if step % SAMPLE_INTERVAL == 0 or step == NUM_STEPS - 1:
                U = compute_batched_strain_energy(curr_pos, energy_ctx)
                d = compute_impactor_displacement(curr_pos, batched_graph, energy_ctx)
                energy_history.append(U.cpu().numpy())
                disp_history.append(d.cpu().numpy())
                pos_snapshots.append(curr_pos.detach().cpu().numpy())

            if step % 50 == 0:
                max_a = pred_accel.abs().max().item()
                print(f"   Step {step:3d}: max_accel={max_a:.6f}")

    print(f"\nDone in {time.time() - t_start:.1f}s")

    # =========================================================
    # FORCES & CFE
    # =========================================================
    forces_energy = derive_force_from_energy(energy_history, disp_history)
    cfe_energy = compute_cfe(forces_energy)

    print(f"\n--- SENSITIVITY ANALYSIS ---")
    print(f"{'Env':>4s} | {'CFE dU/dy':>10s} | {'Peak dU/dy':>11s} | {'Mean dU/dy':>11s} | {'Status':>12s}")
    print("-" * 62)
    for i in range(num_envs):
        pe = np.max(np.abs(forces_energy[:, i]))
        me = np.mean(np.abs(forces_energy[:, i]))
        status = "BASE DESIGN" if i == 0 else "PERTURBED"
        print(f"  {i:2d}  | {cfe_energy[i]:10.4f} | {pe:11.2f} | {me:11.2f} | {status}")

    if num_envs > 1:
        cfe_var = np.var(cfe_energy[1:])
        cfe_std = np.std(cfe_energy[1:])
        print(f"\n   Base CFE:       {cfe_energy[0]:.4f}")
        print(f"   Perturbed Mean: {np.mean(cfe_energy[1:]):.4f}")
        print(f"   Perturbed Std:  {cfe_std:.4f}")

    # =========================================================
    # PLOTS & EXPORT
    # =========================================================
    plot_force_and_cfe(forces_energy, disp_history, cfe_energy, num_envs)

    plot_batch_grid(curr_pos, batched_graph.batch, batched_graph.node_attr, num_envs)
    create_batch_animation(
            pos_snapshots, 
            batched_graph.batch, 
            batched_graph.node_attr, 
            num_envs, 
            forces_energy, 
            disp_history
        )

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()