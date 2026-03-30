import torch
import numpy as np
import time
import multiprocessing as mp
import os
import sys
from tqdm import tqdm
from torch_geometric.data import Batch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.model import MeshGraphNet
from src.data_formatter import MeshData as Data
from rl_environment import generate_rl_environment, BOX_SIZE

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
        world_edge_index=data_dict['world_edge_index'],
        elements=data_dict.get('elements', None)
    )

# =========================================================
# STRAIN TRACKING MATH
# =========================================================
def precompute_strain_context(batched_graph):
    """Isolates TPU edges and elements, computes reference lengths and areas."""
    is_tpu = batched_graph.node_attr[:, 1].bool()
    edge_index = batched_graph.edge_index
    ref_pos = batched_graph.mesh_pos
    batch = batched_graph.batch

    # 1. TPU Edges
    tpu_edge_mask = is_tpu[edge_index[0]] & is_tpu[edge_index[1]]
    src, dst = edge_index[:, tpu_edge_mask]
    edge_batch = batch[src]
    L0 = (ref_pos[src] - ref_pos[dst]).norm(dim=1).clamp(min=1e-9)

    # 2. TPU Elements (Triangles)
    elements = batched_graph.elements
    has_elements = elements is not None and elements.numel() > 0
    
    tpu_elems, elem_batch, A0 = None, None, None
    if has_elements:
        is_tpu_elem = is_tpu[elements[:, 0]] & is_tpu[elements[:, 1]] & is_tpu[elements[:, 2]]
        tpu_elems = elements[is_tpu_elem]
        elem_batch = batch[tpu_elems[:, 0]]
        
        p1, p2, p3 = ref_pos[tpu_elems[:, 0]], ref_pos[tpu_elems[:, 1]], ref_pos[tpu_elems[:, 2]]
        # 2D signed area: 0.5 * ((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
        A0 = 0.5 * ((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - 
                    (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1]))

    return {
        'src': src, 'dst': dst, 'L0': L0, 'edge_batch': edge_batch,
        'has_elements': has_elements,
        'tpu_elems': tpu_elems, 'elem_batch': elem_batch, 'A0': A0,
        'num_envs': batched_graph.num_graphs
    }

def compute_current_strains(pos, ctx):
    """Calculates current edge strain and element area ratio."""
    # Edge Strain: |L - L0| / L0
    L = (pos[ctx['src']] - pos[ctx['dst']]).norm(dim=1)
    edge_strain = torch.abs(L - ctx['L0']) / ctx['L0']
    
    # Area Ratio: A / A0 (If < 0, element inverted)
    area_ratio = None
    if ctx['has_elements']:
        tpu_elems = ctx['tpu_elems']
        p1, p2, p3 = pos[tpu_elems[:, 0]], pos[tpu_elems[:, 1]], pos[tpu_elems[:, 2]]
        A = 0.5 * ((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - 
                   (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1]))
        area_ratio = A / ctx['A0'].clamp(min=1e-9, max=None if ctx['A0'].mean() > 0 else -1e-9)

    return edge_strain, area_ratio

# =========================================================
# MAIN
# =========================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Device: {device}")

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

    ckpt_path = "" # Paste path to model checkpoint
    model = MeshGraphNet(config).to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        clean = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(clean, strict=True)
    model.eval()

    TOTAL_ENVS = 1000
    BATCH_SIZE = 4
    SEEDS_PER_CONFIG = 6
    NUM_STEPS = 200
    DT = config['train']['time_step']
    IMPACTOR_VEL = -0.005

    global_max_edge_strains = []
    global_min_area_ratios = []

    num_batches = TOTAL_ENVS // BATCH_SIZE
    print(f"\nProcessing {TOTAL_ENVS} environments in {num_batches} batches...")

    for batch_idx in tqdm(range(num_batches), desc="Batches"):
        # Generate configs
        seed_batch = [np.random.uniform(0, BOX_SIZE, size=(SEEDS_PER_CONFIG, 2)) for _ in range(BATCH_SIZE)]
        
        num_cores = min(mp.cpu_count(), BATCH_SIZE)
        with mp.Pool(processes=num_cores) as pool:
            pyg_data_list = pool.map(worker_wrapper, seed_batch)

        valid_data_list = [d for d in pyg_data_list if d is not None]
        if not valid_data_list:
            continue

        batched_graph = Batch.from_data_list(valid_data_list).to(device)
        is_steel = batched_graph.node_attr[:, 0].bool()
        is_tpu = ~is_steel
        is_fixed = batched_graph.mask.squeeze()

        strain_ctx = precompute_strain_context(batched_graph)
        num_envs = strain_ctx['num_envs']

        # Trackers per environment in this batch
        env_max_edge_strain = torch.zeros(num_envs, device=device)
        env_min_area_ratio = torch.ones(num_envs, device=device) * 999.0

        curr_pos = batched_graph.pos.clone()
        curr_prev_vel = batched_graph.velocities[:, :2].clone()
        curr_vel = batched_graph.velocities[:, 2:].clone()

        with torch.no_grad():
            for step in range(NUM_STEPS):
                batched_graph.pos = curr_pos
                batched_graph.velocities = torch.cat([curr_prev_vel, curr_vel], dim=-1)

                outputs = model(batched_graph, accumulate_stats=False)
                pred_accel_norm = outputs[0] if isinstance(outputs, tuple) else outputs
                batched_graph.world_edge_index = outputs[1]

                pred_accel = torch.zeros_like(pred_accel_norm)
                if is_tpu.sum() > 0:
                    pred_accel[is_tpu] = model.absorber_target_normalizer.inverse(pred_accel_norm[is_tpu])
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

                # Calculate metrics
                edge_strain, area_ratio = compute_current_strains(curr_pos, strain_ctx)
                
                for i in range(num_envs):
                    edge_mask = strain_ctx['edge_batch'] == i
                    if edge_mask.any():
                        env_max_edge_strain[i] = torch.max(env_max_edge_strain[i], edge_strain[edge_mask].max())
                    
                    if area_ratio is not None:
                        elem_mask = strain_ctx['elem_batch'] == i
                        if elem_mask.any():
                            env_min_area_ratio[i] = torch.min(env_min_area_ratio[i], area_ratio[elem_mask].min())

        global_max_edge_strains.extend(env_max_edge_strain.cpu().numpy().tolist())
        if strain_ctx['has_elements']:
            global_min_area_ratios.extend(env_min_area_ratio.cpu().numpy().tolist())

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n--- ROLLOUT STRAIN SUMMARY ---")
    if len(global_max_edge_strains) > 0:
        high_strain_envs = sum(1 for s in global_max_edge_strains if s > 0.5)
        print(f"Edge Strain (ΔL/L0)    | Mean Max: {np.mean(global_max_edge_strains):.4f} | Absolute Max: {np.max(global_max_edge_strains):.4f}")
        print(f"Extreme Strain (>0.5)  | {high_strain_envs} out of {len(global_max_edge_strains)} environments exceeded 0.5 max strain.")
    
    if len(global_min_area_ratios) > 0:
        inversions = sum(1 for r in global_min_area_ratios if r <= 0.0)
        print(f"Area Ratio (A/A0)      | Mean Min: {np.mean(global_min_area_ratios):.4f} | Absolute Min: {np.min(global_min_area_ratios):.4f}")
        print(f"Catastrophic Failures  | {inversions} out of {len(global_min_area_ratios)} environments inverted an element.")
    else:
        print("Area Ratio (A/A0)      | N/A (No element mapping found in data_dict)")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()