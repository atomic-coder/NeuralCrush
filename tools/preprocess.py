import numpy as np
import torch
import re
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
import os
import glob
from tools.utils import build_world_edges  

# ==========================================
# 1. NASTRAN PARSER (Regex Regex Fallback)
# ==========================================
def parse_nastran_file(nas_path):
    """
    Robustly parses Nastran .nas files. Uses Regex to safely separate 
    coordinates when extreme precision causes COMSOL to overwrite column spaces.
    """
    with open(nas_path, 'r') as f:
        lines = f.readlines()
    
    mesh_vertices = []
    node_ids = []
    tri_elements = []
    
    float_pattern = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip('\n').strip()
        
        if line.startswith('$') or not line:
            i += 1
            continue
            
        if line.startswith('GRID'):
            try:

                clean_line = line.replace('*CONT', '').replace('GRID*', '').replace('GRID', '').replace(',', ' ')
                
                numbers = float_pattern.findall(clean_line)
                
                if len(numbers) >= 3:
                    nid = int(numbers[0])
                    x = float(numbers[1])
                    y = float(numbers[2])
                    
                    node_ids.append(nid)
                    mesh_vertices.append([x, y])
                
                if i + 1 < len(lines) and lines[i+1].strip().startswith('*CONT'):
                    i += 1
            except Exception as e:
                print(f"Warning: parsing GRID line {i}: {e}")

        elif line.startswith('CTRIA3'):
            try:
                clean_line = line.replace('CTRIA3', '').replace(',', ' ')
                numbers = float_pattern.findall(clean_line)
                
                if len(numbers) >= 4:
                    # In standard formatting, the last 3 numbers are always the nodes
                    n1 = int(numbers[-3])
                    n2 = int(numbers[-2])
                    n3 = int(numbers[-1])
                    tri_elements.append([n1, n2, n3])
            except Exception as e:
                print(f"Warning: parsing CTRIA3 line {i}: {e}")
        i += 1

    return np.array(mesh_vertices), np.array(node_ids, dtype=np.int64), np.array(tri_elements, dtype=np.int64)

# ==========================================
# 2. TRAJECTORY PARSER (2D)
# ==========================================
def parse_trajectory_file(traj_path, expected_num_nodes):
    with open(traj_path, 'r') as f:
        lines = f.readlines()
    
    data_start_line = 0
    for i, line in enumerate(lines):
        if not line.strip().startswith('%') and line.strip():
            data_start_line = i
            break
            
    all_values = []
    for line in lines[data_start_line:]:
        clean_line = line.split('%')[0]
        for token in clean_line.split():
            try:
                all_values.append(float(token))
            except ValueError:
                pass
    
    all_values = np.array(all_values, dtype=np.float64)
    total_len = len(all_values)
    
    if total_len == 0:
        raise ValueError("No trajectory data found.")
    if total_len % expected_num_nodes != 0:
        raise ValueError(f"Total data length {total_len} is not divisible by node count {expected_num_nodes}")
        
    num_cols = total_len // expected_num_nodes
    data = all_values.reshape((expected_num_nodes, num_cols))
    
    # 2D: first 2 columns are mesh X, Y
    mesh_coords = data[:, :2]
    remaining_cols = data[:, 2:]
    
    # Time blocks are 3 columns wide: x(t), y(t), is_steel(t)
    if remaining_cols.shape[1] % 4 != 0:
        raise ValueError("Trajectory columns not divisible by 3")
        
    num_timesteps = remaining_cols.shape[1] // 4
    
    timeblocks = []
    for t in range(num_timesteps):
        start_col = t * 4
        pos = remaining_cols[:, start_col:start_col+2]
        is_steel = remaining_cols[:, start_col+2].astype(int)
        reaction_y = remaining_cols[:, start_col+3]
        timeblocks.append({'pos': pos, 'is_steel': is_steel, 'reaction_y': reaction_y})
        
    return mesh_coords, timeblocks

# ==========================================
# 3. HELPER LOGIC (2D Adaptation)
# ==========================================
def get_skin_edges(elements, mesh_vertices):
    """Extracts 2D boundary edges."""
    edges_local_idx = np.array([[0, 1], [1, 2], [2, 0]])
    all_edges = elements[:, edges_local_idx].reshape(-1, 2)
    sorted_edges = np.sort(all_edges, axis=1)
    
    void_dt = np.dtype((np.void, sorted_edges.dtype.itemsize * sorted_edges.shape[1]))
    edges_void = np.ascontiguousarray(sorted_edges).view(void_dt).reshape(-1)
    
    _, indices, counts = np.unique(edges_void, return_index=True, return_counts=True)
    skin_indices = indices[counts == 1]
    skin_edges = all_edges[skin_indices]
    
    parent_tri_indices = skin_indices // 3
    parent_tris = elements[parent_tri_indices]
    
    edge_coords = mesh_vertices[skin_edges]
    elem_coords = mesh_vertices[parent_tris]
    
    edge_centers = np.mean(edge_coords, axis=1)
    elem_centers = np.mean(elem_coords, axis=1)
    
    outward_vec = edge_centers - elem_centers
    v1 = edge_coords[:, 1] - edge_coords[:, 0]
    
    normals = np.column_stack([-v1[:, 1], v1[:, 0]])
    
    dot_prod = np.sum(normals * outward_vec, axis=1)
    flip_mask = dot_prod < 0
    
    if np.any(flip_mask):
        skin_edges[flip_mask, 0], skin_edges[flip_mask, 1] = \
        skin_edges[flip_mask, 1].copy(), skin_edges[flip_mask, 0].copy()
        
    return skin_edges

def load_and_process_masks(mask_path, mesh_pos):
    try:
        data = np.loadtxt(mask_path, comments='%')
        if data.ndim == 1: data = data.reshape(1, -1)
        
        mask_coords = data[:, 0:2]
        mask_values = data[:, -1]
        
        coord_to_best_idx = {}
        for i in range(len(mask_coords)):
            coord_key = tuple(mask_coords[i])
            if coord_key not in coord_to_best_idx:
                coord_to_best_idx[coord_key] = i
            else:
                existing_idx = coord_to_best_idx[coord_key]
                if abs(mask_values[i]) > abs(mask_values[existing_idx]):
                    coord_to_best_idx[coord_key] = i
        
        unique_indices = sorted(coord_to_best_idx.values())
        mask_coords_dedup = mask_coords[unique_indices]
        mask_values_dedup = mask_values[unique_indices]
        
        tree_mask = cKDTree(mask_coords_dedup)
        dists_mask, indices_mask = tree_mask.query(mesh_pos, k=1)
        mapped_values = mask_values_dedup[indices_mask]
        
        is_constraint = ((dists_mask < 1e-3) & (np.abs(mapped_values) > 1e-6))
        return is_constraint.astype(bool)
    except Exception as e:
        print(f"  ⚠️ Error processing masks: {e}")
        return None

def build_edge_index_from_mesh(tri_elements, node_ids):
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    edges = set()
    tri_edges = [(0, 1), (0, 2), (1, 2)]
    for tri in tri_elements:
        try:
            indices = [node_id_to_idx[nid] for nid in tri]
        except KeyError: continue
        for i, j in tri_edges:
            edge = tuple(sorted([indices[i], indices[j]]))
            edges.add(edge)
            edges.add((edge[1], edge[0]))
    return np.array(list(edges), dtype=np.int64).T

def match_trajectory_to_mesh_order(mesh_vertices, traj_mesh_coords, traj_data):
    tree = cKDTree(traj_mesh_coords)
    distances, indices = tree.query(mesh_vertices, k=1)
    if np.max(distances) > 1e-3:
        print(f"Warning: Mesh matching max distance {np.max(distances):.6f}")
    return {
        'pos': traj_data['pos'][indices],
        'is_steel': traj_data['is_steel'][indices]
    }, indices

def geometric_constraint_heuristic(mesh_vertices, tolerance=1e-3):
    min_y = np.min(mesh_vertices[:, 1])
    is_constraint = np.abs(mesh_vertices[:, 1] - min_y) < tolerance
    return is_constraint.astype(bool)

def compute_nodal_inv_mass(mesh_vertices, mapped_elements, is_steel):
    """
    Computes the lumped inverse mass for each node based on connected 2D triangle areas.
    """
    RHO_STEEL = 7850.0
    RHO_TPU = 1210.0
    THICKNESS = 0.05
    
    num_nodes = len(mesh_vertices)
    node_areas = np.zeros(num_nodes, dtype=np.float64)
    
    # 2. Vectorized Triangle Area Calculation
    pts = mesh_vertices[mapped_elements]  # shape: (num_tris, 3, 2)
    p1, p2, p3 = pts[:, 0, :], pts[:, 1, :], pts[:, 2, :]
    
    # Area = 0.5 * |(x2-x1)(y3-y1) - (x3-x1)(y2-y1)|
    areas = 0.5 * np.abs(
        (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - 
        (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
    )
    
    # 3. Distribute 1/3 of the area to each connected vertex
    third_areas = areas / 3.0
    np.add.at(node_areas, mapped_elements[:, 0], third_areas)
    np.add.at(node_areas, mapped_elements[:, 1], third_areas)
    np.add.at(node_areas, mapped_elements[:, 2], third_areas)
    
    # 4. Multiply by density to get Mass
    masses = np.zeros(num_nodes, dtype=np.float64)
    masses[is_steel] = node_areas[is_steel] * RHO_STEEL * THICKNESS
    masses[~is_steel] = node_areas[~is_steel] * RHO_TPU * THICKNESS
    
    # 5. Invert (Add tiny epsilon to prevent division by zero)
    inv_mass = 1.0 / (masses + 1e-8)
    
    return inv_mass

# ==========================================
# 4. PHYSICS PROCESSING
# ==========================================
def compute_smooth_derivatives(pos_sequence, dt):
    pos_smooth = pos_sequence.astype(np.float64)

    # 1. Velocity: v[t] = (p[t] - p[t-1]) / dt
    vel_input = np.zeros_like(pos_smooth)
    vel_input[1:] = (pos_smooth[1:] - pos_smooth[:-1])/dt
    vel_input[0] = vel_input[1] 

    # 2. Delta V: a[t] = (v[t+1] - v[t])
    acc_target = np.zeros_like(pos_smooth)
    acc_target[:-1] = (vel_input[1:] - vel_input[:-1])
    
    # 3. Boundary fixes
    acc_target[0] = acc_target[1]
    acc_target[-1] = acc_target[-2]

    return pos_smooth, vel_input, acc_target

# ==========================================
# 5. MAIN PROCESSOR
# ==========================================
def process_nastran_to_pt(mesh_path, trajectory_path, mask_path, output_dir, file_prefix, dt=1.0e-5, contact_radius=0.015):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Parse Mesh
    mesh_vertices, node_ids, tri_elements = parse_nastran_file(mesh_path)
    if len(mesh_vertices) == 0:
        print(f"❌ Error: No nodes found in {mesh_path}")
        return

    # 2. Parse Trajectory
    try:
        traj_mesh_coords, timeblocks = parse_trajectory_file(trajectory_path, len(mesh_vertices))
    except Exception as e:
        print(f"❌ Trajectory Error in {file_prefix}: {e}")
        return

    # 3. Build Graph Structure
    edge_index_np = build_edge_index_from_mesh(tri_elements, node_ids)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    
    edge_index_flip = torch.stack([edge_index[1], edge_index[0]], dim=0)
    edge_index = torch.unique(torch.cat([edge_index, edge_index_flip], dim=1), dim=1)
    
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    mapped_elements = []
    for tri in tri_elements:
        try:
            mapped_elements.append([node_id_to_idx[nid] for nid in tri])
        except KeyError: continue
    mapped_elements = np.array(mapped_elements, dtype=np.int64)
    
    skin_edges_np = get_skin_edges(mapped_elements, mesh_vertices)
    face_index_tensor = torch.tensor(skin_edges_np.T, dtype=torch.long)

    # 4. Masks
    if mask_path and os.path.exists(mask_path):
        is_constraint_np = load_and_process_masks(mask_path, mesh_vertices)
    else:
        is_constraint_np = geometric_constraint_heuristic(mesh_vertices)
    if is_constraint_np is None:
        is_constraint_np = geometric_constraint_heuristic(mesh_vertices)

    # 5. Build Sequences
    full_pos_list = []
    full_steel_list = []
    full_reaction_y_list = []
    _, sort_indices = match_trajectory_to_mesh_order(mesh_vertices, traj_mesh_coords, timeblocks[0])
    
    for tb in timeblocks:
        full_pos_list.append(tb['pos'][sort_indices])
        full_steel_list.append(tb['is_steel'][sort_indices])
        full_reaction_y_list.append(tb['reaction_y'][sort_indices])
        
    raw_pos_sequence = np.array(full_pos_list)
    steel_mask_sequence = np.array(full_steel_list)
    full_reaction_y_sequence = np.array(full_reaction_y_list)

    static_is_steel = steel_mask_sequence[0].astype(bool)

    inv_mass_np = compute_nodal_inv_mass(mesh_vertices * 0.001, mapped_elements, static_is_steel)
    inv_mass = inv_mass_np

    # SCALING
    SCALING_FACTOR = 1 # Scaling control to increase dt and reduce amplification of noise added to inputs and targets during training.
    scale = 0.001 
    raw_pos_sequence = raw_pos_sequence * SCALING_FACTOR * scale
    mesh_vertices = mesh_vertices * SCALING_FACTOR * scale

    scaled_dt = dt * SCALING_FACTOR 
    print(f"  Scaling Model by {SCALING_FACTOR}x.")

    # DECIMATION
    SKIP_FACTOR = 1 # To subsample frames in dataset
    raw_pos_sequence = raw_pos_sequence[::SKIP_FACTOR]
    steel_mask_sequence = steel_mask_sequence[::SKIP_FACTOR]
    full_reaction_y_sequence = full_reaction_y_sequence[::SKIP_FACTOR]
    effective_dt = scaled_dt * SKIP_FACTOR
    print(f"  New Physics dt: {scaled_dt:.4f}s")
    
    # 6. Compute Physics
    pos_seq, vel_seq, acc_seq = compute_smooth_derivatives(raw_pos_sequence, dt=effective_dt)

    # 7. Save Frames
    valid_frames_count = 0
    
    for t in range(1, len(pos_seq) - 1):
        pos_curr = pos_seq[t].copy()
        vel_curr = vel_seq[t].copy()
        acc_curr = acc_seq[t].copy()
        acc_next = acc_seq[t + 1].copy()
        
        node_type = steel_mask_sequence[t]
        reaction_y = full_reaction_y_sequence[t]
        num_impactors = np.sum(node_type).item()
        
        prev_vel = vel_seq[t - 1].copy()
        
        # --- CONSTRAINT LOGIC ---
        final_constraint = is_constraint_np 
        prev_vel[final_constraint] = 0.0
        vel_curr[final_constraint] = 0.0
        acc_curr[final_constraint] = 0.0
        acc_next[final_constraint] = 0.0

        prev_vel[np.abs(prev_vel) < 1e-6] = 0.0
        vel_curr[np.abs(vel_curr) < 1e-6] = 0.0
        acc_curr[np.abs(acc_curr) < 1e-6] = 0.0
        acc_next[np.abs(acc_next) < 1e-6] = 0.0

        pos_tensor = torch.tensor(pos_curr, dtype=torch.float32)
        batch_dummy = torch.zeros(pos_tensor.size(0), dtype=torch.long)
        
        world_edge_index = build_world_edges(
            pos=pos_tensor,
            batch=batch_dummy,
            face_index=face_index_tensor,
            mesh_edge_index=edge_index,
            radius=contact_radius
        )
        
        if world_edge_index.numel() > 0:
            world_edge_index_flip = torch.stack([world_edge_index[1], world_edge_index[0]], dim=0)
            world_edge_index = torch.unique(torch.cat([world_edge_index, world_edge_index_flip], dim=1), dim=1)

        data = {
            'pos': pos_tensor,
            'mesh_pos': torch.tensor(mesh_vertices, dtype=torch.float32),
            'inv_mass': torch.tensor(inv_mass, dtype=torch.float32),
            'prev_velocity': torch.tensor(prev_vel, dtype=torch.float32),
            'velocity': torch.tensor(vel_curr, dtype=torch.float32),
            'target_accel': torch.tensor(acc_curr, dtype=torch.float32),
            'reaction_y': torch.tensor(reaction_y, dtype=torch.float32),
            'target_accel_next': torch.tensor(acc_next, dtype=torch.float32),
            'node_type': torch.tensor(node_type, dtype=torch.long),
            'is_constraint': torch.tensor(final_constraint, dtype=torch.bool),
            'num_impactors': torch.tensor([num_impactors], dtype=torch.long),
            'face_index': face_index_tensor,
            'edge_index': edge_index,
            'world_edge_index': world_edge_index
        }
        
        output_path = os.path.join(output_dir, f'{file_prefix}_frame_{t:04d}.pt')
        torch.save(data, output_path)
        valid_frames_count += 1

    print(f"✓ {file_prefix}: Saved {valid_frames_count} frames (Scaled {SCALING_FACTOR}x)")

# ==========================================
# 6. EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    # Paste paths to raw GT exports and output folder
    dataset_root = ""
    output_root = ""

    for split in ['train', 'val']:
        split_dir = os.path.join(dataset_root, split)
        output_split_dir = os.path.join(output_root, split)
        os.makedirs(output_split_dir, exist_ok=True)
        
        if not os.path.exists(split_dir): continue
            
        print(f"\n=== PROCESSING SPLIT: {split.upper()} ===")
        sample_folders = sorted([f for f in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, f))])
        
        for sample_name in sample_folders:
            sample_path = os.path.join(split_dir, sample_name)
            mesh_file = os.path.join(sample_path, "mesh.nas")
            traj_file = os.path.join(sample_path, "trajectory_data.txt")
            mask_file = os.path.join(sample_path, "masks.txt")
            
            if not os.path.exists(mesh_file) or not os.path.exists(traj_file):
                continue
                
            process_nastran_to_pt(
                mesh_path=mesh_file,
                trajectory_path=traj_file,
                mask_path=mask_file,
                output_dir=output_split_dir,
                file_prefix=sample_name, 
                dt=1.0e-2,              # replace with appropriate dt
                contact_radius=3.0e-3   # for when to start building world edges
            )
            
    print("\nAll datasets processed successfully.")