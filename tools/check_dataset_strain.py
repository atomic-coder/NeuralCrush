import os
import glob
import torch
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict

# =========================================================
# WORKER FUNCTION (Processes an entire simulation trajectory)
# =========================================================
def process_trajectory(sim_files):
    """
    Loads all frames for a single simulation and finds the 
    maximum edge strain and minimum area ratio across the whole sequence.
    """
    max_sim_strain = 0.0
    min_sim_area_ratio = 999.0
    has_elements = False

    sim_files = sorted(sim_files)
    
    for path in sim_files:
        try:
            data = torch.load(path, map_location='cpu')
            
            pos = data['pos']
            mesh_pos = data['mesh_pos']
            node_type = data['node_type']
            edge_index = data['edge_index']
            elements = data.get('elements', None)
            
            # 0 is TPU/absorber, 1 is Steel/Impactor
            is_tpu = (node_type == 0)
            
            # --- 1. EDGE STRAIN ---
            src, dst = edge_index
            tpu_edges = is_tpu[src] & is_tpu[dst]
            
            if tpu_edges.any():
                src_tpu, dst_tpu = src[tpu_edges], dst[tpu_edges]
                
                L0 = (mesh_pos[src_tpu] - mesh_pos[dst_tpu]).norm(dim=1).clamp(min=1e-9)
                L_curr = (pos[src_tpu] - pos[dst_tpu]).norm(dim=1)
                
                strain = torch.abs(L_curr - L0) / L0
                frame_max_strain = strain.max().item()
                if frame_max_strain > max_sim_strain:
                    max_sim_strain = frame_max_strain

            # --- 2. ELEMENT AREA RATIO ---
            if elements is not None and elements.numel() > 0:
                has_elements = True
                is_tpu_elem = is_tpu[elements[:, 0]] & is_tpu[elements[:, 1]] & is_tpu[elements[:, 2]]
                
                if is_tpu_elem.any():
                    tpu_elems = elements[is_tpu_elem]
                    
                    # Reference Area (A0)
                    p1_0, p2_0, p3_0 = mesh_pos[tpu_elems[:, 0]], mesh_pos[tpu_elems[:, 1]], mesh_pos[tpu_elems[:, 2]]
                    A0 = 0.5 * ((p2_0[:, 0] - p1_0[:, 0]) * (p3_0[:, 1] - p1_0[:, 1]) - 
                                (p3_0[:, 0] - p1_0[:, 0]) * (p2_0[:, 1] - p1_0[:, 1]))
                    
                    # Current Area (A)
                    p1, p2, p3 = pos[tpu_elems[:, 0]], pos[tpu_elems[:, 1]], pos[tpu_elems[:, 2]]
                    A = 0.5 * ((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - 
                               (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1]))
                    
                    # Area Ratio
                    area_ratio = A / A0.clamp(min=1e-9, max=None if A0.mean() > 0 else -1e-9)
                    frame_min_ratio = area_ratio.min().item()
                    
                    if frame_min_ratio < min_sim_area_ratio:
                        min_sim_area_ratio = frame_min_ratio
                        
        except Exception as e:
            print(f"Error reading {os.path.basename(path)}: {e}")
            continue

    return max_sim_strain, min_sim_area_ratio, has_elements


# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    target_dir = "" # Path to your dataset
    
    search_path = os.path.join(target_dir, "*.pt")
    all_files = glob.glob(search_path)
    all_files = [f for f in all_files if not os.path.basename(f).startswith("._")]
    
    if not all_files:
        print(f"No .pt files found in {target_dir}")
        return
        
    print(f"Found {len(all_files)} total frames.")
    
    sims = defaultdict(list)
    for f in all_files:
        basename = os.path.basename(f)
        sim_id = basename.split('_frame_')[0] if '_frame_' in basename else "sim_0"
        sims[sim_id].append(f)
        
    sim_groups = list(sims.values())
    num_sims = len(sim_groups)
    print(f"Grouped into {num_sims} physical simulations.")
    
    max_strains = []
    min_area_ratios = []
    total_elements_found = 0
    
    # Multiprocessing across simulations
    num_cores = min(mp.cpu_count(), 16) # Cap at 16 to avoid IO bottlenecks
    print(f"Processing in parallel using {num_cores} workers...")
    
    with mp.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(process_trajectory, sim_groups), total=num_sims, desc="Scanning Trajectories"))

    for max_strain, min_ratio, has_elems in results:
        max_strains.append(max_strain)
        if has_elems:
            min_area_ratios.append(min_ratio)
            total_elements_found += 1
            
    # =========================================================
    # SUMMARY REPORT
    # =========================================================
    print("\n" + "="*50)
    print("GROUND TRUTH DATASET STRAIN ANALYSIS")
    print("="*50)
    
    if max_strains:
        mean_max_strain = np.mean(max_strains)
        absolute_max_strain = np.max(max_strains)
        print(f"   EDGE STRAIN (ΔL/L0)")
        print(f"   Mean Max Strain per Sim:  {mean_max_strain:.4f}")
        print(f"   Absolute Max in Dataset:  {absolute_max_strain:.4f}")
        print(f"   -> Suggests RL Penalty if Strain > {absolute_max_strain * 1.2:.2f}")
    
    print("-" * 50)
    
    if total_elements_found > 0:
        mean_min_ratio = np.mean(min_area_ratios)
        absolute_min_ratio = np.min(min_area_ratios)
        inversions = sum(1 for r in min_area_ratios if r <= 0.0)
        
        print(f"   ELEMENT AREA RATIO (A/A0)")
        print(f"   Mean Min Ratio per Sim:   {mean_min_ratio:.4f}")
        print(f"   Absolute Min in Dataset:  {absolute_min_ratio:.4f}")
        print(f"   Simulations inverted:     {inversions} / {num_sims}")
        if absolute_min_ratio > 0.0:
            print(f"   -> Suggests RL Penalty if Area Ratio < {absolute_min_ratio * 0.8:.3f}")
        else:
            print(f"   -> Ground truth naturally inverts elements. Threshold must be < 0.0.")
    else:
        print("   ELEMENT AREA RATIO (A/A0)")
        print("   No valid element topologies found in dataset.")
        
    print("="*50 + "\n")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()