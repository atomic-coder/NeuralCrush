import torch
import torch.nn.functional as F

@torch.compiler.disable
@torch.no_grad()
def build_world_edges(pos, batch, face_index, mesh_edge_index, radius, tolerance=-0.2):
    """
    Maximum efficiency GPU world edge builder.
    Now mathematically scale-invariant, supports 2D/3D, AND returns bidirectional edges!
    """
    device = pos.device
    dim = pos.shape[1]  
    
    # --- A. COMPUTE DYNAMIC NORMALS ---
    f_pos = pos[face_index] 
    
    if dim == 3:
        v1 = f_pos[1] - f_pos[0]
        v2 = f_pos[2] - f_pos[0]
        
        v1_n = F.normalize(v1, dim=1, eps=1e-8)
        v2_n = F.normalize(v2, dim=1, eps=1e-8)
        face_normals = torch.cross(v1_n, v2_n, dim=1) 
        num_nodes_per_face = 3
        
    elif dim == 2:
        v = f_pos[1] - f_pos[0]
        v_n = F.normalize(v, dim=1, eps=1e-8)
        face_normals = torch.stack([-v_n[:, 1], v_n[:, 0]], dim=1)
        num_nodes_per_face = 2
        
    else:
        raise ValueError(f"Unsupported spatial dimension: {dim}. Expected 2 or 3.")

    # Scatter to Nodes (Accumulate)
    node_accum_normals = torch.zeros_like(pos)
    for i in range(num_nodes_per_face):
        node_accum_normals.index_add_(0, face_index[i], face_normals)
        
    # The "Surfaceness" Score
    node_mag = node_accum_normals.norm(dim=1) 
    node_normals = F.normalize(node_accum_normals, dim=1, eps=1e-6)

    # --- B. CANDIDATE SEARCH (Radius) ---
    use_torch_cluster = False
    try:
        from torch_cluster import radius_graph
        use_torch_cluster = True
    except ImportError:
        pass

    if use_torch_cluster:
        if device.type == 'cuda':
            # ---- NVIDIA GPU FAST PATH ----
            # The data never leaves the VRAM. No PCIe latency.
            edge_index = radius_graph(pos, r=radius, batch=batch, loop=False)
        else:
            # ---- APPLE SILICON (MPS) FALLBACK ----
            pos_cpu = pos.cpu()
            batch_cpu = batch.cpu() if batch is not None else None
            edge_index = radius_graph(pos_cpu, r=radius, batch=batch_cpu, loop=False).to(device)
        
        # Enforce upper triangle temporarily to save computation time below
        row, col = edge_index
        mask_triu = row < col
        row, col = row[mask_triu], col[mask_triu]
    else:
        print('Please get torch-cluster you bimbo!')

    # --- C. FILTERING ---
    if row.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    # 1. BLOCKLIST (Remove structural bonds)
    cand_keys = (row.to(torch.int64) << 32) | col.to(torch.int64)
    m_u, m_v = mesh_edge_index
    m_min, m_max = torch.min(m_u, m_v), torch.max(m_u, m_v)
    mesh_keys = (m_min.to(torch.int64) << 32) | m_max.to(torch.int64)
    
    mask_new = ~torch.isin(cand_keys, mesh_keys)
    row = row[mask_new]
    col = col[mask_new]

    # 2. DYNAMIC NORMALS
    if row.numel() > 0:
        # Filter Internal Nodes
        SURFACE_THRESH = 1e-4
        mag_i = node_mag[row]
        mag_j = node_mag[col]
        mask_surface = (mag_i > SURFACE_THRESH) & (mag_j > SURFACE_THRESH)
        
        row = row[mask_surface]
        col = col[mask_surface]

        if row.numel() > 0:
            vec_ij = pos[col] - pos[row]
            vec_ij_norm = F.normalize(vec_ij, dim=1, eps=1e-8)

            n_i = node_normals[row]
            n_j = node_normals[col]

            # "Facing" Check (The Cone)
            dot_i = (n_i * vec_ij_norm).sum(dim=1)
            dot_j = (n_j * -vec_ij_norm).sum(dim=1)
            mask_cone = (dot_i > tolerance) & (dot_j > tolerance)

            # "Parallel" Check (Surfaces must be opposing)
            normal_alignment = (n_i * n_j).sum(dim=1)
            mask_not_parallel = normal_alignment < -0.5 

            mask_final = mask_cone & mask_not_parallel
            
            row = row[mask_final]
            col = col[mask_final]

    # --- D. SYMMETRIZE EDGES ---
    edges = torch.stack([row, col], dim=0)
    if edges.numel() > 0:
        edges_flip = torch.stack([col, row], dim=0)
        edges = torch.cat([edges, edges_flip], dim=1)
        
    return edges