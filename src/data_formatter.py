import os.path as osp
import glob
import torch
import numpy as np
from torch_geometric.data import Dataset, Data

class MeshData(Data):
    def __inc__(self, key, value, *args, **kwargs):

        if key == 'face_index':
            return self.num_nodes

        if key == 'world_edge_index':
            return self.num_nodes
        
        if key == 'elements':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

class DataFormatter(Dataset):
    def __init__(self, root_dir, augment=True):
        super().__init__()
        self.root_dir = root_dir
        self.augment = augment
        
        search_path = osp.join(root_dir, "*.pt")
        self.sample_files = sorted(glob.glob(search_path))
        # Filter out temporary OS files(for macos)
        self.sample_files = [f for f in self.sample_files if not osp.basename(f).startswith("._")]
        
        if len(self.sample_files) == 0:
            raise FileNotFoundError(f"No .pt files found in {root_dir}")

        print(f"DataFormatter: Found {len(self.sample_files)} samples in {root_dir}")
        print(f"DataFormatter: Augmentation {'ENABLED' if augment else 'DISABLED'}")

    def len(self):
        return len(self.sample_files)

    def get(self, idx):
        path = self.sample_files[idx]
        data_dict = torch.load(path)

        frame_filename = osp.basename(path)

        # 1. Direct Load
        pos = data_dict['pos']
        mesh_pos = data_dict['mesh_pos']
        velocity = data_dict['velocity']
        prev_velocity = data_dict['prev_velocity']
        target_accel = data_dict['target_accel']
        face_index = data_dict['face_index']
        elements = data_dict['elements']
        
        # A) Mesh Edges
        edge_index = data_dict['edge_index']
        edge_index_flip = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.unique(torch.cat([edge_index, edge_index_flip], dim=1), dim=1)

        # B) World Edges (Only if they exist in the file)
        world_edge_index = data_dict.get('world_edge_index', torch.empty((2,0), dtype=torch.long))
        if world_edge_index.size(1) > 0:
            world_flip = torch.stack([world_edge_index[1], world_edge_index[0]], dim=0)
            world_edge_index = torch.unique(torch.cat([world_edge_index, world_flip], dim=1), dim=1)
        # -----------------------------------------------------------
        
        # 2. Extract Flags
        node_type = data_dict['node_type'] 
        is_fixed = data_dict['is_constraint']
        
        # === AUGMENTATION (Per-Frame) ===
        if self.augment:
            pos, mesh_pos, prev_velocity, velocity, target_accel = self._apply_augmentation(
                pos, mesh_pos, prev_velocity, velocity, target_accel
            )
        
        # 3. Build One-Hot Features [Impactor, Absorber, Fixed]
        is_impactor_feat = node_type.float().unsqueeze(1)       # 1 if Steel
        is_absorber_feat = (1 - node_type).float().unsqueeze(1) # 1 if Membrane
        is_fixed_feat = is_fixed.float().unsqueeze(1)
        inv_mass = data_dict['inv_mass'].float().unsqueeze(1)

        node_attributes = torch.cat([is_impactor_feat, is_absorber_feat, is_fixed_feat], dim=1)

        # For 2D data: this results in N x 4. For 3D data: N x 6
        velocities = torch.cat([prev_velocity, velocity], dim=1)
        
        # 4. Extract Split Index K
        num_impactors = data_dict['num_impactors']

        # 5. Create Data Object
        data = MeshData(
            elements=elements,
            pos=pos,
            mesh_pos=mesh_pos,
            velocities=velocities,
            edge_index=edge_index, 
            face_index=face_index,
            world_edge_index=world_edge_index, 
            node_attr=node_attributes,
            inv_mass=inv_mass, 
            y=target_accel,
            mask=is_fixed, 
            num_impactors=num_impactors,
            frame_id=frame_filename
        )
        
        return data

    def _apply_augmentation(self, pos, mesh_pos, prev_velocity, velocity, target_accel):
        """
        Apply random continuous rotation dynamically for 2D or 3D.
        """
        original_dtype = pos.dtype
        dim = pos.shape[1]  # Detect spatial dimension (2 or 3)
        
        pos = pos.float()
        mesh_pos = mesh_pos.float()
        prev_velocity = prev_velocity.float()
        velocity = velocity.float()
        target_accel = target_accel.float()
        
        # 1. CONTINUOUS RANDOM ROTATION based on dimension
        R = self._random_rotation_matrix(dim).to(pos.device)
        
        # Apply rotation (x' = xR^T)
        pos = torch.mm(pos, R.T)
        mesh_pos = torch.mm(mesh_pos, R.T)
        prev_velocity = torch.mm(prev_velocity, R.T)
        velocity = torch.mm(velocity, R.T)
        target_accel = torch.mm(target_accel, R.T)
        
        pos = pos.to(original_dtype)
        mesh_pos = mesh_pos.to(original_dtype)
        prev_velocity = prev_velocity.to(original_dtype)
        velocity = velocity.to(original_dtype)
        target_accel = target_accel.to(original_dtype)
        
        return pos, mesh_pos, prev_velocity, velocity, target_accel
    
    def _random_rotation_matrix(self, dim):
        """
        Generate a uniformly random rotation matrix.
        - For 2D: samples a random angle and returns an SO(2) matrix.
        - For 3D: samples uniformly from SO(3) using the Shoemake method.
        """
        if dim == 2:
            # 2D Random Rotation
            theta = torch.rand(1).item() * 2 * np.pi
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            R = torch.tensor([
                [cos_t, -sin_t],
                [sin_t,  cos_t]
            ], dtype=torch.float32)
            return R
            
        elif dim == 3:
            # 3D Random Rotation (Quaternions on S³)
            u1, u2, u3 = torch.rand(3).tolist()
            q0 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
            q1 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
            q2 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
            q3 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
            
            R = torch.tensor([
                [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
                [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
                [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
            ], dtype=torch.float32)
            return R
            
        else:
            raise ValueError(f"Unsupported dimension for augmentation: {dim}")