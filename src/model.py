import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from src.nets import MLP, InteractionBlock, Normalizer, SymlogTransform, FixedScaleTransform, MagnitudeSymlogTransform
from tools.utils import build_world_edges


class MeshGraphNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        node_in_dim = config['model']['node_in_dim']
        mesh_edge_in_dim = config['model']['mesh_edge_in_dim'] 
        world_edge_in_dim = config['model']['world_edge_in_dim'] 
        hidden_dim = config['model']['hidden_dim']
        self.hidden_dim = hidden_dim
        out_dim = config['model']['output_dim']
        self.num_layers = config['model']['num_layers']
        self.mode = config['mode']
        self.dt = float(config['train']['time_step'])

        self.radius = float(config['model']['radius'])

        # --- DYNAMIC SPATIAL DIMENSION ---
        # Infers 2D or 3D based on output_dim
        self.spatial_dim = config['model'].get('spatial_dim', out_dim)

        # Update normalizers to use dynamic dimensions
        self.node_normalizer = Normalizer(size=4, zscore=True)

        # Inv mass: symlog + z-score (spans orders of magnitude)
        self.mass_normalizer = Normalizer(size=1, transforms=[SymlogTransform()], zscore=True)
        
        self.mesh_edge_normalizer = Normalizer(size=8, zscore=True)
        self.world_edge_normalizer =  Normalizer(size=4, zscore=True)

        self.absorber_target_normalizer = Normalizer(size=2, zscore=True)
        #self.impactor_target_normalizer for dynamic simulations 

        self.node_encoder = MLP(node_in_dim, hidden_dim, hidden_dim, num_layers=3)
        self.mesh_edge_encoder = MLP(mesh_edge_in_dim, hidden_dim, hidden_dim, num_layers=3)
        self.world_edge_encoder = MLP(world_edge_in_dim, hidden_dim, hidden_dim, num_layers=3)

        self.layers = nn.ModuleList(
            [InteractionBlock(hidden_dim) for _ in range(self.num_layers)]
        )

        self.decoder = MLP(hidden_dim, hidden_dim, out_dim, num_layers=3, layernorm=False)
        last_layer = self.decoder.net[-1] 
        last_layer.bias.data.fill_(0.0)

    
    # ------------------------------
    # Forward pass
    # ------------------------------
    def forward(self, data, accumulate_stats=True):

        pos = data.pos
        static_pos = data.mesh_pos
        vel = data.velocities
        inv_mass = data.inv_mass
        
        curr_vel = vel[:, self.spatial_dim:] 
        
        mesh_edge_index = data.edge_index
        face_index = data.face_index
        batch = data.batch
        type_flags = data.node_attr
        device = pos.device

        # ------------------------------
        # A) Build world edges
        # ------------------------------
        if self.mode == 'rollout':
            world_edges = build_world_edges(
                pos.detach(),
                batch.detach(),
                face_index,
                mesh_edge_index,
                self.radius
            )
        else:
            world_edges = data.world_edge_index

        world_edge_index = world_edges.to(device)

        # ------------------------------
        # B) Node encoding
        # ------------------------------
        norm_vel = self.node_normalizer(vel, accumulate=accumulate_stats, mask=data.mask)
        inv_mass = self.mass_normalizer(inv_mass, accumulate=accumulate_stats, mask=data.mask)
        x = torch.cat([norm_vel, inv_mass, type_flags], dim=1)
        x = self.node_encoder(x)

        # ------------------------------
        # C) Mesh Edge encoding
        # ------------------------------
        src, dst = mesh_edge_index
        rel = pos[src] - pos[dst]
        dist = rel.norm(dim=1, keepdim=True)
        rel_s = static_pos[src] - static_pos[dst]
        dist_s = rel_s.norm(dim=1, keepdim=True)
        strain = (dist - dist_s) / (dist_s + 1e-6)
        strain_vec = strain * (rel / dist + 1e-6)
        rel_vel = curr_vel[src] - curr_vel[dst]

        mesh_raw = torch.cat([rel, strain_vec, rel_vel, dist, dist_s], dim=1)

        mesh_norm = self.mesh_edge_normalizer(mesh_raw, accumulate=accumulate_stats)
        
        # 2. World Attributes
        if world_edge_index.size(1) > 0:
            src_w, dst_w = world_edge_index
            rel_w = pos[src_w] - pos[dst_w]
            rel_vel_w = curr_vel[src_w] - curr_vel[dst_w]
            
            world_raw = torch.cat([rel_w, rel_vel_w], dim=1)
            
            world_norm = self.world_edge_normalizer(world_raw, accumulate=accumulate_stats)
        else:

            world_norm = torch.empty((0, self.spatial_dim * 2), device=device)

        mesh_edge_attr = self.mesh_edge_encoder(mesh_norm)
        world_edge_attr = self.world_edge_encoder(world_norm)

        # ------------------------------
        # E) Message passing
        # ------------------------------
        dt = self.dt 

        for layer in self.layers:
            x, mesh_edge_attr, world_edge_attr = checkpoint(
                layer, 
                x, 
                mesh_edge_index, 
                mesh_edge_attr, 
                world_edge_index, 
                world_edge_attr,
                dt,
                use_reentrant=False 
            )

        out = self.decoder(x)
        

        if self.mode == 'rollout':
            return out, world_edge_index
        else:
            return out

    # =========================================================
    # MATERIAL CONSTANTS (Ogden)
    # =========================================================
    OGDEN_MU = [11.5138e6, -0.33239e6]
    OGDEN_ALPHA = [1.457, -7.8825]

    # =========================================================
    # LOSS FUNCTION FOR QUASISTATIC SIMULATIONS.    
    # =========================================================
    def loss(self, pred, target, batch=None, mask=None, node_loss_mask=None, 
             accumulate_stats=True):
        is_steel = batch.node_attr[:, 0].bool()
        is_tpu = ~is_steel 
        is_fixed = mask.squeeze().bool()
        
        target_norm = torch.zeros_like(target)

        tpu_mask = mask[is_tpu] if mask is not None else None

        if is_tpu.sum() > 0:
            target_norm[is_tpu] = self.absorber_target_normalizer(
                target[is_tpu], accumulate=accumulate_stats, mask=tpu_mask
            )
        
        target_norm[is_steel] = target[is_steel]

        # ---------------------------------------------------------
        # Zero out prescribed nodes for edge losses
        # ---------------------------------------------------------
        prescribed = is_fixed | is_steel
    
        pred_clean = pred.clone()
        target_clean = target_norm.clone()
        pred_clean[prescribed] = 0.0
        target_clean[prescribed] = 0.0

        # ---------------------------------------------------------
        # A. Node Loss — MAE (per-node)
        # ---------------------------------------------------------
        node_loss = torch.abs(pred - target_norm)

        # ---------------------------------------------------------
        # B. Structural Loss — MAE on relative predictions
        # ---------------------------------------------------------
        mesh_loss_sum = torch.zeros_like(node_loss)
        world_loss_sum = torch.zeros_like(node_loss)

        if batch is not None:
            def get_rel_error(edges):
                if edges.numel() == 0: return None
                row, col = edges
                pred_diff = pred_clean[row] - pred_clean[col]
                target_diff = target_clean[row] - target_clean[col]
                return torch.abs(pred_diff - target_diff)

            mesh_err = get_rel_error(batch.edge_index)
            if mesh_err is not None:
                row, col = batch.edge_index
                mesh_loss_sum = mesh_loss_sum.index_add(0, row, 0.5 * mesh_err)
                mesh_loss_sum = mesh_loss_sum.index_add(0, col, 0.5 * mesh_err)

            if hasattr(batch, 'world_edge_index') and batch.world_edge_index.numel() > 0:
                world_err = get_rel_error(batch.world_edge_index)
                if world_err is not None:
                    row, col = batch.world_edge_index
                    world_loss_sum = world_loss_sum.index_add(0, row, 0.5 * world_err)
                    world_loss_sum = world_loss_sum.index_add(0, col, 0.5 * world_err)


        # ---------------------------------------------------------
        # D. Combine all terms
        # ---------------------------------------------------------
        total_error = (node_loss 
                       + mesh_loss_sum * 0.5 
                       + world_loss_sum * 0.5)
        
        final_loss = total_error
        
        is_impactor = batch.node_attr[:, 0].bool()
        exclude_mask = is_fixed | is_impactor

        # ---------------------------------------------------------
        # E. Apply node loss mask
        # ---------------------------------------------------------
        if node_loss_mask is not None:
            if mask is not None:
                combined_mask = node_loss_mask & (~exclude_mask)
            else:
                combined_mask = node_loss_mask
            
            if combined_mask.sum() > 0:
                return final_loss[combined_mask].mean()
            else:
                return final_loss.sum() * 0.0
        else:
            if mask is not None:
                return final_loss[~exclude_mask].mean()
            else:
                return final_loss.mean()