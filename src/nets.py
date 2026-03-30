import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

# =============================================================================
# NORMALIZER COMPONENTS (Composable)
# =============================================================================

class ZScoreNormalizer(nn.Module):
    """
    Standalone running z-score. Can be chained after any transform.
    """
    def __init__(self, size, std_epsilon=1e-8):
        super().__init__()
        self.register_buffer('acc_count', torch.tensor(0, dtype=torch.float))
        self.register_buffer('acc_sum', torch.zeros(size, dtype=torch.float))
        self.register_buffer('acc_sum_squared', torch.zeros(size, dtype=torch.float))
        self.std_epsilon = std_epsilon

    def forward(self, x, accumulate=True, mask=None):
        if accumulate and self.training:
            if mask is not None:
                active = ~mask.squeeze()
                if active.sum() > 0:
                    self._accumulate(x[active])
            else:
                self._accumulate(x)
        return (x - self._mean()) / self._std()

    def inverse(self, x_norm):
        return x_norm * self._std() + self._mean()

    def _accumulate(self, x):
        with torch.no_grad():
            x = x.detach()
            self.acc_count += x.shape[0]
            self.acc_sum += x.sum(dim=0)
            self.acc_sum_squared += (x ** 2).sum(dim=0)

    def _mean(self):
        safe = torch.max(self.acc_count, torch.tensor(1.0, device=self.acc_count.device))
        return self.acc_sum / safe

    def _std(self):
        safe = torch.max(self.acc_count, torch.tensor(1.0, device=self.acc_count.device))
        var = self.acc_sum_squared / safe - self._mean() ** 2
        std = torch.sqrt(torch.clamp(var, min=0.0))
        return torch.max(std, torch.tensor(self.std_epsilon, device=self.acc_count.device))


class SymlogTransform(nn.Module):
    """Pure symlog: sign(x) * log1p(|x|). No statistics."""
    def forward(self, x):
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def inverse(self, y):
        return torch.sign(y) * torch.expm1(torch.abs(y))


class FixedScaleTransform(nn.Module):
    """Pure division by a constant. No statistics."""
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x / self.scale

    def inverse(self, y):
        return y * self.scale


class MagnitudeSymlogTransform(nn.Module):
    """
    Symlog applied to vector MAGNITUDE, preserving direction.
    Rotationally equivariant — commutes with rotation augmentation.
    Input: [N, D] where D is the vector dimension (2 or 3).
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mag = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + self.eps**2)
        direction = x / mag
        compressed_mag = torch.log1p(mag)
        return compressed_mag * direction

    def inverse(self, y):
        norm_mag = y.norm(dim=1, keepdim=True).clamp(min=self.eps)
        direction = y / norm_mag
        raw_mag = torch.expm1(norm_mag)
        return raw_mag * direction


class MagnitudeZScoreNormalizer(nn.Module):
    """
    Z-score on vector magnitude, preserving direction.
    Rotationally equivariant — magnitude is rotation-invariant,
    so mean/std are scalars applied equally to both axes.
    """
    def __init__(self, std_epsilon=1e-8, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.std_epsilon = std_epsilon
        self.register_buffer('acc_count', torch.tensor(0, dtype=torch.float))
        self.register_buffer('acc_sum', torch.tensor(0, dtype=torch.float))
        self.register_buffer('acc_sum_squared', torch.tensor(0, dtype=torch.float))

    def forward(self, x, accumulate=True, mask=None):
        mag = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + self.eps**2)
        direction = x / mag

        if accumulate and self.training:
            if mask is not None:
                active = ~mask.squeeze()
                if active.sum() > 0:
                    self._accumulate(mag[active])
            else:
                self._accumulate(mag)

        mag_normed = (mag - self._mean()) / self._std()
        return mag_normed * direction

    def inverse(self, y):
        norm_mag = y.norm(dim=1, keepdim=True)
        direction = y / norm_mag.clamp(min=self.eps)
        raw_mag = norm_mag * self._std() + self._mean()
        return raw_mag * direction

    def _accumulate(self, mag):
        with torch.no_grad():
            mag = mag.detach()
            self.acc_count += mag.shape[0]
            self.acc_sum += mag.sum()
            self.acc_sum_squared += (mag ** 2).sum()

    def _mean(self):
        safe = torch.max(self.acc_count, torch.tensor(1.0, device=self.acc_count.device))
        return self.acc_sum / safe

    def _std(self):
        safe = torch.max(self.acc_count, torch.tensor(1.0, device=self.acc_count.device))
        var = self.acc_sum_squared / safe - self._mean() ** 2
        std = torch.sqrt(torch.clamp(var, min=0.0))
        return torch.max(std, torch.tensor(self.std_epsilon, device=self.acc_count.device))


# =============================================================================
# COMPOSABLE NORMALIZER
# =============================================================================

class Normalizer(nn.Module):
    """
    Composable normalizer. Chains transforms in order, optionally adds z-score
    and/or max-scaling.
    
    Usage:
        # Magnitude-preserving symlog + equivariant z-score on vectors
        Normalizer(transforms=[MagnitudeSymlogTransform()], magnitude_zscore=True)
        
        # Per-component z-score for scalars
        Normalizer(size=4, transforms=[SymlogTransform()], zscore=True)
    """
    def __init__(self, size=None, transforms=None, zscore=False, magnitude_zscore=False, 
                 scale_by_max=False, std_epsilon=1e-8):
        super().__init__()
        self.transforms = nn.ModuleList(transforms or [])
        
        self.zscore = None
        if zscore:
            assert size is not None, "size required when zscore=True"
            self.zscore = ZScoreNormalizer(size, std_epsilon)
        
        self.mag_zscore = None
        if magnitude_zscore:
            self.mag_zscore = MagnitudeZScoreNormalizer(std_epsilon)
        
        self.scale_by_max = scale_by_max
        if scale_by_max:
            self.register_buffer('running_max', torch.tensor(1.0, dtype=torch.float))

    def forward(self, x, accumulate=True, mask=None):
        for t in self.transforms:
            x = t(x)
        
        if self.scale_by_max:
            if accumulate and self.training:
                with torch.no_grad():
                    if mask is not None:
                        active = ~mask.squeeze()
                        if active.sum() > 0:
                            batch_max = x[active].abs().max()
                        else:
                            batch_max = self.running_max
                    else:
                        batch_max = x.abs().max()
                    self.running_max = torch.max(self.running_max, batch_max)
            x = x / self.running_max
        
        if self.mag_zscore is not None:
            x = self.mag_zscore(x, accumulate=accumulate, mask=mask)
        
        if self.zscore is not None:
            x = self.zscore(x, accumulate=accumulate, mask=mask)
        return x

    def inverse(self, y):
        if self.zscore is not None:
            y = self.zscore.inverse(y)
        if self.mag_zscore is not None:
            y = self.mag_zscore.inverse(y)
        if self.scale_by_max:
            y = y * self.running_max
        for t in reversed(self.transforms):
            y = t.inverse(y)
        return y            


# =============================================================================
# MLP
# =============================================================================

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, layernorm=True):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        if layernorm: layers.insert(1, nn.LayerNorm(hidden_size))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if layernorm: layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, output_size))
        if layernorm: layers.append(nn.LayerNorm(output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class GatedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, layernorm=True):
        super().__init__()
        
        # 1. The Update Branch
        self.update_net = MLP(input_size, hidden_size, output_size, num_layers, layernorm)
        
        # 2. The Gate Branch
        self.gate_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

        nn.init.xavier_uniform_(self.gate_net[2].weight, gain=0.01)
        nn.init.constant_(self.gate_net[2].bias, -2.0) 

    def forward(self, x, old_state):
        
        # Calculate the proposed new edge features
        candidate = self.update_net(x)
        
        # Calculate the mixing gate
        gate = self.gate_net(x)
        
        # Mix the candidate and residual. This prevents over-smoothing by allowing the network 
        new_state = old_state * (1.0 - gate) + candidate * gate
        
        return new_state

class InteractionBlock(MessagePassing):
    def __init__(self, hidden_size):
        super().__init__(aggr='add') 
        
        # 1. GATED EDGE MODELS (Combats edge over-smoothing)
        self.mesh_edge_mlp = GatedMLP(hidden_size * 3, hidden_size, hidden_size, 3)
        self.world_edge_mlp = GatedMLP(hidden_size * 3, hidden_size, hidden_size, 3)
        self.node_mlp = GatedMLP(hidden_size * 3, hidden_size, hidden_size, 3)

    def forward(self, x, mesh_edge_index, mesh_edge_attr, world_edge_index, world_edge_attr, dt):

        # --- 1. Update Mesh Edges (GATED) ---
        src_m, dst_m = mesh_edge_index
        mesh_inputs = torch.cat([x[src_m], x[dst_m], mesh_edge_attr], dim=1)
        mesh_edge_updated = self.mesh_edge_mlp(mesh_inputs, mesh_edge_attr)

        # --- 2. Update World Edges (GATED) ---
        if world_edge_index.size(1) > 0:
            src_w, dst_w = world_edge_index
            world_inputs = torch.cat([x[src_w], x[dst_w], world_edge_attr], dim=1)
            world_edge_updated = self.world_edge_mlp(world_inputs, world_edge_attr)
        else:
            world_edge_updated = torch.zeros((0, x.shape[1]), device=x.device, dtype=x.dtype)

        # --- 3. Aggregation (SUM ONLY) ---
        mesh_aggr_sum = torch.zeros_like(x, dtype=x.dtype)
        world_aggr_sum = torch.zeros_like(x, dtype=x.dtype)
        
        mesh_aggr_sum.index_add_(0, dst_m, mesh_edge_updated)
        if world_edge_index.size(1) > 0:
            world_aggr_sum.index_add_(0, dst_w, world_edge_updated)

        # --- 4. STANDARD NODE UPDATE (MLP + Residual) ---
        node_inputs = torch.cat([x, mesh_aggr_sum, world_aggr_sum], dim=1)
        
        x_updated = self.node_mlp(node_inputs, x)

        return x_updated, mesh_edge_updated, world_edge_updated