import torch
import torch.nn as nn
import numpy as np
import os
import time
import gc
import multiprocessing as mp
from torch_geometric.data import Batch
import yaml
import argparse
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.model import MeshGraphNet
from src.data_formatter import MeshData as Data
from tools.utils import build_world_edges
from tools.rl_environment import generate_rl_environment, BOX_SIZE

# =============================================================================
# 0. VIDEO GENERATION
# =============================================================================
def save_episode_video(seeds_sequence, iteration, save_dir, status="unknown", box_size=BOX_SIZE):
    """Renders a sequence of seed arrays into a .gif showing the Voronoi structure."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def update(frame):
        ax.clear()
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        ax.set_title(f"Iter {iteration} | Step {frame} | {status.upper()}")
        ax.set_aspect('equal')
        
        seeds = seeds_sequence[frame]
        
        ax.scatter(seeds[:, 0], seeds[:, 1], c='red', s=50, zorder=5, label='Seeds')
        
        dummies = np.array([[-500, -500], [box_size+500, -500], 
                            [box_size+500, box_size+500], [-500, box_size+500]])
        all_pts = np.vstack([seeds, dummies])
        try:
            vor = Voronoi(all_pts)
            for ridge in vor.ridge_vertices:
                if -1 not in ridge:
                    pt1 = vor.vertices[ridge[0]]
                    pt2 = vor.vertices[ridge[1]]
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'k-', lw=2)
        except Exception:
            pass

    ani = FuncAnimation(fig, update, frames=len(seeds_sequence))
    save_path = os.path.join(save_dir, f"episode_iter_{iteration}_{status}_{int(time.time())}.gif")
    ani.save(save_path, writer=PillowWriter(fps=1/0.3))
    plt.close(fig)

# =============================================================================
# 0.5 GEOMETRIC TOPOLOGY EXTRACTION (The "Eyes" for Actor & Critic)
# =============================================================================
def extract_topology_geometric(seeds_np, num_seeds, box_size=BOX_SIZE):
    """
    Per seed-pair: [abs(dx), abs(dy), edge_length] if neighbors, [0,0,0] if not.
    Normalized by box_size. Symmetric to preserve MLP permutation invariance.
    """
    feat = np.zeros((num_seeds, num_seeds, 3), dtype=np.float32)
    try:
        dummies = np.array([[-500, -500], [box_size+500, -500], 
                            [box_size+500, box_size+500], [-500, box_size+500]])
        all_pts = np.vstack([seeds_np, dummies])
        vor = Voronoi(all_pts)
        
        for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
            if p1 < num_seeds and p2 < num_seeds:
                v1, v2 = vor.ridge_vertices[ridge_idx]
                if v1 >= 0 and v2 >= 0:
                    edge_len = np.linalg.norm(vor.vertices[v1] - vor.vertices[v2])
                    
                    dx = abs(seeds_np[p2, 0] - seeds_np[p1, 0])
                    dy = abs(seeds_np[p2, 1] - seeds_np[p1, 1])
                    
                    sym_feat = [dx / box_size, dy / box_size, edge_len / box_size]
                    
                    feat[p1, p2] = sym_feat
                    feat[p2, p1] = sym_feat
    except Exception:
        pass
    
    return feat.flatten()

# =============================================================================
# 1. MESH WORKER (Stateless — runs in persistent subprocess)
# =============================================================================
def _mesh_worker(seeds):
    data_dict = generate_rl_environment(seeds)
    if data_dict is None:
        return None

    return {
        'pos': data_dict['pos'].numpy(),
        'mesh_pos': data_dict['mesh_pos'].numpy(),
        'inv_mass': data_dict['inv_mass'].numpy(),
        'velocities_prev': data_dict['prev_velocity'].numpy(),
        'velocities_curr': data_dict['velocity'].numpy(),
        'node_type': data_dict['node_type'].numpy(),
        'is_constraint': data_dict['is_constraint'].numpy(),
        'num_impactors': data_dict['num_impactors'].numpy(),
        'face_index': data_dict['face_index'].numpy(),
        'edge_index': data_dict['edge_index'].numpy(),
        'world_edge_index': data_dict['world_edge_index'].numpy(),
    }

def _dicts_to_pyg_batch(raw_dicts, device):
    data_list = []
    for d in raw_dicts:
        nt = torch.from_numpy(d['node_type'])
        ic = torch.from_numpy(d['is_constraint'])
        node_attr = torch.stack([
            nt.float(),
            (1 - nt).float(),
            ic.float(),
        ], dim=1)

        vel = np.concatenate([d['velocities_prev'], d['velocities_curr']], axis=1)

        data_list.append(Data(
            pos=torch.from_numpy(d['pos']).float(),
            mesh_pos=torch.from_numpy(d['mesh_pos']).float(),
            inv_mass=torch.from_numpy(d['inv_mass']).float().unsqueeze(1),
            velocities=torch.from_numpy(vel).float(),
            node_attr=node_attr,
            mask=ic,
            num_impactors=torch.from_numpy(d['num_impactors']),
            face_index=torch.from_numpy(d['face_index']).long(),
            edge_index=torch.from_numpy(d['edge_index']).long(),
            world_edge_index=torch.from_numpy(d['world_edge_index']).long(),
        ))

    return Batch.from_data_list(data_list).to(device)

# =============================================================================
# 2. CRUSH SIMULATOR
# =============================================================================
class CrushSimulator:
    def __init__(self, config, device):
        self.device = device
        self.num_steps = config['rollout']['num_steps']
        self.dt = config['rollout']['time_step']
        self.impactor_vel = config['rollout']['impactor_vel']
        self.edge_interval = config['rollout']['world_edge_interval']
        self.num_workers = config['mesh']['num_workers']

        self.mu = torch.tensor(config['material']['mu'], dtype=torch.float32, device=device)
        self.alpha = torch.tensor(config['material']['alpha'], dtype=torch.float32, device=device)
        self.rho_tpu = config['material']['rho_tpu']

        mgn_cfg = config['mgn']
        mgn_config = {
            'mode': 'rollout',
            'train': {'time_step': self.dt},
            'model': {k: mgn_cfg[k] for k in [
                'spatial_dim', 'node_in_dim', 'mesh_edge_in_dim',
                'world_edge_in_dim', 'hidden_dim', 'output_dim',
                'num_layers', 'radius',
            ]}
        }
        self.mgn = MeshGraphNet(mgn_config).to(device)

        ckpt_path = mgn_cfg['checkpoint']
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            clean = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()}
            self.mgn.load_state_dict(clean, strict=False)
            print(f"Loaded MGN from {ckpt_path}")
        else:
            print(f"MGN checkpoint not found: {ckpt_path}")

        self.mgn.eval()
        for p in self.mgn.parameters():
            p.requires_grad_(False)

        self.mgn = torch.compile(self.mgn, mode='default', dynamic=True)

        self.radius = mgn_cfg['radius']
        self._pool = mp.Pool(processes=self.num_workers, maxtasksperchild=20)
        self._num_energy_samples = len(range(0, self.num_steps, self.edge_interval)) + 1

    def shutdown(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    @torch.no_grad()
    def evaluate_seeds(self, seed_batch):
        raw_results = self._pool.map(_mesh_worker, seed_batch)

        valid_dicts = []
        valid_idx = []
        for i, d in enumerate(raw_results):
            if d is not None:
                valid_dicts.append(d)
                valid_idx.append(i)

        if not valid_dicts:
            return torch.zeros(0, device=self.device), []

        num_envs = len(valid_dicts)
        graph = _dicts_to_pyg_batch(valid_dicts, self.device)

        is_steel = graph.node_attr[:, 0].bool()
        is_tpu = ~is_steel
        is_fixed = graph.mask.squeeze().bool()
        free_mask = is_tpu & ~is_fixed

        energy_ctx = self._precompute_energy(graph, num_envs)

        N = graph.pos.size(0)
        curr_pos = graph.pos.clone()
        curr_vel = graph.velocities[:, 2:].clone()
        curr_prev_vel = graph.velocities[:, :2].clone()
        delta_v = torch.zeros(N, 2, device=self.device)
        vel_buf = torch.zeros(N, 4, device=self.device)

        T = self._num_energy_samples
        energy_buf = torch.zeros(T, num_envs, device=self.device)
        disp_buf = torch.zeros(T, num_envs, device=self.device)
        sample_idx = 0
        
        env_max_strains = torch.zeros(num_envs, device=self.device)

        world_edges = None
        for step in range(self.num_steps):
            vel_buf[:, :2] = curr_prev_vel
            vel_buf[:, 2:] = curr_vel
            graph.pos = curr_pos
            graph.velocities = vel_buf

            if step % self.edge_interval == 0:
                world_edges = build_world_edges(
                    curr_pos, graph.batch, graph.face_index,
                    graph.edge_index, self.radius
                )
            graph.world_edge_index = world_edges

            outputs = self.mgn(graph, accumulate_stats=False)
            pred = outputs[0] if isinstance(outputs, tuple) else outputs

            delta_v.zero_()
            if free_mask.sum() > 0:
                delta_v[free_mask] = self.mgn.absorber_target_normalizer.inverse(pred[free_mask])

            next_vel = curr_vel + delta_v
            next_vel[is_steel, 0] = 0.0
            next_vel[is_steel, 1] = self.impactor_vel
            next_vel[is_fixed] = 0.0

            curr_prev_vel = curr_vel
            curr_vel = next_vel
            curr_pos = curr_pos + curr_vel * self.dt

            if step % self.edge_interval == 0 or step == self.num_steps - 1:
                energy_buf[sample_idx] = self._strain_energy(curr_pos, energy_ctx)
                disp_buf[sample_idx] = self._impactor_displacement(curr_pos, graph.batch, is_steel, energy_ctx)
                
                L_curr = (curr_pos[energy_ctx['src']] - curr_pos[energy_ctx['dst']]).norm(dim=1)
                strain = torch.abs(L_curr - energy_ctx['L0']) / energy_ctx['L0']
                
                for i in range(num_envs):
                    mask = energy_ctx['edge_batch'] == i
                    if mask.any():
                        env_max_strains[i] = torch.max(env_max_strains[i], strain[mask].max())
                        
                sample_idx += 1

        cfe = self._compute_cfe_gpu(energy_buf[:sample_idx], disp_buf[:sample_idx])

        physics_valid_idx = []
        physics_cfes = []
        
        for rank, orig_idx in enumerate(valid_idx):
            if env_max_strains[rank] <= 1.0:
                physics_valid_idx.append(orig_idx)
                physics_cfes.append(cfe[rank])

        final_cfe_tensor = torch.tensor(physics_cfes, device=self.device) if physics_cfes else torch.zeros(0, device=self.device)

        del graph, energy_buf, disp_buf, energy_ctx, env_max_strains
        del curr_pos, curr_vel, curr_prev_vel, delta_v, vel_buf
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        return final_cfe_tensor, physics_valid_idx
    
    def _precompute_energy(self, graph, num_envs):
        ei = graph.edge_index
        is_tpu = graph.node_attr[:, 1].bool()
        is_steel = graph.node_attr[:, 0].bool()
        batch = graph.batch
        ref = graph.mesh_pos
        inv_mass = graph.inv_mass.squeeze()

        mask = is_tpu[ei[0]] & is_tpu[ei[1]]
        src, dst = ei[:, mask]
        uniq = src < dst
        src, dst = src[uniq], dst[uniq]

        L0 = (ref[src] - ref[dst]).norm(dim=1).clamp(min=1e-9)
        edge_batch = batch[src]

        tpu_mass = torch.zeros_like(inv_mass)
        tpu_active = is_tpu & (inv_mass > 1e-6) 
        tpu_mass[tpu_active] = 1.0 / inv_mass[tpu_active]

        env_volume = torch.zeros(num_envs, device=ref.device)
        for i in range(num_envs):
            env_mass = tpu_mass[is_tpu & (batch == i)].sum()
            env_volume[i] = env_mass / self.rho_tpu

        L0_sum_per_env = torch.zeros(num_envs, device=ref.device)
        L0_sum_per_env.index_add_(0, edge_batch, L0)

        V_per_edge = env_volume[edge_batch] * (L0 / L0_sum_per_env[edge_batch].clamp(min=1e-9))

        ref_steel_y = torch.zeros(num_envs, device=ref.device)
        steel_counts = torch.zeros(num_envs, device=ref.device)
        ref_steel_y.index_add_(0, batch[is_steel], ref[is_steel, 1])
        steel_counts.index_add_(0, batch[is_steel], torch.ones(is_steel.sum(), device=ref.device))
        ref_steel_y = ref_steel_y / steel_counts.clamp(min=1)

        return {
            'src': src, 'dst': dst, 'L0': L0,
            'edge_batch': edge_batch, 'V_per_edge': V_per_edge,
            'ref_steel_y': ref_steel_y,
            'num_envs': num_envs,
        }
    
    def _strain_energy(self, pos, ctx):
        src, dst, L0 = ctx['src'], ctx['dst'], ctx['L0']
        L = (pos[src] - pos[dst]).norm(dim=1).clamp(min=1e-9)
        lam = L / L0

        lam_e = lam.unsqueeze(0)
        a = self.alpha.unsqueeze(1)
        m = self.mu.unsqueeze(1)
        W = ((m / a) * (lam_e ** a + lam_e ** (-a) - 2.0)).sum(dim=0)

        n = ctx['num_envs']
        U = torch.zeros(n, device=pos.device)
        U.index_add_(0, ctx['edge_batch'], W * ctx['V_per_edge'])
        return U

    def _impactor_displacement(self, pos, batch, is_steel, ctx):
        num_envs = ctx['num_envs']
        s = torch.zeros(num_envs, device=pos.device)
        c = torch.zeros(num_envs, device=pos.device)
        sy = pos[is_steel, 1]
        sb = batch[is_steel]
        s.index_add_(0, sb, sy)
        c.index_add_(0, sb, torch.ones_like(sy))
        curr_mean_y = s / c.clamp(min=1)

        return curr_mean_y - ctx['ref_steel_y']

    def _compute_cfe_gpu(self, energy, disp):
        dU = energy[1:] - energy[:-1]
        dy = disp[1:] - disp[:-1]

        safe_dy = dy.clone()
        safe_dy[dy.abs() < 1e-9] = 1e-9
        forces = (dU / safe_dy).abs()

        skip = max(1, forces.size(0) // 10)
        steady = forces[skip:]
        if steady.size(0) == 0:
            return torch.zeros(energy.size(1), device=energy.device)

        peak = steady.max(dim=0).values
        mean = steady.mean(dim=0)
        return torch.where(peak > 1e-6, mean / peak, torch.zeros_like(peak))

# =============================================================================
# 3. PPO NETWORKS
# =============================================================================
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128, num_layers=2, max_action=1.0):
        super().__init__()
        self.max_action = max_action
        
        layers = [nn.Linear(obs_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        
        # Safe exploration noise initialization (~0.3 mm)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -1.2)
        
        nn.init.uniform_(self.mean_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs):
        h = self.backbone(obs)
        mean = self.mean_head(h)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    @staticmethod
    def _stable_log_squash(u):
        """
        Compute log(1 - tanh²(u)) without ever calling tanh.
        
        Uses the identity: 1 - tanh²(u) = sech²(u) = 1/cosh²(u)
          log(1 - tanh²(u)) = -2·log(cosh(u))
                             = 2·log(2) - 2|u| - 2·softplus(-2|u|)
        
        This is numerically stable for all u. The naive implementation
        tanh(u) → square → subtract from 1 → log saturates to a flat
        plateau with zero gradient for |u| > ~4 in float32.
        """
        return 2 * np.log(2) - 2 * u.abs() - 2 * torch.nn.functional.softplus(-2 * u.abs())

    def get_action(self, obs, deterministic=False):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)

        if deterministic:
            u = mean
        else:
            u = dist.sample()

        # 1. Squash the action to guarantee absolute bounds
        action = torch.tanh(u) * self.max_action

        # 2. Log-probability with Jacobian correction
        squash_correction = self._stable_log_squash(u)
        log_prob = dist.log_prob(u) - squash_correction - np.log(self.max_action)
        log_prob = log_prob.sum(dim=-1)

        # 3. Action-space entropy: H[a] = H[N(μ,σ²)] + E[Σ log(1 - tanh²(u))] + D·log(a_max)
        #    The Jacobian cancels in the PPO ratio, but the entropy bonus must
        #    reflect the true action-space distribution to prevent premature collapse.
        gaussian_entropy = dist.entropy()  # [B, D]
        entropy = (gaussian_entropy + squash_correction).sum(dim=-1) + np.log(self.max_action) * u.shape[-1]
        
        return action, log_prob, entropy

    def evaluate_action(self, obs, action):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)

        # 1. Recover the raw Gaussian sample (u) by reversing the Tanh.
        #    This u is DETACHED — the stored action has no connection to θ.
        epsilon = 1e-6
        action_normalized = action / self.max_action
        action_normalized = torch.clamp(action_normalized, -1.0 + epsilon, 1.0 - epsilon)
        u = torch.atanh(action_normalized)

        # 2. Log-probability with Jacobian correction.
        #    The correction cancels in the PPO ratio anyway, but we include it
        #    for numerical consistency with get_action.
        squash_correction = self._stable_log_squash(u)
        log_prob = dist.log_prob(u) - squash_correction - np.log(self.max_action)
        log_prob = log_prob.sum(dim=-1)

        # 3. Action-space entropy.
        #    The stored action is detached (requires_grad=False), so computing
        #    the squash correction from it yields zero gradient w.r.t. θ — the
        #    optimizer would only see the Gaussian entropy gradient, making the
        #    correction cosmetic.
        #
        #    To get a gradient that actually reflects tanh compression, we draw
        #    a fresh sample from the CURRENT policy via the reparameterization
        #    trick (u = μ + σ·ε). This u depends on θ through μ and σ, so the
        #    squash correction contributes to the gradient and can push back
        #    against collapse at the boundaries.
        u_fresh = dist.rsample()
        squash_correction_fresh = self._stable_log_squash(u_fresh)
        gaussian_entropy = dist.entropy()  # [B, D] — analytical, depends on σ_θ
        entropy = (gaussian_entropy + squash_correction_fresh).sum(dim=-1) + np.log(self.max_action) * action.shape[-1]
        
        return log_prob, entropy

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)

# =============================================================================
# 4. PPO AGENT (Episodic + Geometric Critic & Actor)
# =============================================================================
class PPOAgent:
    def __init__(self, config):
        ppo = config['ppo']
        self.num_seeds = ppo['num_seeds']
        
        obs_dim = (self.num_seeds * 2) + (self.num_seeds ** 2 * 3) + 1
        act_dim = self.num_seeds * 2

        self.policy = PolicyNetwork(obs_dim, act_dim,
                                    hidden_dim=ppo['hidden_dim'], 
                                    num_layers=ppo['num_hidden_layers'],
                                    max_action=ppo['max_action'])
        
        self.value = ValueNetwork(obs_dim, ppo['hidden_dim'], ppo['num_hidden_layers'])

        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=ppo['lr_policy'])
        self.opt_value = torch.optim.Adam(self.value.parameters(), lr=ppo['lr_value'])

        self.lr_policy_base = ppo['lr_policy']
        self.lr_value_base = ppo['lr_value']

        self.clip_eps = ppo['clip_epsilon']
        self.entropy_coeff = ppo['entropy_coeff']
        self.max_grad_norm = ppo['max_grad_norm']
        self.mini_batch_size = ppo['mini_batch_size']
        self.max_action = ppo['max_action']
        self.seed_margin = ppo['seed_margin']
        
        # Phasic update: critic settles first, then policy uses clean advantages
        self.value_epochs = ppo.get('value_epochs', 16)
        self.policy_epochs = ppo.get('policy_epochs', 4)

    def to(self, device):
        self.policy = self.policy.to(device)
        self.value = self.value.to(device)
        self.device = device
        return self

    def propose_seeds(self, base_seeds_np, current_cfe=None, deterministic=False):
        B = base_seeds_np.shape[0]
        
        obs_raw = torch.tensor(base_seeds_np.reshape(B, -1), dtype=torch.float32, device=self.device)
        obs_coords = (obs_raw - BOX_SIZE / 2.0) / (BOX_SIZE / 2.0)
        
        topologies = [extract_topology_geometric(base_seeds_np[i], self.num_seeds, BOX_SIZE) for i in range(B)]
        topology_tensor = torch.tensor(np.array(topologies), dtype=torch.float32, device=self.device)
        
        obs = torch.cat([obs_coords, topology_tensor], dim=1)

        if current_cfe is not None:

            cfe_feat = current_cfe.detach().clone().float().to(self.device)
            if cfe_feat.dim() == 1:
                cfe_feat = cfe_feat.unsqueeze(1)
            obs = torch.cat([obs, cfe_feat], dim=1)

        with torch.no_grad():
            actions, log_probs, _ = self.policy.get_action(obs, deterministic=deterministic)
            values = self.value(obs)

        deltas = actions.cpu().numpy().reshape(B, self.num_seeds, 2)
        proposed = np.clip(base_seeds_np + deltas,
                           self.seed_margin, BOX_SIZE - self.seed_margin)
        
        return proposed, obs, actions, log_probs, values

    @staticmethod
    def _cosine_lr(base_lr, step, total_steps, end_fraction=0.1):
        """Cosine anneal from base_lr to base_lr * end_fraction over total_steps."""
        if total_steps <= 1:
            return base_lr
        cosine = 0.5 * (1.0 + np.cos(np.pi * step / total_steps))
        return base_lr * (end_fraction + (1.0 - end_fraction) * cosine)

    def _set_lr(self, optimizer, lr):
        """Set learning rate on all param groups."""
        for pg in optimizer.param_groups:
            pg['lr'] = lr

    @staticmethod
    def _reset_optimizer(optimizer):
        """
        Reset Adam's internal momentum buffers (m and v) and step counter.
        
        Per "Resetting the Optimizer in Deep RL" (NeurIPS 2023):
        Stale moment estimates from previous iterations contaminate the current
        optimization landscape. Resetting lets Adam's bias correction handle
        the cold start cleanly — that's exactly what it was designed for.
        """
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state.get(p)
                if state is not None:
                    if 'exp_avg' in state:
                        state['exp_avg'].zero_()
                    if 'exp_avg_sq' in state:
                        state['exp_avg_sq'].zero_()
                    if 'step' in state:
                        state['step'].zero_()

    def update(self, obs, actions, old_log_probs, returns, advantages):
        """
        Phasic PPO update with optimizer reset + intra-iteration cosine LR annealing.
        
        At the start of each phase, Adam's momentum buffers are zeroed so the
        optimizer sees the current loss landscape fresh — no contamination from
        stale gradients. Cosine LR annealing within each phase gives large
        steps early (broad search) tapering to fine adjustments late (precision fit).
        
        Phase 1: Critic fits returns. Reset → cosine(lr_value → 0) over value_epochs.
        Phase 2: Recompute advantages with fitted critic. (No gradients.)
        Phase 3: Policy updates with clean advantages. Reset → cosine(lr_policy → 0) over policy_epochs.
        """
        N = obs.size(0)
        steps_per_epoch = max(1, (N + self.mini_batch_size - 1) // self.mini_batch_size)

        # ── PHASE 1: Fit the critic ──────────────────────────────────
        self._reset_optimizer(self.opt_value)
        total_vl = 0.0
        n_value_updates = 0
        total_value_steps = self.value_epochs * steps_per_epoch

        for epoch in range(self.value_epochs):
            perm = torch.randperm(N, device=self.device)
            for mb_idx, start in enumerate(range(0, N, self.mini_batch_size)):
                # Cosine anneal within this phase
                global_step = epoch * steps_per_epoch + mb_idx
                lr = self._cosine_lr(self.lr_value_base, global_step, total_value_steps)
                self._set_lr(self.opt_value, lr)

                idx = perm[start:start + self.mini_batch_size]
                mb_obs = obs[idx]
                mb_ret = returns[idx]

                with torch.no_grad():
                    old_values = self.value(mb_obs)

                values = self.value(mb_obs)
                values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)

                v_loss1 = torch.nn.functional.smooth_l1_loss(values, mb_ret, reduction='none')
                v_loss2 = torch.nn.functional.smooth_l1_loss(values_clipped, mb_ret, reduction='none')
                v_loss = torch.max(v_loss1, v_loss2).mean()

                self.opt_value.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.opt_value.step()

                total_vl += v_loss.item()
                n_value_updates += 1

        # Reset value LR for next iteration
        self._set_lr(self.opt_value, self.lr_value_base)

        # ── PHASE 2: Recompute advantages with the fitted critic ─────
        with torch.no_grad():
            fitted_values = self.value(obs)

        refined_adv = returns - fitted_values

        if refined_adv.numel() > 1:
            refined_adv = (refined_adv - refined_adv.mean()) / (refined_adv.std() + 1e-8)

        # ── PHASE 3: Policy update with clean advantages ─────────────
        self._reset_optimizer(self.opt_policy)
        total_pg = total_ent = 0.0
        n_policy_updates = 0
        total_policy_steps = self.policy_epochs * steps_per_epoch

        for epoch in range(self.policy_epochs):
            perm = torch.randperm(N, device=self.device)
            for mb_idx, start in enumerate(range(0, N, self.mini_batch_size)):
                # Cosine anneal within this phase
                global_step = epoch * steps_per_epoch + mb_idx
                lr = self._cosine_lr(self.lr_policy_base, global_step, total_policy_steps)
                self._set_lr(self.opt_policy, lr)

                idx = perm[start:start + self.mini_batch_size]
                mb_obs = obs[idx]
                mb_act = actions[idx]
                mb_old_lp = old_log_probs[idx]
                mb_adv = refined_adv[idx]

                new_lp, entropy = self.policy.evaluate_action(mb_obs, mb_act)

                ratio = (new_lp - mb_old_lp).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                loss = pg_loss - self.entropy_coeff * entropy.mean()

                self.opt_policy.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt_policy.step()

                total_pg += pg_loss.item()
                total_ent += entropy.mean().item()
                n_policy_updates += 1

        # Reset policy LR for next iteration
        self._set_lr(self.opt_policy, self.lr_policy_base)

        return {'pg_loss': total_pg / max(1, n_policy_updates),
                'v_loss': total_vl / max(1, n_value_updates),
                'entropy': total_ent / max(1, n_policy_updates)}

    def save(self, path):
        torch.save({'policy': self.policy.state_dict(),
                     'value': self.value.state_dict(),
                     'opt_policy': self.opt_policy.state_dict(),
                     'opt_value': self.opt_value.state_dict()}, path)

    def load(self, path, device='cpu'):
        ckpt = torch.load(path, map_location=device)
        self.policy.load_state_dict(ckpt['policy'])
        self.value.load_state_dict(ckpt['value'])
        self.opt_policy.load_state_dict(ckpt['opt_policy'])
        self.opt_value.load_state_dict(ckpt['opt_value'])

# =============================================================================
# 5. TRAINING LOOP (Episodic Autoregressive with GAE)
# =============================================================================
def run_training(config):
    device_str = config['rollout']['device']
    if device_str == 'cuda' and not torch.cuda.is_available():
        device_str = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"Device: {device}")

    ppo_cfg = config['ppo']
    num_seeds = ppo_cfg['num_seeds']
    batch_size = ppo_cfg['batch_size']
    collect_batches = ppo_cfg['collect_batches']
    num_iters = ppo_cfg['num_iterations']
    margin = ppo_cfg['seed_margin']
    mesh_fail_penalty = ppo_cfg.get('mesh_fail_penalty', -1.5)
    
    max_steps = ppo_cfg.get('episode_max_steps', 15)
    gamma = ppo_cfg.get('gamma', 0.99)
    gae_lambda = ppo_cfg.get('gae_lambda', 0.95)

    ckpt_dir = ppo_cfg['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    simulator = CrushSimulator(config, device)
    agent = PPOAgent(config).to(device)

    resume_path = os.path.join(ckpt_dir, "latest_policy.pth")
    if os.path.exists(resume_path):
        try:
            agent.load(resume_path, device)
            print(f"Resumed from {resume_path}")
        except RuntimeError:
            print("\nARCHITECTURE MISMATCH: Delete 'latest_policy.pth' and start fresh!")
            return

    print(f"\n PPO Autoregressive Training: {num_iters} iterations")
    print(f"   Seeds: {num_seeds} | Env Batch: {batch_size} | Episode Length: {max_steps}")
    print(f"   Action: ±{ppo_cfg['max_action']} mm | Discount (γ): {gamma} | GAE (λ): {gae_lambda}\n")

    best_cfe = 0.0

    try:
        for iteration in range(1, num_iters + 1):
            t0 = time.time()

            batch_obs, batch_act, batch_lp, batch_ret, batch_adv = [], [], [], [], []
            total_failed_steps = 0
            total_valid_steps = 0

            for _ in range(collect_batches):
                current_seeds = np.random.uniform(margin, BOX_SIZE - margin, size=(batch_size, num_seeds, 2))
                base_cfes, valid_idx = simulator.evaluate_seeds(current_seeds)
                
                current_cfe = torch.zeros(batch_size, device=device)
                active_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                
                for rank, orig_idx in enumerate(valid_idx):
                    current_cfe[orig_idx] = base_cfes[rank]
                    active_mask[orig_idx] = True

                ep_obs, ep_act, ep_lp, ep_val, ep_rew, ep_done, ep_valid = [], [], [], [], [], [], []
                
                # VIDEO TRACKING: Initialize history with the starting state
                seeds_history = [current_seeds.copy()]

                for step in range(max_steps):
                    ep_valid.append(active_mask.clone())
                    
                    proposed_seeds, obs, actions, log_probs, values = agent.propose_seeds(current_seeds, current_cfe=current_cfe)

                    active_idx_np = active_mask.nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
                    eval_seeds = [proposed_seeds[i] for i in active_idx_np]
                    
                    if len(eval_seeds) > 0:
                        step_cfes, step_valid_ranks = simulator.evaluate_seeds(eval_seeds)
                    else:
                        step_cfes, step_valid_ranks = torch.zeros(0, device=device), []

                    step_cfe_map = {}
                    for rank, idx_in_active in enumerate(step_valid_ranks):
                        b_idx = active_idx_np[idx_in_active]
                        step_cfe_map[b_idx] = step_cfes[rank]

                    rewards = torch.zeros(batch_size, device=device)
                    dones = torch.ones(batch_size, dtype=torch.bool, device=device)

                    for i in range(batch_size):
                        if active_mask[i]:
                            if i in step_cfe_map:
                                new_cfe = step_cfe_map[i]
                                rewards[i] = new_cfe - current_cfe[i]
                                current_cfe[i] = new_cfe
                                dones[i] = False
                                current_seeds[i] = proposed_seeds[i] 
                            else:
                                rewards[i] = mesh_fail_penalty
                                dones[i] = True
                                active_mask[i] = False
                                total_failed_steps += 1

                    ep_obs.append(obs)
                    ep_act.append(actions)
                    ep_lp.append(log_probs)
                    ep_val.append(values)
                    ep_rew.append(rewards)
                    ep_done.append(dones)
                    
                    # VIDEO TRACKING: Append the state after actions resolve
                    seeds_history.append(current_seeds.copy())

                # VIDEO GENERATION: Save 10% of the batch every single iteration
                num_to_save = max(1, int(batch_size * 0.10))
                
                # Pick 10% unique random environments from the batch
                vid_envs = np.random.choice(batch_size, num_to_save, replace=False)
                
                for vid_env in vid_envs:
                    seq = [h[vid_env] for h in seeds_history]
                    vid_dir = os.path.join(ckpt_dir, "videos")
                    
                    # Check the active mask to see if this specific environment died
                    status = "survived" if active_mask[vid_env] else "failed"
                    
                    # Pass the vid_env ID so the files don't overwrite each other in the same second
                    save_episode_video(seq, f"{iteration}_env{vid_env}", vid_dir, status=status)

                _, _, _, _, next_values = agent.propose_seeds(current_seeds, current_cfe=current_cfe)
                
                advantages = torch.zeros((max_steps, batch_size), device=device)
                last_gae_lam = 0
                for t in reversed(range(max_steps)):
                    if t == max_steps - 1:
                        next_non_terminal = 1.0 - ep_done[t].float()
                        next_values_t = next_values
                    else:
                        next_non_terminal = 1.0 - ep_done[t].float()
                        next_values_t = ep_val[t+1]

                    delta = ep_rew[t] + gamma * next_values_t * next_non_terminal - ep_val[t]
                    advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

                returns = advantages + torch.stack(ep_val)

                valid_mask = torch.stack(ep_valid).view(-1)
                total_valid_steps += valid_mask.sum().item()

                batch_obs.append(torch.stack(ep_obs).view(-1, obs.shape[-1])[valid_mask])
                batch_act.append(torch.stack(ep_act).view(-1, actions.shape[-1])[valid_mask])
                batch_lp.append(torch.stack(ep_lp).view(-1)[valid_mask])
                batch_ret.append(returns.view(-1)[valid_mask])
                batch_adv.append(advantages.view(-1)[valid_mask])

            obs_cat = torch.cat(batch_obs)
            act_cat = torch.cat(batch_act)
            lp_cat = torch.cat(batch_lp)
            ret_cat = torch.cat(batch_ret)
            adv_cat = torch.cat(batch_adv)

            if obs_cat.size(0) > 0:
                stats = agent.update(obs_cat, act_cat, lp_cat, ret_cat, adv_cat)
            else:
                stats = {'pg_loss': 0, 'v_loss': 0, 'entropy': 0}
                print("Warning: Zero valid transitions collected this batch.")

            dt = time.time() - t0
            
            mean_ret = ret_cat.mean().item() if ret_cat.size(0) > 0 else 0.0
            fail_pct = 100.0 * total_failed_steps / max(1, total_valid_steps + total_failed_steps)

            if iteration % ppo_cfg['log_interval'] == 0:
                print(f"Iter {iteration:4d} | "
                      f"R (Adv+V): {mean_ret:+.4f} | "
                      f"PG: {stats['pg_loss']:.4f} | "
                      f"VL: {stats['v_loss']:.4f} | "
                      f"H: {stats['entropy']:.2f} | "
                      f"Trans: {obs_cat.size(0)} | "
                      f"Fail: {fail_pct:.1f}% | "
                      f"{dt:.1f}s")

            if mean_ret > best_cfe and iteration > 1:
                best_cfe = mean_ret
                agent.save(os.path.join(ckpt_dir, "best_policy.pth"))

            if iteration % ppo_cfg['save_interval'] == 0:
                agent.save(os.path.join(ckpt_dir, f"policy_iter_{iteration}.pth"))
                agent.save(os.path.join(ckpt_dir, "latest_policy.pth"))

    finally:
        simulator.shutdown()
        agent.save(os.path.join(ckpt_dir, "latest_policy.pth"))

    print(f"\nTraining complete.")

# =============================================================================
# 6. ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Run PPO Crush Optimization")
    parser.add_argument('--config', type=str, default='config/PPG_config.yaml', help='Path to your config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    run_training(config)
