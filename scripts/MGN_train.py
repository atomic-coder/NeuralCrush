import torch
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
import time

from src.data_formatter import DataFormatter
from src.model import MeshGraphNet
from scripts.MGN_evaluate import evaluate_epoch 


# =============================================================================
# FRAME-LEVEL BALANCED SAMPLER (60:40 High:Low Split)
# =============================================================================
def create_balanced_frame_sampler(dataset, threshold, high_ratio=0.6):
    """
    Scans every frame in the dataset and builds a WeightedRandomSampler 
    that ensures a target ratio of high-accel vs low-accel frames.
    
    Args:
        dataset: DataFormatter dataset
        threshold: acceleration magnitude threshold for "high" classification
        high_ratio: target fraction of high-accel frames in each epoch (default 0.6)
    """
    print(f"\nScanning dataset to build balanced frame sampler (threshold={threshold})...")
    
    num_high = 0
    num_low = 0
    is_high_frame = torch.zeros(len(dataset), dtype=torch.bool)
    
    for i in tqdm(range(len(dataset)), desc="Analyzing frames"):
        data = dataset[i]
        active_mask = ~data.mask.squeeze().bool()
        accel_mag = data.y.norm(dim=1)
        
        # Frame is "high" if ANY active node exceeds threshold
        if accel_mag[active_mask].max() >= threshold:
            is_high_frame[i] = True
            num_high += 1
        else:
            num_low += 1
    
    low_ratio = 1.0 - high_ratio
    
    # Weights: each high frame gets (high_ratio / num_high), 
    # each low frame gets (low_ratio / num_low)
    # This makes the expected sampled fraction match the target ratio.
    weight_high = high_ratio / max(1, num_high)
    weight_low = low_ratio / max(1, num_low)
    
    weights = torch.zeros(len(dataset))
    for i in range(len(dataset)):
        weights[i] = weight_high if is_high_frame[i] else weight_low
    
    sampler = WeightedRandomSampler(
        weights=weights, 
        num_samples=len(dataset),  
        replacement=True
    )
    
    print(f"📊 Frame Analysis Complete:")
    print(f"   High-accel frames: {num_high} ({100*num_high/len(dataset):.1f}% of dataset)")
    print(f"   Low-accel frames:  {num_low} ({100*num_low/len(dataset):.1f}% of dataset)")
    print(f"   Target sampling:   {high_ratio*100:.0f}% high / {low_ratio*100:.0f}% low")
    
    return sampler


# =============================================================================
# NODE-LEVEL LOSS MASK
# =============================================================================
def build_node_loss_mask(target, fixed_mask, threshold, low_node_sample_count=100):
    """
    Builds a boolean mask selecting which nodes contribute to the loss.
    
    High-accel frame (any node > threshold):
        - ALL high-accel nodes included
        - + `low_node_sample_count` randomly sampled low-accel nodes
        
    Quiet frame (no node > threshold):
        - `low_node_sample_count` randomly sampled active nodes
    
    Args:
        target: [N, spatial_dim] ground truth acceleration tensor
        fixed_mask: [N] or [N,1] boolean mask (True = fixed/constrained node)
        threshold: acceleration magnitude threshold
        low_node_sample_count: number of low-accel nodes to sample per frame
        
    Returns:
        loss_mask: [N] boolean tensor, True = include in loss
        is_high_frame: bool, whether this frame has high-accel nodes
        num_high: int, count of high-accel nodes selected
        num_low: int, count of low-accel nodes selected
    """
    device = target.device
    accel_mag = target.norm(dim=1)
    
    # Flatten fixed mask
    fixed = fixed_mask.squeeze().bool()
    active = ~fixed
    
    # Split active nodes into high and low acceleration
    high_accel = active & (accel_mag >= threshold)
    low_accel = active & (accel_mag < threshold)
    
    high_indices = torch.where(high_accel)[0]
    low_indices = torch.where(low_accel)[0]
    
    loss_mask = torch.zeros(target.shape[0], dtype=torch.bool, device=device)
    
    is_high_frame = len(high_indices) > 0
    
    if is_high_frame:
        # HIGH-ACCEL FRAME: all high nodes + random sample of low nodes
        loss_mask[high_indices] = True
        
        n_low_sample = min(low_node_sample_count, len(low_indices))
        if n_low_sample > 0:
            perm = torch.randperm(len(low_indices), device=device)[:n_low_sample]
            loss_mask[low_indices[perm]] = True
            
        num_high = len(high_indices)
        num_low = n_low_sample
    else:
        # QUIET FRAME: random sample of active nodes
        n_sample = min(low_node_sample_count, len(low_indices))
        if n_sample > 0:
            perm = torch.randperm(len(low_indices), device=device)[:n_sample]
            loss_mask[low_indices[perm]] = True
        
        num_high = 0
        num_low = n_sample
    
    return loss_mask, is_high_frame, num_high, num_low


# =============================================================================
# NOISE INJECTION (Unchanged from your original)
# =============================================================================
def add_training_noise(dt, batch, noise_std, gamma=0.1):
    """
    Exact implementation of Appendix A.2.2 from MeshGraphNet paper.
    Dynamically supports 2D and 3D data.
    """
    spatial_dim = batch.pos.shape[1] 
    
    vel_prev_clean = batch.velocities[:, :spatial_dim]
    vel_curr_clean = batch.velocities[:, spatial_dim:spatial_dim*2]
    
    pos_curr_clean = batch.pos
    pos_prev_clean = pos_curr_clean - (vel_curr_clean * dt)
    
    noise_curr = torch.randn_like(pos_curr_clean) * noise_std
    noise_prev = torch.randn_like(pos_prev_clean) * noise_std
    
    static_mask = batch.mask.squeeze().bool()
    noise_curr[static_mask] = 0.0
    noise_prev[static_mask] = 0.0
    
    pos_curr_noisy = pos_curr_clean + noise_curr
    pos_prev_noisy = pos_prev_clean + noise_prev
    
    vel_curr_noisy = (pos_curr_noisy - pos_prev_noisy) / dt
    vel_prev_noisy = vel_prev_clean + (noise_prev / dt)
    
    noise_term = (noise_curr * (1.0 + gamma)) - noise_prev
    correction_accel = - noise_term / (dt)
    
    noisy_targets = torch.clone((batch.y + correction_accel))
    noisy_pos = torch.clone(pos_curr_noisy)
    noisy_velocities = torch.cat([vel_prev_noisy, vel_curr_noisy], dim=1)

    return noisy_pos.detach(), noisy_velocities.detach(), noisy_targets.detach()


# =============================================================================
# TRAINING EPOCH (Modified with node-level loss masking)
# =============================================================================
def print_memory_stats(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[{stage}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
    elif torch.backends.mps.is_available():
        try:
            allocated = torch.mps.current_allocated_memory() / 1e9
            driver = torch.mps.driver_allocated_memory() / 1e9
            print(f"[{stage}] MPS Allocated: {allocated:.2f} GB | Driver: {driver:.2f} GB")
        except AttributeError:
            print(f"[{stage}] MPS enabled, but memory tracking requires PyTorch 2.0+")


def train_epoch(model, loader, optimizer, config, device, update_weights=True, 
                scheduler=None, accumulate_stats=True):
    context = torch.enable_grad() if update_weights else torch.no_grad()
    
    acc_steps = config['train']['target_batch_size'] // config['train']['batch_size']
    accel_threshold = config['train']['accel_threshold']
    low_node_sample_count = config['train']['low_node_sample_count']
    
    high_error_report = []

    with context:
        model.train()
        total_loss = 0.0
        grad_norm = 0
        num_batches = 0
        
        # Per-regime tracking
        total_loss_high = 0.0
        total_loss_low = 0.0
        count_high_frames = 0
        count_low_frames = 0
        total_high_nodes = 0
        total_low_nodes = 0
        
        if update_weights:
            optimizer.zero_grad()
        
        pbar = tqdm(loader, desc="Training" if update_weights else "Warming Up", leave=False)
        
        for i, batch in enumerate(pbar): 
            batch = batch.to(device)
            
            device_type = "cuda" if "cuda" in str(device) else "cpu"
            use_autocast = "cuda" in str(device)
            
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_autocast):

                clean_targets = batch.y.clone()

                # -----------------------------------------------------------
                # BUILD NODE LOSS MASK
                # -----------------------------------------------------------
                node_mask, is_high_frame, n_high, n_low = build_node_loss_mask(
                    target=clean_targets,
                    fixed_mask=batch.mask,
                    threshold=accel_threshold,
                    low_node_sample_count=low_node_sample_count
                )

                if config['train']['input_noise_std'] > 0.0:
                    batch.pos, batch.velocities, batch.y = add_training_noise(
                    dt=config['train']['time_step'],
                    batch=batch,
                    noise_std=config['train']['input_noise_std'],
                    gamma=config['train']['input_noise_gamma']
                )
                
                # -----------------------------------------------------------
                # FORWARD PASS (full graph, all nodes, unchanged)
                # -----------------------------------------------------------
                pred = model(batch, accumulate_stats=accumulate_stats)

                # -----------------------------------------------------------
                # LOSS COMPUTATION (only on selected nodes)
                # -----------------------------------------------------------
                loss = model.loss(
                    pred, batch.y, batch, 
                    mask=batch.mask, 
                    node_loss_mask=node_mask,
                    accumulate_stats=accumulate_stats
                )

                # -----------------------------------------------------------
                # MONITORING: per-regime loss
                # -----------------------------------------------------------
                
                with torch.no_grad():
                    is_steel = batch.node_attr[:, 0].bool()
                    is_tpu = ~is_steel
                    
                    # Denormalize predictions back to real acceleration space
                    
                    pred_raw = torch.zeros_like(pred)
                    pred_raw = model.absorber_target_normalizer.inverse(
                        pred.detach()
                    )
                
                    mse = ((pred_raw - batch.y) ** 2).detach()
                    node_mse = mse.mean(dim=1)
                    
                    accel_mag = clean_targets.norm(dim=1)
                    active_tpu = (~batch.mask.squeeze().bool()) & is_tpu
                    high_nodes = active_tpu & (accel_mag >= accel_threshold)
                    low_nodes = active_tpu & (accel_mag < accel_threshold)
                    
                    if high_nodes.sum() > 0:
                        total_loss_high += node_mse[high_nodes].mean().item()
                        count_high_frames += 1
                    if low_nodes.sum() > 0:
                        total_loss_low += node_mse[low_nodes].mean().item()
                        count_low_frames += 1
                    
                    total_high_nodes += n_high
                    total_low_nodes += n_low
                    
                    masked_node_mse = node_mse.clone()
                    masked_node_mse[is_steel] = 0.0
                    masked_node_mse[batch.mask.squeeze().bool()] = 0.0
                    
                    max_mse_val, max_idx = torch.max(masked_node_mse, dim=0)
                    active_mask = (~batch.mask.squeeze().bool()) & (~is_steel)
                    num_nodes_mse_gt1 = (node_mse[active_mask] > 0.2).sum().item()
                    
                    if max_mse_val.item() > 0.005:  # choose the appropriate threshold
                        error_event = {
                            'batch_step': i,
                            'frame_id': batch.frame_id,
                            'node_idx': max_idx.item(),
                            'mse': max_mse_val.item(),
                            'pos': batch.pos[max_idx].detach().float().cpu().tolist(),
                            'vel': batch.velocities[max_idx].detach().float().cpu().tolist(),
                            'target_raw': batch.y[max_idx].detach().float().cpu().tolist(),
                            'target_norm': batch.y[max_idx].detach().float().cpu().tolist(),
                            'pred_raw': pred[max_idx].detach().float().cpu().tolist(),
                            'pred_norm': pred[max_idx].detach().float().cpu().tolist(),  # <-- add this
                            'is_static': batch.mask[max_idx].item() if hasattr(batch, 'mask') else False,
                            'num_nodes_gt1': num_nodes_mse_gt1
                        }
                        high_error_report.append(error_event)
                
            if update_weights:
                scaled_loss = loss / acc_steps 
                scaled_loss.backward()
                
                if (i + 1) % acc_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad() 
    
            total_loss += loss.item()
            num_batches += 1
            
            if update_weights:
                lr = optimizer.param_groups[0]['lr']
                frame_type = "H" if is_high_frame else "Q"
                pbar.set_postfix({
                    'type': frame_type,
                    'nodes': f"{n_high}h+{n_low}l",
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{lr:.2e}",
                    'gnorm': f"{grad_norm:.1f}"
                })
            else:
                pbar.set_postfix({'status': 'accumulating stats'})

    # Handle any remaining gradients
    if update_weights and (num_batches % acc_steps != 0):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        optimizer.zero_grad()

    # Print per-regime loss summary
    avg_high = total_loss_high / max(1, count_high_frames)
    avg_low = total_loss_low / max(1, count_low_frames)
    print(f"Regime Loss → High-accel: {avg_high:.6f} ({count_high_frames} frames, {total_high_nodes} nodes) | "
          f"Low-accel: {avg_low:.6f} ({count_low_frames} frames, {total_low_nodes} nodes)")

    # High error report
    if len(high_error_report) > 0:
        print(f"\n{'='*80}")
        print(f"HIGH ERROR REPORT (Nodes with MSE > 20.0) | Count: {len(high_error_report)}")
        print(f"{'='*80}")
        for event in high_error_report[:10]:
            print(f"Frame/Batch: {event['frame_id']} | Node: {event['node_idx']} | MSE: {event['mse']:.4f} | Static: {event['is_static']} | Nodes MSE>1: {event['num_nodes_gt1']}")
            print(f"   Pos:  {['{:.3f}'.format(x) for x in event['pos']]}")
            print(f"   Vel:  {['{:.3f}'.format(x) for x in event['vel']]}")
            print(f"   Tgt (norm): {['{:.3f}'.format(x) for x in event['target_norm']]}")
            print(f"   Tgt (raw): {['{:.3f}'.format(x) for x in event['target_raw']]}")
            print(f"   Pred (norm): {['{:.3f}'.format(x) for x in event['pred_norm']]}")
            print(f"   Pred (raw):  {['{:.3f}'.format(x) for x in event['pred_raw']]}")
            print("-" * 40)
        print(f"{'='*80}\n")

    return total_loss / max(1, num_batches)


# =============================================================================
# NORMALIZER STATS PRINTER
# =============================================================================
def print_norm_stats(model):
    print("\n" + "="*60)
    print("NORMALIZATION STATISTICS (After Warmup)")
    print("="*60)

    def print_layer_stats(name, layer):
        if not hasattr(layer, 'acc_count'): return
        count = int(layer.acc_count.item())
        mean = layer._mean().detach().cpu().numpy()
        std = layer._std().detach().cpu().numpy()
        
        print(f"🔹 {name} (Samples Seen: {count})")
        mean_str = ", ".join([f"{x:.4e}" for x in mean.flatten()])
        std_str  = ", ".join([f"{x:.4e}" for x in std.flatten()])
        print(f"   Mean: [{mean_str}]")
        print(f"   Std:  [{std_str}]")
        print("-" * 60)

    if hasattr(model, 'node_normalizer'):
        print_layer_stats("Node Normalizer", model.node_normalizer)
    if hasattr(model, 'mesh_edge_normalizer'):
        print_layer_stats("Mesh Edge Normalizer", model.mesh_edge_normalizer)
    if hasattr(model, 'world_edge_normalizer'):
        print_layer_stats("World Edge Normalizer", model.world_edge_normalizer)
    if hasattr(model, 'impactor_target_normalizer'):
        print_layer_stats("Impactor Target Normalizer", model.impactor_target_normalizer)
    if hasattr(model, 'absorber_target_normalizer'):
        print_layer_stats("Absorber_Target Normalizer", model.absorber_target_normalizer)
    
    print("="*60 + "\n")


# =============================================================================
# TRAINING MANAGER (Called by main.py)
# =============================================================================
def run_training(config, device):

    torch.set_float32_matmul_precision('high')

    # 1. Load Datasets
    train_set = DataFormatter(config['data']['train_path'], augment=True)
    val_set = DataFormatter(config['data']['val_path'], augment=False)
    
    # 2. Build Balanced Frame Sampler
    accel_threshold = config['train']['accel_threshold']
    high_frame_ratio = config['train'].get('high_frame_ratio', 0.6)

    if config['train']['sample_frames']:
        train_sampler = create_balanced_frame_sampler(
            train_set, accel_threshold, high_ratio=high_frame_ratio
        )
    else:
        train_sampler=None
    
    # Use sampler instead of shuffle (mutually exclusive in DataLoader)
    train_loader = DataLoader(
        train_set, 
        batch_size=config['train']['batch_size'], 
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config['train']['num_workers']
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=config['train']['batch_size'], 
        shuffle=False, 
        num_workers=config['train']['num_workers'],
    )

    # 3. Create Model
    model = MeshGraphNet(config).to(device)
    if config['model']['compile_model']:
        print("Compiling the model for faster execution...")
        model = torch.compile(model, mode='default', dynamic=True)

    print("\n" + "="*60)
    print(f"MODEL ARCHITECTURE (Device: {device})")
    print("="*60)
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("-" * 60)
    print(f"🔹 Hidden Dimension:   {model.hidden_dim}")
    print(f"🔹 Interaction Layers: {model.num_layers}")
    print(f"🔹 Spatial Dimension:  {model.spatial_dim}D")
    print(f"🔹 Graph Radius:       {model.radius}")
    print(f"🔹 Total Parameters:   {total_params:,}")
    print(f"🔹 Trainable Params:   {trainable_params:,}")
    print(f"🔹 Accel Threshold:    {accel_threshold}")
    print(f"🔹 Frame Ratio:        {high_frame_ratio*100:.0f}% high / {(1-high_frame_ratio)*100:.0f}% low")
    print(f"🔹 Low Node Samples:   {config['train']['low_node_sample_count']}")
    print("="*60 + "\n")

    # 4. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['train']['learning_rate'], 
        weight_decay=1e-4
    )
    
    # 5. Scheduler
    start_lr = float(config['train']['learning_rate'])
    end_lr = 1e-6
    
    updates_per_epoch = len(train_loader) // max(1, (config['train']['target_batch_size'] // config['train']['batch_size']))
    total_steps = config['train']['epochs'] * updates_per_epoch
    
    decay_gamma = (end_lr / start_lr) ** (1.0 / total_steps) if total_steps > 0 else 1.0
    
    print(f"📉 Scheduler: Exponential Decay ({start_lr} -> {end_lr})")
    print(f"   Total Updates: {total_steps}")
    print(f"   Step Gamma:    {decay_gamma:.8f}")

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, 
        gamma=decay_gamma
    )
    
    # 6. Resume
    resume_path = config['train']['resume_path'] 
    starting_epoch = 1

    if resume_path is not None:
        if os.path.exists(resume_path):
            print(f"Loading checkpoint from {resume_path}...")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            print("✅ Weights loaded successfully (with strict=False).")
            
            starting_epoch = config['train'].get('resume_epoch', 1)

            if config['train'].get('recalibrate_normalizers', False):
                print("Clearing old Normalizer statistics for fresh re-calibration...")
                for module in model.modules():
                    if hasattr(module, 'acc_count'):
                        module.acc_count.fill_(0)
                        module.acc_sum.fill_(0)
                        module.acc_sum_squared.fill_(0)
        else:
            print(f"Warning: Resume path '{resume_path}' provided but file not found. Starting scratch.")

    # 7. Training Loop
    ckpt_dir = config['logging']['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    start_time = time.time()
    for epoch in range(starting_epoch, config['train']['epochs'] + 1):
        epoch_start = time.time()
        
        if epoch < starting_epoch + 1 and config['train'].get('recalibrate_normalizers', False):
            print("Warmup Epoch: Accumulating statistics without updating weights...")
            train_loss = train_epoch(model, train_loader, optimizer, config, device, update_weights=False)
            print(f"Warmup Complete. Normalizers initialized.")
            print_norm_stats(model)
            if starting_epoch > 1:  
                val_loss = evaluate_epoch(model, val_loader, config, device)
                print(f"Post-warmup validation: {val_loss:.6f}")
        else:
            train_loss = train_epoch(
                model, train_loader, optimizer, config, device, 
                update_weights=True, scheduler=scheduler, accumulate_stats=False
            )
            val_loss = evaluate_epoch(model, val_loader, config, device)

            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            remaining = epoch_time * (config['train']['epochs'] - epoch)

            print(f"Epoch {epoch} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Time: {epoch_time:.1f}s | Elapsed: {elapsed:.1f}s | ETA: {remaining/60:.1f}min")
            
            latest_path = os.path.join(ckpt_dir, "latest_model.pth")
            torch.save(model.state_dict(), latest_path)

            save_freq = config['logging']['save_frequency']
            if epoch % save_freq == 0:
                history_path = os.path.join(ckpt_dir, f"model_epoch_{epoch}.pth")
                torch.save(model.state_dict(), history_path)
                print(f"Checkpoint saved: {history_path}")