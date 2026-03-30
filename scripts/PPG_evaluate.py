import torch
import numpy as np
import os
import math
import matplotlib.pyplot as plt

from PPG_train import PPOAgent, CrushSimulator
from tools.rl_environment import BOX_SIZE


def run_evaluation(config, num_trials=16, deterministic=True):
    """
    Load trained policy, generate optimized structures, compare vs random baseline.
    """
    device_str = config['rollout']['device']
    if device_str == 'cuda' and not torch.cuda.is_available():
        device_str = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device_str)

    ppo_cfg = config['ppo']
    num_seeds = ppo_cfg['num_seeds']
    margin = ppo_cfg['seed_margin']

    # Load policy
    agent = PPOAgent(config).to(device)
    ckpt_path = os.path.join(ppo_cfg['checkpoint_dir'], "best_policy.pth")
    if os.path.exists(ckpt_path):
        agent.load(ckpt_path, device)
        print(f"✅ Loaded policy from {ckpt_path}")
    else:
        print(f"⚠️  No checkpoint at {ckpt_path}, using random policy")

    agent.policy.eval()
    agent.value.eval()

    simulator = CrushSimulator(config, device)

    # --- A. EVALUATE OPTIMIZED STRUCTURES ---
    print(f"\n🔬 Evaluating {num_trials} optimized structures...")
    base_seeds = np.random.uniform(
        margin, BOX_SIZE - margin, size=(num_trials, num_seeds, 2)
    )

    B = base_seeds.shape[0]
    obs_raw = torch.tensor(base_seeds.reshape(B, -1), dtype=torch.float32, device=device)
    obs = (obs_raw - BOX_SIZE / 2.0) / (BOX_SIZE / 2.0)  # ← add this

    with torch.no_grad():
        if deterministic:
            actions, _, _ = agent.policy.get_action(obs, deterministic=True)
        else:
            actions, _, _ = agent.policy.get_action(obs, deterministic=False)

    clamped = actions.clamp(-ppo_cfg['max_action'], ppo_cfg['max_action'])
    deltas = clamped.cpu().numpy().reshape(B, num_seeds, 2)
    optimized_seeds = np.clip(base_seeds + deltas, margin, BOX_SIZE - margin)

    opt_list = [optimized_seeds[i] for i in range(num_trials)]
    opt_cfe_t, opt_valid = simulator.evaluate_seeds(opt_list)
    opt_cfe = opt_cfe_t.cpu().numpy()

    # --- B. BASELINE: RANDOM STRUCTURES (same base seeds, no policy) ---
    print(f"🎲 Evaluating {num_trials} random baseline structures...")
    rand_list = [base_seeds[i] for i in range(num_trials)]
    rand_cfe_t, rand_valid = simulator.evaluate_seeds(rand_list)
    rand_cfe = rand_cfe_t.cpu().numpy()

    # Clean up
    simulator.shutdown()

    # --- C. REPORT ---
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)

    if len(opt_cfe) > 0:
        print(f"\n  OPTIMIZED (Policy):")
        print(f"    Mean CFE: {opt_cfe.mean():.4f}")
        print(f"    Max CFE:  {opt_cfe.max():.4f}")
        print(f"    Min CFE:  {opt_cfe.min():.4f}")
        print(f"    Valid:    {len(opt_valid)}/{num_trials}")
        for i, idx in enumerate(opt_valid):
            print(f"    Env {idx}: CFE = {opt_cfe[i]:.4f}")

    if len(rand_cfe) > 0:
        print(f"\n  RANDOM (Baseline):")
        print(f"    Mean CFE: {rand_cfe.mean():.4f}")
        print(f"    Max CFE:  {rand_cfe.max():.4f}")
        print(f"    Min CFE:  {rand_cfe.min():.4f}")
        print(f"    Valid:    {len(rand_valid)}/{num_trials}")

    if len(opt_cfe) > 0 and len(rand_cfe) > 0:
        improvement = opt_cfe.mean() - rand_cfe.mean()
        print(f"\n  Δ Mean CFE: {improvement:+.4f} "
              f"({'better' if improvement > 0 else 'worse'})")

    print("=" * 60)

    # --- D. VISUALIZATION ---
    _plot_comparison(opt_cfe, opt_valid, rand_cfe, rand_valid, num_trials)
    _plot_seed_deltas(base_seeds, optimized_seeds, opt_cfe, opt_valid)


def _plot_comparison(opt_cfe, opt_valid, rand_cfe, rand_valid, num_trials):
    """Bar chart comparing optimized vs random CFE."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Per-structure comparison
    ax = axes[0]
    x = np.arange(num_trials)
    width = 0.35

    rand_vals = np.zeros(num_trials)
    opt_vals = np.zeros(num_trials)
    for i, idx in enumerate(rand_valid):
        rand_vals[idx] = rand_cfe[i]
    for i, idx in enumerate(opt_valid):
        opt_vals[idx] = opt_cfe[i]

    ax.bar(x - width / 2, rand_vals, width, label='Random', color='gray', alpha=0.7)
    ax.bar(x + width / 2, opt_vals, width, label='Optimized', color='steelblue', alpha=0.8)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal')
    ax.set_xlabel("Structure")
    ax.set_ylabel("CFE")
    ax.set_title("Per-Structure CFE")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # Distribution
    ax2 = axes[1]
    if len(rand_cfe) > 0:
        ax2.hist(rand_cfe, bins=15, alpha=0.6, color='gray', label='Random', edgecolor='black')
    if len(opt_cfe) > 0:
        ax2.hist(opt_cfe, bins=15, alpha=0.7, color='steelblue', label='Optimized', edgecolor='black')
    ax2.axvline(x=1.0, color='green', linestyle='--', label='Ideal')
    ax2.set_xlabel("CFE")
    ax2.set_ylabel("Count")
    ax2.set_title("CFE Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rl_eval_comparison.png", dpi=200)
    print("✅ Saved comparison to 'rl_eval_comparison.png'")
    plt.close()


def _plot_seed_deltas(base_seeds, optimized_seeds, cfe_scores, valid_idx):
    """Show how the policy moved each seed point."""
    n = min(8, len(valid_idx))
    if n == 0:
        return

    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for plot_i in range(n):
        env_i = valid_idx[plot_i]
        ax = axes[plot_i]

        base = base_seeds[env_i]
        opt = optimized_seeds[env_i]

        ax.set_xlim(0, BOX_SIZE)
        ax.set_ylim(0, BOX_SIZE)
        ax.set_aspect('equal')
        ax.set_title(f"Env {env_i} (CFE={cfe_scores[plot_i]:.3f})", fontsize=9)
        ax.grid(True, alpha=0.2)

        # Draw arrows from base to optimized
        for s in range(len(base)):
            ax.annotate("", xy=opt[s], xytext=base[s],
                         arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

        ax.scatter(base[:, 0], base[:, 1], c='gray', s=40, zorder=3, label='Base')
        ax.scatter(opt[:, 0], opt[:, 1], c='steelblue', s=40, zorder=3, label='Optimized')

        if plot_i == 0:
            ax.legend(fontsize=7)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("rl_eval_seed_deltas.png", dpi=200)
    print("✅ Saved seed deltas to 'rl_eval_seed_deltas.png'")
    plt.close()