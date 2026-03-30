import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
import numpy as np
import yaml
import multiprocessing as mp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import Voronoi
from shapely.geometry import LineString, MultiPolygon, box as shapely_box
from shapely.ops import unary_union
import gmsh

from tools.rl_environment import (generate_rl_environment, BOX_SIZE, WEB_THICKNESS, FRAME_THICKNESS)
from scripts.PPG_train import PPOAgent, CrushSimulator, extract_topology_geometric


# =============================================================================
# STEP EXPORT (for COMSOL verification)
# =============================================================================
def export_step(seeds, save_path, box_size=BOX_SIZE, web_t=WEB_THICKNESS,
                frame_t=FRAME_THICKNESS, thickness=35.0):
    """
    Builds the Voronoi absorber geometry (no impactor), extrudes to 
    the specified thickness (mm), and exports as a STEP file for COMSOL.
    """
    dummies = np.array([[-500, -500], [box_size+500, -500],
                        [box_size+500, box_size+500], [-500, box_size+500]])
    all_pts = np.vstack([seeds, dummies])
    vor = Voronoi(all_pts)

    lines = []
    for ridge in vor.ridge_vertices:
        if -1 not in ridge:
            p1, p2 = vor.vertices[ridge[0]], vor.vertices[ridge[1]]
            lines.append(LineString([p1, p2]))

    merged = unary_union(lines)
    web = merged.buffer(web_t / 2.0, cap_style=2, join_style=2)

    target_box = shapely_box(0, 0, box_size, box_size)
    frame = target_box.exterior.buffer(frame_t / 2.0, cap_style=2, join_style=2)

    shape = unary_union([web, frame]).intersection(target_box)
    shape = shape.simplify(0.05, preserve_topology=False)

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("STEP_Export")

        polys = shape.geoms if isinstance(shape, MultiPolygon) else [shape]
        surface_tags = []
        for poly in polys:
            def draw_ring(ring):
                coords = list(ring.coords)[:-1]
                p_tags = [gmsh.model.occ.addPoint(x, y, 0) for x, y in coords]
                l_tags = [gmsh.model.occ.addLine(p_tags[i], p_tags[(i+1) % len(p_tags)])
                          for i in range(len(p_tags))]
                return gmsh.model.occ.addCurveLoop(l_tags)

            ext_loop = draw_ring(poly.exterior)
            int_loops = [draw_ring(hole) for hole in poly.interiors]
            tag = gmsh.model.occ.addPlaneSurface([ext_loop] + int_loops)
            surface_tags.append(tag)

        gmsh.model.occ.synchronize()

        # Extrude all surfaces by thickness in the Z direction
        for tag in surface_tags:
            gmsh.model.occ.extrude([(2, tag)], 0, 0, thickness)

        gmsh.model.occ.synchronize()

        gmsh.write(save_path)
        print(f"Exported STEP ({thickness}mm thick, no impactor): {save_path}")

    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def build_physical_shape(seeds, box_size=BOX_SIZE, web_t=WEB_THICKNESS, frame_t=FRAME_THICKNESS):
    """
    Reconstruct the actual thick-walled Voronoi structure from seeds.
    Returns a Shapely geometry (the real physical shape the impactor crushes).
    """
    dummies = np.array([[-500, -500], [box_size+500, -500],
                        [box_size+500, box_size+500], [-500, box_size+500]])
    all_pts = np.vstack([seeds, dummies])

    try:
        vor = Voronoi(all_pts)
    except Exception:
        return None

    lines = []
    for ridge in vor.ridge_vertices:
        if -1 not in ridge:
            p1, p2 = vor.vertices[ridge[0]], vor.vertices[ridge[1]]
            lines.append(LineString([p1, p2]))

    if not lines:
        return None

    merged = unary_union(lines)
    web = merged.buffer(web_t / 2.0, cap_style=2, join_style=2)

    target_box = shapely_box(0, 0, box_size, box_size)
    frame = target_box.exterior.buffer(frame_t / 2.0, cap_style=2, join_style=2)

    shape = unary_union([web, frame]).intersection(target_box)
    return shape


def draw_structure(ax, shape, seeds, seed_colors, num_seeds, box_size=BOX_SIZE):
    """Draw the thick-walled Voronoi structure with holes on an axis."""
    if shape is not None and not shape.is_empty:
        polys = shape.geoms if isinstance(shape, MultiPolygon) else [shape]
        for poly in polys:
            ext = MplPolygon(np.array(poly.exterior.coords), closed=True,
                             facecolor='#2196F3', edgecolor='black',
                             linewidth=0.5, alpha=0.7)
            ax.add_patch(ext)
            for interior in poly.interiors:
                hole = MplPolygon(np.array(interior.coords), closed=True,
                                  facecolor='white', edgecolor='#2196F3',
                                  linewidth=0.3, alpha=1.0)
                ax.add_patch(hole)

    ax.scatter(seeds[:, 0], seeds[:, 1], c=seed_colors[:num_seeds],
               s=100, zorder=8, edgecolors='black', linewidths=0.8)


def render_structure_video(trajectory_seeds, cfe_history, save_path, box_size=BOX_SIZE):
    """
    Renders video showing the actual thick-walled Voronoi structure evolving,
    with seed trails and progressive CFE curve.
    """
    num_frames = len(trajectory_seeds)
    num_seeds = trajectory_seeds[0].shape[0]
    seed_colors = plt.cm.Set1(np.linspace(0, 1, max(num_seeds, 3)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 7),
                              gridspec_kw={'width_ratios': [1.2, 1]})

    # Precompute all shapes (fast — no gmsh, just Shapely)
    print("🔧 Precomputing Voronoi structures for video...")
    shapes = [build_physical_shape(s) for s in trajectory_seeds]

    def update(frame):
        # === LEFT: Physical Structure ===
        ax = axes[0]
        ax.clear()
        ax.set_xlim(-3, box_size + 3)
        ax.set_ylim(-3, box_size + 3)
        ax.set_aspect('equal')
        ax.set_xlabel("X (mm)", fontsize=10)
        ax.set_ylabel("Y (mm)", fontsize=10)

        seeds = trajectory_seeds[frame]

        # Draw structure
        draw_structure(ax, shapes[frame], seeds, seed_colors, num_seeds, box_size)

        # Draw seed trails
        for s in range(num_seeds):
            trail_x = [trajectory_seeds[f][s, 0] for f in range(frame + 1)]
            trail_y = [trajectory_seeds[f][s, 1] for f in range(frame + 1)]
            ax.plot(trail_x, trail_y, '-', color=seed_colors[s],
                    linewidth=1.5, alpha=0.4, zorder=6)

        # Draw initial positions (hollow)
        init = trajectory_seeds[0]
        ax.scatter(init[:, 0], init[:, 1], c='none',
                   edgecolors=seed_colors[:num_seeds], s=80,
                   linewidths=2, zorder=7, marker='o')

        # Draw movement arrows from previous step
        if frame > 0:
            prev = trajectory_seeds[frame - 1]
            for s in range(num_seeds):
                dx = seeds[s, 0] - prev[s, 0]
                dy = seeds[s, 1] - prev[s, 1]
                if abs(dx) > 0.05 or abs(dy) > 0.05:
                    ax.annotate("", xy=seeds[s], xytext=prev[s],
                                arrowprops=dict(arrowstyle="-|>", color=seed_colors[s],
                                                lw=2, alpha=0.7),
                                zorder=9)

        # Title
        title = f"Step {frame}"
        if frame < len(cfe_history):
            title += f"  |  CFE = {cfe_history[frame]:.4f}"
            if frame > 0:
                delta = cfe_history[frame] - cfe_history[frame - 1]
                title += f"  (Δ{delta:+.4f})"
        ax.set_title(title, fontsize=12, fontweight='bold')

        # === RIGHT: CFE Progress ===
        ax2 = axes[1]
        ax2.clear()
        ax2.set_title("CFE Optimization Progress", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Step", fontsize=10)
        ax2.set_ylabel("CFE", fontsize=10)
        ax2.set_xlim(-0.5, num_frames - 0.5)

        if len(cfe_history) > 1:
            pad = max(0.01, (max(cfe_history) - min(cfe_history)) * 0.15)
            ax2.set_ylim(min(cfe_history) - pad, max(cfe_history) + pad)
        else:
            ax2.set_ylim(0, 1)

        ax2.grid(True, alpha=0.3)

        cfes = cfe_history[:frame + 1]

        if len(cfes) > 0:
            # Color segments green/red by improvement
            for i in range(1, len(cfes)):
                color = '#4CAF50' if cfes[i] >= cfes[i-1] else '#F44336'
                ax2.plot([i-1, i], [cfes[i-1], cfes[i]], '-', color=color, linewidth=2.5)
                ax2.plot(i, cfes[i], 'o', color=color, markersize=6, zorder=4)

            ax2.plot(0, cfes[0], 'o', color='gray', markersize=8, zorder=5)
            ax2.plot(len(cfes)-1, cfes[-1], 'o', color='red', markersize=12, zorder=6)

            ax2.axhline(y=cfe_history[0], color='gray', linestyle='--',
                         alpha=0.5, label=f'Start: {cfe_history[0]:.4f}')
            ax2.axhline(y=max(cfes), color='green', linestyle=':',
                         alpha=0.4, label=f'Best: {max(cfes):.4f}')
            ax2.legend(fontsize=9, loc='lower right')

    plt.tight_layout()

    print(f"🎬 Rendering {num_frames} frames...")
    anim = FuncAnimation(fig, update, frames=num_frames, interval=600)

    try:
        anim.save(save_path, writer='ffmpeg', dpi=150, fps=2)
        print(f"Saved MP4: {save_path}")
    except Exception as e:
        gif_path = save_path.replace('.mp4', '.gif')
        print(f"ffmpeg failed ({e}), saving GIF...")
        anim.save(gif_path, writer=PillowWriter(fps=2), dpi=150)
        print(f"Saved GIF: {gif_path}")

    plt.close(fig)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    with open("config/PPG_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    config['mesh']['num_workers'] = 1

    num_seeds = config['ppo']['num_seeds']
    margin = config['ppo']['seed_margin']

    # --- Load Agent ---
    agent = PPOAgent(config).to(device)
    ckpt_path = os.path.join(config['ppo']['checkpoint_dir'], "best_policy.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(config['ppo']['checkpoint_dir'], "latest_policy.pth")

    agent.load(ckpt_path, device)
    agent.policy.eval()
    print(f"Loaded policy from {ckpt_path}")

    # --- Load Simulator ---
    simulator = CrushSimulator(config, device)

    # --- Generate Valid Starting Structure ---
    print("Finding valid starting structure...")
    current_seeds = None
    for attempt in range(50):
        seeds = np.random.uniform(margin, BOX_SIZE - margin, size=(num_seeds, 2))
        data = generate_rl_environment(seeds)
        if data is not None:
            current_seeds = seeds
            print(f"Valid structure on attempt {attempt + 1}")
            break

    if current_seeds is None:
        raise RuntimeError("Failed to generate valid starting mesh")

    # --- Evaluate Initial CFE ---
    trajectory_seeds = [current_seeds.copy()]
    cfe_history = []

    init_cfe, init_valid = simulator.evaluate_seeds([current_seeds])
    if len(init_valid) > 0:
        cfe_history.append(init_cfe[0].item())
        print(f"Initial CFE: {cfe_history[0]:.4f}")
    else:
        cfe_history.append(0.0)
        print("Initial structure failed physics check")

    # --- Best Tracking ---
    best_cfe = cfe_history[0]
    best_seeds = current_seeds.copy()
    best_step = 0
    
    current_cfe = init_cfe

    # --- Autoregressive Optimization ---
    STEPS = config['inference'].get('steps', 15)
    print(f"\nOptimizing for {STEPS} steps (deterministic)...")
    print("-" * 55)

    for step in range(STEPS):
        proposed, _, _, _, _ = agent.propose_seeds(
            current_seeds[np.newaxis], current_cfe=current_cfe, deterministic=True
        )
        current_seeds = proposed[0]
        trajectory_seeds.append(current_seeds.copy())

        step_cfe, step_valid = simulator.evaluate_seeds([current_seeds])
        current_cfe = step_cfe
        if len(step_valid) > 0:
            cfe_history.append(step_cfe[0].item())
            delta = cfe_history[-1] - cfe_history[-2]
            marker = "📈" if delta > 0.001 else "📉" if delta < -0.001 else "➡️"
            print(f"  Step {step+1:2d}: CFE = {cfe_history[-1]:.4f}  "
                  f"Δ = {delta:+.4f}  {marker}")

            # Track best
            if cfe_history[-1] > best_cfe:
                best_cfe = cfe_history[-1]
                best_seeds = current_seeds.copy()
                best_step = step + 1
        else:
            print(f"  Step {step+1:2d}: ❌ Mesh failed")
            cfe_history.append(cfe_history[-1])
            break

    simulator.shutdown()

    # --- Summary ---
    print("-" * 55)
    total_delta = cfe_history[-1] - cfe_history[0]
    improving = sum(1 for i in range(1, len(cfe_history)) if cfe_history[i] > cfe_history[i-1])
    print(f"   CFE: {cfe_history[0]:.4f} → {cfe_history[-1]:.4f}  (Δ = {total_delta:+.4f})")
    print(f"   Best: {best_cfe:.4f} at step {best_step}")
    print(f"   Improving steps: {improving}/{len(cfe_history)-1}")

    # --- Save Outputs ---
    # --- Save Outputs ---
    out_dir = "inference_output"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Export BEST structure (NPY and STEP)
    step_path = os.path.join(out_dir, f"best_structure_step{best_step}_cfe{best_cfe:.4f}.step")
    export_step(best_seeds, step_path)
    np.save(os.path.join(out_dir, "best_seeds.npy"), best_seeds)

    # 2. Export INITIAL structure for comparison (NPY and STEP)
    init_step_path = os.path.join(out_dir, f"initial_structure_cfe{cfe_history[0]:.4f}.step")
    export_step(trajectory_seeds[0], init_step_path)
    np.save(os.path.join(out_dir, "initial_seeds.npy"), trajectory_seeds[0])

    # 3. Export FINAL structure (NPY and STEP)
    final_step = len(cfe_history) - 1
    final_cfe = cfe_history[-1]
    final_seeds = trajectory_seeds[-1]
    
    final_step_path = os.path.join(out_dir, f"final_structure_step{final_step}_cfe{final_cfe:.4f}.step")
    export_step(final_seeds, final_step_path)
    np.save(os.path.join(out_dir, "final_seeds.npy"), final_seeds)

    # Also export initial for comparison
    init_step_path = os.path.join(out_dir, f"initial_structure_cfe{cfe_history[0]:.4f}.step")
    export_step(trajectory_seeds[0], init_step_path)

    # Static CFE plot
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(1, len(cfe_history)):
        color = '#4CAF50' if cfe_history[i] >= cfe_history[i-1] else '#F44336'
        ax.plot([i-1, i], [cfe_history[i-1], cfe_history[i]], 'o-', color=color,
                linewidth=2, markersize=6)
    ax.plot(0, cfe_history[0], 'o', color='gray', markersize=10)
    ax.axhline(y=cfe_history[0], color='gray', linestyle='--', alpha=0.5,
               label=f'Start: {cfe_history[0]:.4f}')
    ax.axhline(y=max(cfe_history), color='green', linestyle=':', alpha=0.4,
               label=f'Best: {max(cfe_history):.4f}')
    ax.axvline(x=best_step, color='green', linestyle=':', alpha=0.3)
    ax.set_xlabel("Step")
    ax.set_ylabel("CFE")
    ax.set_title(f"Optimization: {cfe_history[0]:.4f} → Best {best_cfe:.4f} (step {best_step})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cfe_progress.png"), dpi=200)
    print(f"Saved {out_dir}/cfe_progress.png")
    plt.close()

    # Side-by-side initial vs best
    fig, axes_pair = plt.subplots(1, 2, figsize=(12, 6))
    for idx, (title, frame_seeds, cfe_val) in enumerate([
        ("Initial", trajectory_seeds[0], cfe_history[0]),
        (f"Best (Step {best_step})", best_seeds, best_cfe)
    ]):
        ax = axes_pair[idx]
        ax.set_xlim(-3, BOX_SIZE + 3)
        ax.set_ylim(-3, BOX_SIZE + 3)
        ax.set_aspect('equal')

        shape = build_physical_shape(frame_seeds)
        sc = plt.cm.Set1(np.linspace(0, 1, max(num_seeds, 3)))
        draw_structure(ax, shape, frame_seeds, sc, num_seeds)

        ax.set_title(f"{title} (CFE = {cfe_val:.4f})", fontsize=13, fontweight='bold')
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "initial_vs_best.png"), dpi=200)
    print(f"Saved {out_dir}/initial_vs_best.png")
    plt.close()

    # Animated video
    render_structure_video(
        trajectory_seeds, cfe_history,
        save_path=os.path.join(out_dir, "optimization_trajectory.mp4")
    )

    # Raw data
    np.save(os.path.join(out_dir, "seed_trajectory.npy"), np.array(trajectory_seeds))
    np.save(os.path.join(out_dir, "cfe_history.npy"), np.array(cfe_history))
    print(f"Saved raw data to {out_dir}/")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()