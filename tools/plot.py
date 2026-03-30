import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib import animation
import os

# ==========================================
# VISUALIZATION SETTINGS
# ==========================================
PLOT_BOX_SIZE_CM = 10.0
DOT_SIZE = 5
OPACITY = 0.6
EDGE_COLOR = 'red'
EDGE_WIDTH = 0.8


def plot_results(config):
    rollout_dir = os.path.expanduser(config['data']['rollout_dir'])
    rollout_path = os.path.join(rollout_dir, "rollout.npy")
    edges_path = os.path.join(rollout_dir, "world_edges.npy")
    force_path = os.path.join(rollout_dir, "force_data.npy")

    if not os.path.exists(rollout_path):
        print(f"Error: Rollout file not found at {rollout_path}")
        return

    # Load Data
    print(f"Loading rollout from {rollout_path}...")
    rollout_data = np.load(rollout_path)
    dim = rollout_data.shape[-1]
    print(f"Detected {dim}D rollout data. {rollout_data.shape[0]} frames, {rollout_data.shape[1]} nodes.")

    world_edges_data = None
    if os.path.exists(edges_path):
        world_edges_data = np.load(edges_path, allow_pickle=True)

    # Load force data if available
    force_data = None
    if os.path.exists(force_path):
        force_data = np.load(force_path, allow_pickle=True).item()
        print(f"Loaded force data: CFE = {force_data['cfe']:.4f}")

    rollout_data_cm = rollout_data * 100.0

    initial_pos = rollout_data_cm[0]
    center = np.mean(initial_pos, axis=0)
    half_size = PLOT_BOX_SIZE_CM / 2.0

    has_forces = force_data is not None and dim == 2

    if has_forces:
        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1], wspace=0.15)
        ax_mesh = fig.add_subplot(gs[0, 0])
        ax_force = fig.add_subplot(gs[0, 1])
    else:
        fig = plt.figure(figsize=(10, 10))
        if dim == 3:
            ax_mesh = fig.add_subplot(111, projection='3d')
        else:
            ax_mesh = fig.add_subplot(111)
        ax_force = None

    x_lim = [center[0] - half_size, center[0] + half_size]
    y_lim = [center[1] - half_size, center[1] + half_size]
    ax_mesh.set_xlim(x_lim)
    ax_mesh.set_ylim(y_lim)
    if dim == 3:
        z_lim = [center[2] - half_size, center[2] + half_size]
        ax_mesh.set_zlim(z_lim)

    if ax_force is not None:
        forces = force_data['forces']
        disp_mm = force_data['disp_mm']
        cfe = force_data['cfe']
        sample_interval = force_data.get('sample_interval', 3)

        max_disp = np.max(disp_mm) * 1.1 if np.max(disp_mm) > 0 else 1.0
        max_force = np.max(np.abs(forces)) * 1.1 if np.max(np.abs(forces)) > 0 else 1.0

        ax_force.set_xlim(0, max_disp)
        ax_force.set_ylim(0, max_force)
        ax_force.set_xlabel("Crush Displacement (mm)")
        ax_force.set_ylabel("Force (N)")
        ax_force.set_title(f"Force-Displacement (CFE = {cfe:.4f})")
        ax_force.grid(True, alpha=0.3)

        force_line, = ax_force.plot([], [], 'steelblue', linewidth=2)
        force_dot, = ax_force.plot([], [], 'ro', markersize=6)

    def frame_to_force_idx(frame_idx):
        """Rollout frame i corresponds to force sample i // sample_interval."""
        if force_data is None:
            return 0
        si = force_data.get('sample_interval', 3)
        return min(frame_idx // si, len(forces) - 1)

    def animate(i):
        curr_xlim = ax_mesh.get_xlim()
        curr_ylim = ax_mesh.get_ylim()
        if dim == 3:
            curr_zlim = ax_mesh.get_zlim()
            curr_elev = ax_mesh.elev
            curr_azim = ax_mesh.azim

        ax_mesh.clear()

        # --- MESH ---
        pos = rollout_data_cm[i]
        colors = pos[:, 1]

        if dim == 3:
            ax_mesh.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                            s=DOT_SIZE, c=colors, cmap='turbo', alpha=OPACITY)
        else:
            ax_mesh.scatter(pos[:, 0], pos[:, 1],
                            s=DOT_SIZE, c=colors, cmap='turbo', alpha=OPACITY)

        # --- EDGES ---
        title_text = f"Step {i}"

        if world_edges_data is not None and i > 0 and (i - 1) < len(world_edges_data):
            edges = world_edges_data[i - 1] 
            if len(edges) > 0:
                segments = pos[edges]
                if dim == 3:
                    lc = Line3DCollection(segments, colors=EDGE_COLOR,
                                          linewidths=EDGE_WIDTH, alpha=0.8)
                    ax_mesh.add_collection3d(lc)
                else:
                    lc = LineCollection(segments, colors=EDGE_COLOR,
                                        linewidths=EDGE_WIDTH, alpha=0.8)
                    ax_mesh.add_collection(lc)
                title_text += f" | Collisions: {len(edges)}"
            else:
                title_text += " | No Collisions"

        # --- RESTORE VIEW ---
        ax_mesh.set_xlim(curr_xlim)
        ax_mesh.set_ylim(curr_ylim)

        if dim == 3:
            ax_mesh.set_zlim(curr_zlim)
            ax_mesh.set_box_aspect([1, 1, 1])
            ax_mesh.set_zlabel("Z (cm)")
            ax_mesh.view_init(elev=curr_elev, azim=curr_azim)
        else:
            ax_mesh.set_aspect('equal')
            ax_mesh.grid(True, linestyle='--', alpha=0.3)

        ax_mesh.set_title(title_text)
        ax_mesh.set_xlabel("X (cm)")
        ax_mesh.set_ylabel("Y (cm)")

        # --- FORCE CURVE ---
        if ax_force is not None and i > 0:
            fi = frame_to_force_idx(i)
            if fi > 0:
                force_line.set_data(disp_mm[:fi + 1], np.abs(forces[:fi + 1]))
                force_dot.set_data([disp_mm[fi]], [np.abs(forces[fi])])

    num_frames = len(rollout_data)
    ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=1)
    plt.show()


def plot_force_static(config):
    """
    Static (non-animated) force-displacement plot.
    Call separately if you just want the force curve without animation.
    """
    rollout_dir = os.path.expanduser(config['data']['rollout_dir'])
    force_path = os.path.join(rollout_dir, "force_data.npy")

    if not os.path.exists(force_path):
        print("No force_data.npy found. Run rollout first.")
        return

    force_data = np.load(force_path, allow_pickle=True).item()
    forces = force_data['forces']
    disp_mm = force_data['disp_mm']
    energy = force_data['energy']
    cfe = force_data['cfe']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Force-Displacement
    ax1 = axes[0]
    ax1.plot(disp_mm, np.abs(forces), 'steelblue', linewidth=2)
    ax1.set_xlabel("Crush Displacement (mm)")
    ax1.set_ylabel("Force (N)")
    ax1.set_title(f"Force-Displacement (CFE = {cfe:.4f})")
    ax1.grid(True, alpha=0.3)

    # Energy vs Displacement
    ax2 = axes[1]
    ax2.plot(disp_mm, energy, 'darkorange', linewidth=2)
    ax2.set_xlabel("Crush Displacement (mm)")
    ax2.set_ylabel("Strain Energy (J)")
    ax2.set_title("Strain Energy vs Displacement")
    ax2.grid(True, alpha=0.3)

    # Force vs Time (sample index)
    ax3 = axes[2]
    ax3.plot(np.abs(forces), 'steelblue', linewidth=1.5)
    ax3.set_xlabel("Sample Index")
    ax3.set_ylabel("|Force| (N)")
    ax3.set_title("Force History")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(rollout_dir, "force_analysis.png"), dpi=300)
    print(f"Saved force_analysis.png")
    plt.show()


if __name__ == "__main__":
    config = {
        'data': {
            'rollout_dir': 'output/rollouts'
        }
    }

    import sys
    if '--static' in sys.argv:
        plot_force_static(config)
    else:
        plot_results(config)