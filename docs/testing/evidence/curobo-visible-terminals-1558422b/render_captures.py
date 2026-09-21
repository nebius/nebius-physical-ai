"""Reproduce factual plots from the included sanitized recorded coordinates."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def render(meta, row, destination):
    goal = np.asarray(row['query']['goal_pose']['position_xyz'], dtype=float)
    path = np.asarray((row.get('trajectory') or {}).get('tool_position') or [], dtype=float)
    all_points = np.vstack([path, goal]) if len(path) else goal.reshape(1, 3)
    span = max(float(np.ptp(all_points, axis=0).max()), 0.12)
    center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
    bounds = [[float(c - span * 0.65), float(c + span * 0.65)] for c in center]
    fig = plt.figure(figsize=(12, 8), dpi=120)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    if len(path):
        line, = ax.plot(*path.T, lw=4, color='#0068b5', zorder=1, label=f'complete tool path ({len(path)} samples)')
        assert np.array_equal(np.asarray(line.get_data_3d()).T, path)
        ax.scatter(*path[0], c='#00a651', s=110, edgecolors='white', linewidths=1, label='path start', depthshade=False, zorder=2)
        ax.scatter(*path[-1], facecolors='none', edgecolors='#6f42c1', s=650, linewidths=3, marker='s', label='path terminal', depthshade=False, zorder=3)
    else:
        ax.text2D(0.5, 0.55, 'NO TRAJECTORY EMITTED', transform=ax.transAxes, ha='center', fontsize=18, color='#b42318', weight='bold')
    ax.scatter(*goal, c='#d62728', s=180, marker='*', label='declared goal', depthshade=False, zorder=4)
    for name, limits in zip(('set_xlim', 'set_ylim', 'set_zlim'), bounds):
        getattr(ax, name)(*limits)
    ax.set_xlabel('tool x (m)')
    ax.set_ylabel('tool y (m)')
    ax.set_zlabel('tool z (m)')
    ax.view_init(elev=25, azim=-55)
    ax.set_box_aspect((1, 1, 1))
    ax.grid(True, alpha=0.35)
    extra = '\nNO FAILURE OBSERVED — LAST SUCCESS FALLBACK' if meta.get('role') == 'last_success_no_failure_observed' else ''
    ax.set_title(f"{meta['hardware']} factual cuRobo capture\n{row['dataset']} | {row['mode']} | {row['problem_id']}\nSTATUS: {row['status'].upper()}{extra}", weight='bold', fontsize=11)
    ax.legend(loc='upper left')
    fig.text(0.01, 0.01, 'Review scope: factual status, declared goal, and recorded tool path only. No collision, torque, or robot-safety claim.', fontsize=8)
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return {'path_samples': len(path), 'path_coordinates_sha256': hashlib.sha256(path.astype('<f8').tobytes()).hexdigest(), 'goal': goal.tolist(), 'terminal': path[-1].tolist() if len(path) else None, 'all_samples_inside_bounds': bool(all((np.all(all_points[:, i] > lo) and np.all(all_points[:, i] < hi) for i, (lo, hi) in enumerate(bounds)))), 'bounds': bounds, 'image_sha256': sha(destination)}

if __name__ == '__main__':
    root = Path(__file__).parent
    output = root/'reproduced'
    output.mkdir(exist_ok=False)
    for item in json.loads((root/'render-data.json').read_text()):
        assert Path(item['filename']).name == item['filename']
        render(item['meta'],item['row'],output/item['filename'])
