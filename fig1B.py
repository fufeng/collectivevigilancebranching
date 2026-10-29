import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

P = dict(alpha=-2.0, r=8.0, N=16, vmax=3.0, n_grid=161, steps=60000, dt=0.05)
COLORS = ['#bbbbbb', '#cfe3f3', '#0072B2', '#7a3b00']
LABELS = {1:'1 sentinel (Nash)', 2:'2 sentinels (not Nash)', 3:'3+ sentinels'}


def classify(v, vmax):
    n_sent = (v > vmax - 1e-3).sum(-1)
    n_int = ((v > 1e-3) & (v < vmax - 1e-3)).sum(-1)
    return n_sent, n_int


def simulate(alpha, r, N, vmax, n_grid, steps, dt):
    assert r > 0 and N >= 2 and vmax > 0 and n_grid >= 2 and steps > 0 and dt > 0
    x = np.linspace(0, vmax, n_grid)
    X1, X2 = np.meshgrid(x, x)
    v = np.zeros((n_grid, n_grid, N))
    v[..., 0], v[..., 1] = X1, X2
    start = time.monotonic()
    for t in range(steps):
        g = r * np.exp(-v.sum(-1, keepdims=True)) - np.exp(alpha * v)
        v = np.clip(v + dt * g, 0, vmax)
        if (t + 1) % 10000 == 0:
            print(f'{t+1:,}/{steps:,} Euler steps; {time.monotonic()-start:.1f} s', flush=True)
    n_sent, n_int = classify(v, vmax)
    d = dict(x=x, n_sent=n_sent, n_int=n_int, final_v=v, alpha=alpha, r=r,
             N=N, vmax=vmax, n_grid=n_grid, steps=steps, dt=dt,
             final_time=steps*dt, classification_tolerance=1e-3)
    validate(d)
    return d


def classes(d):
    return np.clip(np.where(d['n_int'] > 0, 0, d['n_sent']), 0, 3)


def validate(d):
    x = np.asarray(d['x'])
    assert x.ndim == 1 and np.all(np.diff(x) > 0)
    assert np.isclose(x[0], 0) and np.isclose(x[-1], float(d['vmax']))
    for key in ['n_sent', 'n_int']:
        z = np.asarray(d[key])
        assert z.shape == (len(x), len(x)) and np.isfinite(z).all()
        assert np.all((z >= 0) & (z <= int(d['N'])) & (z == z.astype(int)))
    assert np.all(d['n_sent'] + d['n_int'] <= int(d['N']))
    if 'final_v' in d:
        v = d['final_v']
        assert v.shape == (len(x), len(x), int(d['N'])) and np.isfinite(v).all()
        assert np.all((v >= 0) & (v <= float(d['vmax'])))
        ns, ni = classify(v, float(d['vmax']))
        assert np.array_equal(ns, d['n_sent']) and np.array_equal(ni, d['n_int'])


def stability_markers(d):
    """Validate the corner equilibria, edge saddles, and Nash labels.

    These annotations apply to the default alpha=-2, r=8, V=3 case.
    At the origin all players initially move, so that marker denotes a
    symmetric initial condition, not a stationary zero-effort equilibrium.
    """
    alpha, r, vmax, N = (float(d['alpha']), float(d['r']),
                         float(d['vmax']), int(d['N']))
    assert alpha < -1 and r > 1 and N > 2
    interior = (np.log(r)-vmax)/(1+alpha)
    assert 0 < interior < vmax
    stable = np.array([[vmax,0], [0,vmax], [vmax,vmax]])
    edge = np.array([[vmax,interior], [interior,vmax]])
    residuals = []
    for pair in np.vstack([stable,edge]):
        v = np.r_[pair,np.zeros(N-2)]
        g = r*np.exp(-v.sum())-np.exp(alpha*v)
        projected = np.clip(v+g,0,vmax)-v
        residuals.append(float(np.max(np.abs(projected))))
        assert np.allclose(projected,0,atol=1e-12)
        assert np.all(g[v==0]<0) and np.all(g[v==vmax]>0)
    edge_unstable_eigenvalue = -(1+alpha)*np.exp(alpha*interior)
    assert edge_unstable_eigenvalue > 0
    # For alpha<-1, a single player's interior stationary point is a minimum;
    # global best replies are endpoints, so compare v=V against v=0.
    capped_cost = np.expm1(alpha*vmax)/alpha
    one_gain = r*(-np.expm1(-vmax))-capped_cost
    two_gain = r*np.exp(-vmax)*(-np.expm1(-vmax))-capped_cost
    assert one_gain > 0 and two_gain < 0
    symmetric_v = np.log(r)/(N+alpha)
    assert 0 < symmetric_v < vmax
    return {'filled_stable_corners':stable.tolist(),
            'open_edge_saddles':edge.tolist(), 'open_origin':[0.,0.],
            'maximum_projected_marker_residual':max(residuals),
            'edge_unstable_eigenvalue':float(edge_unstable_eigenvalue),
            'one_sentinel_payoff_gain_over_zero':float(one_gain),
            'second_sentinel_payoff_gain_over_zero':float(two_gain),
            'origin_is_equilibrium':False,
            'symmetric_interior_limit':float(symmetric_v)}


def draw(ax, d):
    d = {**P, **({'alpha':d['a']} if 'a' in d else {}), **d}
    validate(d)
    x, vmax = d['x'], float(d['vmax'])
    cls = classes(d)
    cmap = ListedColormap(COLORS)
    mesh = ax.pcolormesh(x, x, cls, cmap=cmap,
                         norm=BoundaryNorm(np.arange(-.5, 4), 4),
                         shading='auto', rasterized=True)
    present = [int(k) for k in np.unique(cls) if k != 0]
    ax.legend(handles=[Patch(fc=cmap(k), label=LABELS[k]) for k in present],
              fontsize=8.5, loc='upper right', framealpha=.95,
              bbox_to_anchor=(.97,.97), borderaxespad=0,
              handlelength=1., handletextpad=.5, borderpad=.4)
    ax.plot([0, vmax], [0, vmax], color='w', lw=.6, ls=':')
    markers = stability_markers(d)
    for name, points, fill in [
            ('stable-corners',markers['filled_stable_corners'],'#20262B'),
            ('edge-saddles',markers['open_edge_saddles'],'white'),
            ('symmetric-origin',[markers['open_origin']],'white')]:
        points = np.asarray(points)
        ax.plot(points[:,0], points[:,1], linestyle='none', marker='o',
                markersize=6, markerfacecolor=fill, markeredgecolor='#20262B',
                markeredgewidth=1.1, clip_on=False, zorder=10, gid=name)
    ax.set(xlabel=r'Initial vigilance, $v_1(0)$', ylabel=r'Initial vigilance, $v_2(0)$',
           aspect='equal', xlim=(0, vmax), ylim=(0, vmax),
           xticks=np.linspace(0, vmax, 4), yticks=np.linspace(0, vmax, 4))
    ax.tick_params(labelsize=8.5)
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    return mesh


import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.text import Text

"""Lossless merging of equal-color heatmap cells for vector exports."""
import numpy as np
from matplotlib.collections import PolyCollection

def vectorize_mesh(mesh):
    """Replace a rectilinear QuadMesh with an exactly matching rectangle union."""
    coords = np.asarray(mesh.get_coordinates())
    nr, nc = coords.shape[0]-1, coords.shape[1]-1
    rgba = np.asarray(mesh.get_facecolors())
    assert rgba.shape == (nr*nc, 4)
    # Horizontal colorbars store their rectangular grid with rows/columns swapped.
    if not np.allclose(coords[:,:,0], coords[0,:,0][None,:]):
        coords = coords.transpose(1,0,2)
        rgba = rgba.reshape(nr,nc,4).transpose(1,0,2).reshape(-1,4)
        nr,nc = nc,nr
    xs, ys = coords[0,:,0], coords[:,0,1]
    assert np.allclose(coords[:,:,0], xs[None,:])
    assert np.allclose(coords[:,:,1], ys[:,None])
    colors, ids = np.unique(rgba, axis=0, return_inverse=True)
    grid = ids.reshape(nr, nc)
    active, rectangles = {}, []
    for row in range(nr):
        bounds = np.r_[0, np.flatnonzero(np.diff(grid[row]))+1, nc]
        current = {}
        for left, right in zip(bounds[:-1], bounds[1:]):
            key = (int(left), int(right), int(grid[row,left]))
            current[key] = active.pop(key, row)
        for (left,right,color), start in active.items():
            rectangles.append((start,row,left,right,color))
        active = current
    for (left,right,color), start in active.items():
        rectangles.append((start,nr,left,right,color))
    # Reconstruct all rendered cell colors to verify merging is lossless.
    reconstructed = np.full(grid.shape, -1, dtype=int)
    coverage = np.zeros(grid.shape, dtype=np.uint8)
    verts, fills = [], []
    for bottom,top,left,right,color in rectangles:
        reconstructed[bottom:top,left:right] = color
        coverage[bottom:top,left:right] += 1
        verts.append([(xs[left],ys[bottom]), (xs[right],ys[bottom]),
                      (xs[right],ys[top]), (xs[left],ys[top])])
        fills.append(colors[color])
    assert np.array_equal(reconstructed,grid) and np.all(coverage==1)
    collection = PolyCollection(verts, facecolors=fills, edgecolors='none',
                                linewidths=0, antialiaseds=False, rasterized=False,
                                transform=mesh.get_transform(), zorder=mesh.get_zorder())
    ax = mesh.axes
    ax.add_collection(collection, autolim=False)
    collection.set_clip_on(mesh.get_clip_on())
    collection.set_clip_box(mesh.get_clip_box())
    if mesh.get_clip_path() is not None:
        collection.set_clip_path(mesh.get_clip_path())
    mesh.remove()
    return {'source_cells':int(nr*nc), 'vector_rectangles':len(rectangles),
            'distinct_colors':len(colors), 'lossless_color_and_coverage_check':True}



def style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8.5, 'axes.labelsize': 9., 'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5, 'axes.linewidth': .65,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.width': .65, 'ytick.major.width': .65,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'hatch.linewidth': .35, 'savefig.facecolor': 'white',
    })

def load_data(letter, data_dir):
    path = Path(data_dir) / f'panel_{letter}.npz'
    if not path.exists():
        raise FileNotFoundError(f'{path.name} is missing. Run fig1B.py --simulate first.')
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)

def draw_figure(ax, data_dir):
    letter = 'B'
    data = load_data(letter, data_dir)
    draw(ax, data)
    (values, counts) = np.unique(classes(data), return_counts=True)
    checks = {'initial_conditions': int(data['n_sent'].size), 'class_counts': {int(k): int(n) for (k, n) in zip(values, counts)}, 'markers': stability_markers(data)}
    # Apply the same type sizes to all panels, including scientific offsets.
    for text in ax.findobj(Text):
        text.set_fontsize(max(text.get_fontsize(), 8.5))
    ax.tick_params(labelsize=8.5)
    ax.xaxis.label.set_fontsize(9.)
    ax.yaxis.label.set_fontsize(9.)
    return checks

def add_letter(fig, letter, rect):
    x, y, width, height = rect
    fw, fh = fig.get_size_inches() * 25.4
    # A consistent physical offset reproduces the compact reference layout.
    fig.text((x - 12.) / fw, (y + height + .2) / fh, letter,
             ha='left', va='bottom', fontsize=15., weight='bold', color='black')

def export(fig, stem, output_dir, checks, formats=('pdf', 'svg', 'png')):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    mesh_reports = [vectorize_mesh(mesh) for mesh in fig.findobj(QuadMesh)]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if not (-1 <= box.x0 and -1 <= box.y0 and
                box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1):
            raise ValueError(f'Text outside canvas: {text.get_text()}')
    if 'pdf' in formats:
        fig.savefig(output_dir / f'{stem}.pdf', dpi=600)
    if 'svg' in formats:
        fig.savefig(output_dir / f'{stem}.svg', dpi=600)
    if 'png' in formats:
        fig.savefig(output_dir / f'{stem}.png', dpi=600)
    if 'tiff' in formats:
        fig.savefig(output_dir / f'{stem}.tiff', dpi=600,
                    pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(output_dir / f'{stem}_preview.png', dpi=300)
    report = {'canvas_mm': (fig.get_size_inches() * 25.4).tolist(),
              'all_text_within_canvas': True, 'vector_meshes': mesh_reports,
              'scientific_checks': checks}
    (output_dir / f'{stem}_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    plt.close(fig)
    return report
ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Reproduce Figure 1B.")
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'fig1_output')
    parser.add_argument('--formats', nargs='+', choices=['pdf', 'svg', 'png', 'tiff'], default=['pdf', 'svg', 'png'])
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'fig1_data')
    parser.add_argument('--simulate', action='store_true', help='Regenerate this panel dataset before plotting (overwrites the selected data file).')
    args = parser.parse_args()
    if args.simulate:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        data = simulate(**P)
        np.savez_compressed(args.data_dir / 'panel_B.npz', **data)
    data_dir = args.data_dir
    style()
    width, height = 44., 44.
    fw, fh = 17.5 + width + 6., 16. + height + 9.
    fig = plt.figure(figsize=(fw / 25.4, fh / 25.4))
    rect = (17.5, 16., width, height)
    ax = fig.add_axes([17.5 / fw, 16. / fh, width / fw, height / fh])
    checks = draw_figure(ax, data_dir)
    add_letter(fig, 'B', rect)
    export(fig, 'fig1B', args.output_dir, {'B': checks}, args.formats)
    print(f"Saved fig1B to {args.output_dir.resolve()}")

if __name__ == '__main__':
    main()
