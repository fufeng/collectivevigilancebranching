import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

P = dict(alpha=-2.0, r=8.0, N=16, vmax=3.0, G=250, gens=25000,
         beta=1.0, mu=0.02, sd=0.015, v0=1.0, seed=0, every=50, nbins=120)


def cost(v, alpha):
    if alpha == 0:
        return v.copy()
    return (np.exp(alpha * v) - 1) / alpha


def simulate(alpha, r, N, vmax, G, gens, beta, mu, sd, v0, seed, every, nbins):
    params = dict(alpha=alpha, r=r, N=N, vmax=vmax, G=G, gens=gens,
                  beta=beta, mu=mu, sd=sd, v0=v0, seed=seed, every=every, nbins=nbins)
    assert r > 0 and vmax > 0 and 0 <= v0 <= vmax
    assert N > 0 and G > 0 and gens > 0 and every > 0 and nbins > 0
    assert 0 <= mu <= 1 and sd >= 0
    rng = np.random.default_rng(seed)
    n = G * N
    v = np.full(n, v0)
    bins = np.linspace(0, vmax, nbins + 1)
    hist, times = [np.histogram(v, bins)[0]], [0]
    start = time.monotonic()
    for t in range(gens):
        rng.shuffle(v)
        X = v.reshape(G, N)
        w = (r * (1 - np.exp(-X.sum(1, keepdims=True))) - cost(X, alpha)).ravel()
        p = np.exp(beta * (w - w.max()))
        p /= p.sum()
        v = v[rng.choice(n, n, p=p)]
        m = rng.random(n) < mu
        v[m] = np.clip(v[m] + sd * rng.standard_normal(m.sum()), 0, vmax)
        if t % every == 0:
            hist.append(np.histogram(v, bins)[0])
            times.append(t + 1)
        if (t + 1) % 5000 == 0:
            print(f'{t+1:,}/{gens:,} generations; {time.monotonic()-start:.1f} s', flush=True)
    if times[-1] != gens:
        hist.append(np.histogram(v, bins)[0])
        times.append(gens)
    d = dict(hist=np.array(hist), times=np.array(times), bins=bins, v=v,
             vstar=np.log(r)/(N+alpha), **params)
    validate(d)
    return d


def validate(d):
    H, t, b, v = d['hist'], d['times'], d['bins'], d['v']
    n = int(d['G'])*int(d['N'])
    assert H.shape == (len(t), int(d['nbins']))
    assert np.all(H >= 0) and np.all(H.sum(axis=1) == n)
    assert np.all(np.diff(t) > 0) and t[0] == 0 and t[-1] == int(d['gens'])
    assert len(v) == n and np.isfinite(v).all()
    assert np.all((v >= 0) & (v <= float(d['vmax'])))
    assert np.array_equal(H[-1], np.histogram(v, b)[0])


def draw(ax, d):
    validate(d)
    H = d['hist'].astype(float)
    H /= H.sum(axis=1, keepdims=True)
    # The pseudocount is a display floor, not a change to population counts.
    floor = 1e-4
    assert np.all(H + floor > 0)
    bins = d['bins']
    mesh = ax.pcolormesh(d['times'], .5*(bins[1:]+bins[:-1]),
                         np.log10(H.T+floor), cmap='Greys', shading='auto',
                         vmin=-4, vmax=0, rasterized=True)
    star = float(d['vstar'])
    ax.axhline(star, color='#D55E00', lw=.8, ls='--')
    ax.text(float(d['gens'])*.98, star+.08,
            r'$v^*=\ln r/(N+\alpha)$',
            ha='right', va='bottom', color='#D55E00', fontsize=7,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))
    ax.set(xlim=(0,float(d['gens'])), ylim=(0,float(d['vmax'])),
           xlabel='Generation', ylabel=r'Vigilance trait, $v$',
           xticks=[0,5000,10000,15000,20000,25000], yticks=[0,1,2,3])
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((3,3))
    ax.xaxis.set_major_formatter(formatter)
    for side in ['top','right']:
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
        raise FileNotFoundError(f'{path.name} is missing. Run fig1D.py --simulate first.')
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)

def draw_figure(ax, data_dir):
    letter = 'D'
    data = load_data(letter, data_dir)
    draw(ax, data)
    checks = {'snapshots': len(data['times']), 'population_size': len(data['v']), 'singular_strategy': float(data['vstar']), 'final_above_one': int(np.count_nonzero(data['v'] > 1))}
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
    parser = argparse.ArgumentParser(description="Reproduce Figure 1D.")
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'fig1_output')
    parser.add_argument('--formats', nargs='+', choices=['pdf', 'svg', 'png', 'tiff'], default=['pdf', 'svg', 'png'])
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'fig1_data')
    parser.add_argument('--simulate', action='store_true', help='Regenerate this panel dataset before plotting (overwrites the selected data file).')
    args = parser.parse_args()
    if args.simulate:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        data = simulate(**P)
        np.savez_compressed(args.data_dir / 'panel_D.npz', **data)
    data_dir = args.data_dir
    style()
    width, height = 160., 43.
    fw, fh = 17.5 + width + 6., 16. + height + 9.
    fig = plt.figure(figsize=(fw / 25.4, fh / 25.4))
    rect = (17.5, 16., width, height)
    ax = fig.add_axes([17.5 / fw, 16. / fh, width / fw, height / fh])
    checks = draw_figure(ax, data_dir)
    add_letter(fig, 'D', rect)
    export(fig, 'fig1D', args.output_dir, {'D': checks}, args.formats)
    print(f"Saved fig1D to {args.output_dir.resolve()}")

if __name__ == '__main__':
    main()
