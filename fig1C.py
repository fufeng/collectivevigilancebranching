import numpy as np
import matplotlib
matplotlib.use('Agg')

P = dict(alpha=-0.5, r=float(np.exp(4)), N=16, vmax=3.0, duration=160.0,
         dt=0.001, initial_high=0.34, initial_low=0.18)
BLUE, ORANGE, GRAY = '#247B9C', '#BA7434', '#626C72'
LABELS = ['1 sentinel','1 interior','14 nonsentinels']


def simulate(alpha, r, N, vmax, duration, dt, initial_high, initial_low):
    assert r>0 and vmax>0 and dt>0 and duration>0
    assert N==16 and 0<=initial_low<=initial_high<=vmax
    steps = round(duration/dt)
    assert np.isclose(steps*dt,duration)
    time = np.arange(steps+1)*dt
    traits = np.empty((len(time),N))
    traits[0] = np.linspace(initial_high,initial_low,N)
    for k in range(steps):
        gradient = r*np.exp(-traits[k].sum())-np.exp(alpha*traits[k])
        traits[k+1] = np.clip(traits[k]+dt*gradient,0,vmax)
    validate(time,traits,vmax)
    return dict(time=time, traits=traits, alpha=alpha, r=r, N=N, vmax=vmax,
                duration=duration, dt=dt, initial_high=initial_high,initial_low=initial_low)


def validate(time, traits, vmax=3.0):
    assert time.ndim==1 and traits.shape==(len(time),16)
    assert np.isfinite(time).all() and np.isfinite(traits).all()
    assert np.all(np.diff(time)>0) and np.all((traits>=0)&(traits<=vmax))
    # The three final-state labels apply to this parameter configuration.
    assert np.allclose(traits[-1],[3.,2.]+[0.]*14,atol=1e-7)


def draw_panel(ax, time, traits, font_scale=1.0):
    validate(time,traits)
    for i in range(2,16):
        ax.plot(time,traits[:,i],color='#A6ADB1',lw=.8)
    ax.plot(time,traits[:,0],color=BLUE,lw=1.6)
    ax.plot(time,traits[:,1],color=ORANGE,lw=1.6)
    for value,label,color in zip([3,2,0],LABELS,[BLUE,ORANGE,GRAY]):
        ax.text(time[-1]*.97,value+.10,label,ha='right',va='bottom',
                color=color,fontsize=12*font_scale)
    ax.set(xlim=(float(time[0]),float(time[-1])),ylim=(-.15,3.6),
           xticks=[0,40,80,120,160],yticks=[0,1,2,3],
           xlabel='Time',ylabel=r'Individual vigilance, $v_i$')
    ax.set_box_aspect(1)
    ax.tick_params(labelsize=11.5*font_scale)
    ax.xaxis.label.set_fontsize(13*font_scale)
    ax.yaxis.label.set_fontsize(13*font_scale)
    ax.spines[['top','right']].set_visible(False)
    return {'times':len(time),'individuals':traits.shape[1],'final_traits':traits[-1].tolist()}


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
        raise FileNotFoundError(f'{path.name} is missing. Run fig1C.py --simulate first.')
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)

def draw_figure(ax, data_dir):
    letter = 'C'
    data = load_data(letter, data_dir)
    checks = draw_panel(ax, data['time'], data['traits'], font_scale=8.5 / 12)
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
    parser = argparse.ArgumentParser(description="Reproduce Figure 1C.")
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'fig1_output')
    parser.add_argument('--formats', nargs='+', choices=['pdf', 'svg', 'png', 'tiff'], default=['pdf', 'svg', 'png'])
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'fig1_data')
    parser.add_argument('--simulate', action='store_true', help='Regenerate this panel dataset before plotting (overwrites the selected data file).')
    args = parser.parse_args()
    if args.simulate:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        data = simulate(**P)
        np.savez_compressed(args.data_dir / 'panel_C.npz', **data)
    data_dir = args.data_dir
    style()
    width, height = 44., 44.
    fw, fh = 17.5 + width + 6., 16. + height + 9.
    fig = plt.figure(figsize=(fw / 25.4, fh / 25.4))
    rect = (17.5, 16., width, height)
    ax = fig.add_axes([17.5 / fw, 16. / fh, width / fw, height / fh])
    checks = draw_figure(ax, data_dir)
    add_letter(fig, 'C', rect)
    export(fig, 'fig1C', args.output_dir, {'C': checks}, args.formats)
    print(f"Saved fig1C to {args.output_dir.resolve()}")

if __name__ == '__main__':
    main()
