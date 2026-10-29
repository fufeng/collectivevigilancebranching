import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Polygon

N, V = 16, 3.0
ALPHA_LIMITS = (-4.0, 0.0)
LOG10_R_LIMITS = (-1.0, 5.0)
LN10 = np.log(10.0)
SLOPE = V/LN10
PURPLE, INK = '#8054A3', '#25343D'
COLORS = ['#879BAA', '#00589C', '#00866A', '#B66B00',
          '#C43C39', '#303B49', '#A23B72', '#6B7419']
INTERIOR_COLORS = ['#F6E9C8', '#EFD5BA', '#E7C5C1', '#DDD7ED']
POINTS = {'B':(-2.0,8.0), 'C':(-0.5,float(np.exp(4.0))), 'D':(-2.0,8.0)}


def clip_polygon(vertices, a, b, c):
    """Clip a polygon in (alpha, log10 r) to a*x+b*y <= c."""
    if not vertices:
        return []
    result = []
    previous = vertices[-1]
    fp = a*previous[0] + b*previous[1] - c
    for current in vertices:
        fc = a*current[0] + b*current[1] - c
        if (fp <= 0) != (fc <= 0):
            t = fp/(fp-fc)
            result.append((previous[0]+t*(current[0]-previous[0]),
                           previous[1]+t*(current[1]-previous[1])))
        if fc <= 0:
            result.append(current)
        previous, fp = current, fc
    return result


def domain():
    x0,x1 = ALPHA_LIMITS
    y0,y1 = LOG10_R_LIMITS
    return [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]


def capped_region(n):
    poly = domain()
    if n > 0:
        poly = clip_polygon(poly, SLOPE, -1, -n*SLOPE)
    if n < N:
        poly = clip_polygon(poly, 0, 1, n*SLOPE)
    return poly


def stable_counts(alpha, r):
    assert r > 0
    L = np.log(r)
    return [n for n in range(N+1)
            if (n==0 or L>(n+alpha)*V) and (n==N or L<n*V)]


def interior_count(alpha, r):
    if not -1 < alpha < 0:
        return None
    L = np.log(r)
    for n in range(N):
        if n*V < L < (n+1+alpha)*V:
            return n
    return None


def nash_counts(alpha, r):
    assert r > 0
    if alpha < -1:
        cost_ratio = np.expm1(alpha*V)/(alpha*(-np.expm1(-V)))
        q = cost_ratio/r
        return [n for n in range(N+1)
                if (n==0 or q<=np.exp(-(n-1)*V)) and
                   (n==N or q>=np.exp(-n*V))]
    if -1 < alpha < 0:
        if r < 1:
            return [0]
        if r > 1:
            return list(range(min(N,int(np.ceil(np.log(r)/V))),N+1))
    return None


def add_polygon(ax, vertices, **kwargs):
    if len(vertices) >= 3:
        xy = np.asarray(vertices, dtype=float)
        xy[:,1] = 10.0**xy[:,1]
        ax.add_patch(Polygon(xy, closed=True, **kwargs))


def draw_panel(ax, font_scale=1.0):
    """Draw on an existing axis; return analytical labels and marker checks."""
    body = 11.5*font_scale
    xmin,xmax = ALPHA_LIMITS
    ymin,ymax = LOG10_R_LIMITS
    ax.set_yscale('log', base=10)
    ax.set(xlim=ALPHA_LIMITS, ylim=(10.0**ymin,10.0**ymax),
           xlabel=r'Vigilance cost scaling, $\alpha$',
           ylabel=r'Predation threat (in Log scale), $r$')
    ax.set_box_aspect(1)
    ax.set_xticks(np.arange(-4,1))
    powers = np.arange(int(ymin),int(ymax)+1)
    ax.set_yticks(10.0**powers, [rf'$10^{{{p}}}$' for p in powers])
    ax.minorticks_off()
    ax.tick_params(labelsize=body)
    ax.xaxis.label.set_fontsize(12.5*font_scale)
    ax.yaxis.label.set_fontsize(12.5*font_scale)
    ax.spines[['top','right']].set_visible(False)
    visible_counts = []
    alpha = np.linspace(xmin,xmax,1201)
    for n in range(N+1):
        poly = capped_region(n)
        if len(poly)<3:
            continue
        color = COLORS[n % len(COLORS)]
        visible_counts.append(n)
        add_polygon(ax,poly,facecolor=color,edgecolor='none',alpha=.16,zorder=1)
        if n>0:
            ax.plot(alpha,10.0**((n+alpha)*SLOPE),color='white',lw=3.4,zorder=3)
            ax.plot(alpha,10.0**((n+alpha)*SLOPE),color=color,lw=1.8,zorder=4)
        if n<N and ymin<=n*SLOPE<=ymax:
            ax.axhline(10.0**(n*SLOPE),color='white',lw=3.4,zorder=3)
            ax.axhline(10.0**(n*SLOPE),color=color,lw=1.8,zorder=4)
    interior_records = []
    for n in range(N):
        poly = clip_polygon(domain(),-1,0,1)
        poly = clip_polygon(poly,0,-1,-n*SLOPE)
        poly = clip_polygon(poly,-SLOPE,1,(n+1)*SLOPE)
        if len(poly)<3:
            continue
        color = INTERIOR_COLORS[min(n,3)]
        add_polygon(ax,poly,facecolor=color,edgecolor='#AB9574',
                    linewidth=0,hatch='//',zorder=2)
        x,y = (-.43,.65) if n==0 else (-.49,n*SLOPE+.65)
        if not ymin<y<ymax:
            continue
        label = '1 interior' if n==0 else f'{n} + 1 interior'
        assert interior_count(x,10**y)==n
        ax.text(-.045,10**y,label,ha='right',va='center',fontsize=body,
                color='#805F2C',bbox=dict(facecolor=color,edgecolor='none',alpha=.94,pad=.7),zorder=8)
        interior_records.append({'capped':n,'alpha':x,'log10_r':y})
    ax.axvline(-1,color='#7C8287',ls=':',lw=.9,zorder=5)
    # Global Nash comparisons have different thresholds from local stability.
    left_alpha = np.linspace(xmin,-1,1001)
    argument = np.expm1(left_alpha*V)/(left_alpha*(-np.expm1(-V)))
    assert np.all(argument>0)
    nash_labels = [(-2.10,-.85,'Nash (0 sentinels)'),(-.49,-.78,'Nash (0)')]
    for n in range(N):
        boundary = np.log10(argument)+n*SLOPE
        if boundary.min()>ymax:
            break
        ax.plot(left_alpha,10.0**boundary,color=PURPLE,ls=(0,(4,2.5)),lw=1.3,zorder=6)
        x = -3.20 if n==4 else (-2.65 if n==1 else -2.10)
        ratio = np.expm1(x*V)/(x*(-np.expm1(-V)))
        y = np.log10(ratio)+n*SLOPE+.14
        if ymin<y<ymax:
            word = 'sentinel' if n==0 else 'sentinels'
            nash_labels.append((x,float(y),f'Nash ({n+1} {word})'))
        if ymin<n*SLOPE<ymax:
            ax.plot([-1,0],[10.0**(n*SLOPE)]*2,color=PURPLE,ls=(0,(4,2.5)),lw=1.3,zorder=6)
            nash_labels.append((-.49,n*SLOPE+.14,f'Nash ({n+1}–{N})'))
    # Each horizontal strip has constant minimum capped count; its diagonal
    # cuts partition all alternative maximum counts, including n=0 correctly.
    edges = sorted(set([ymin,ymax]+[n*SLOPE for n in range(N+1) if ymin<n*SLOPE<ymax]))
    region_records = []
    for lo,hi in zip(edges[:-1],edges[1:]):
        y = (lo+hi)/2
        cuts = sorted(set([xmin,xmax]+[y/SLOPE-n for n in range(1,N+1)
                                      if xmin<y/SLOPE-n<xmax]))
        for lo_x,hi_x in zip(cuts[:-1],cuts[1:]):
            x = (lo_x+hi_x)/2
            counts = stable_counts(x,10**y)
            if not counts:
                continue
            if lo_x<-1<hi_x and counts != [0]:
                x = (lo_x-1)/2
            if counts == [0]:
                x = -.49
            x = max(x,-3.70)
            label = '('+','.join(map(str,counts))+')'
            assert stable_counts(x,10**y)==counts
            ax.text(x,10**y,label,ha='center',va='center',fontsize=body,color=INK,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.78,pad=.7),zorder=7)
            region_records.append({'counts':counts,'alpha':x,'log10_r':y})
    nash_records = []
    for x,y,label in nash_labels:
        expected = nash_counts(x,10**y)
        numbers = [int(v) for v in re.findall(r'\d+',label)]
        displayed = list(range(numbers[0],numbers[1]+1)) if len(numbers)==2 else numbers
        assert displayed == expected, (label,x,y,expected)
        ax.text(x,10**y,label,ha='center',va='center',fontsize=body,color=PURPLE,
                bbox=dict(boxstyle='round,pad=.16',facecolor='white',
                          edgecolor=PURPLE,lw=.55,alpha=.96),zorder=9)
        nash_records.append({'alpha':x,'log10_r':y,'label':label,'counts':expected})
    # Shared parameters are marked at one exact coordinate, with no jitter.
    for name,xy,offset in [('B, D',POINTS['B'],(20,-17)),('C',POINTS['C'],(17,-1))]:
        ax.plot(*xy,'o',ms=6.5,mfc='#111111',mec='white',mew=.9,zorder=11)
        ax.annotate(name,xy=xy,xytext=offset,textcoords='offset points',
                    ha='left',va='center',fontsize=12*font_scale,weight='bold',
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.95,pad=1),
                    arrowprops=dict(arrowstyle='-',color='#333333',lw=.8),zorder=12)
    return {'N':N,'V':V,'alpha_range':list(ALPHA_LIMITS),'log10_r_range':list(LOG10_R_LIMITS),
            'vertical_axis':'r on a base-10 logarithmic scale',
            'visible_capped_counts':visible_counts,'region_labels':region_records,
            'interior_labels':interior_records,'nash_labels':nash_records,
            'panel_points':{k:{'alpha':a,'r':r,'log10_r':float(np.log10(r)),
                              'stable_capped_counts':stable_counts(a,r),
                              'interior_capped_count':interior_count(a,r),
                              'nash_positive_counts':nash_counts(a,r)}
                            for k,(a,r) in POINTS.items()}}


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
        raise FileNotFoundError(f'{path.name} is missing. Run fig1A.py --simulate first.')
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)

def draw_figure(ax, data_dir):
    letter = 'A'
    checks = draw_panel(ax, font_scale=8.5 / 11.5)
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
    parser = argparse.ArgumentParser(description="Reproduce Figure 1A.")
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'fig1_output')
    parser.add_argument('--formats', nargs='+', choices=['pdf', 'svg', 'png', 'tiff'], default=['pdf', 'svg', 'png'])
    args = parser.parse_args()
    data_dir = None
    style()
    width, height = 100., 100.
    fw, fh = 17.5 + width + 6., 16. + height + 9.
    fig = plt.figure(figsize=(fw / 25.4, fh / 25.4))
    rect = (17.5, 16., width, height)
    ax = fig.add_axes([17.5 / fw, 16. / fh, width / fw, height / fh])
    checks = draw_figure(ax, data_dir)
    add_letter(fig, 'A', rect)
    export(fig, 'fig1A', args.output_dir, {'A': checks}, args.formats)
    print(f"Saved fig1A to {args.output_dir.resolve()}")

if __name__ == '__main__':
    main()
