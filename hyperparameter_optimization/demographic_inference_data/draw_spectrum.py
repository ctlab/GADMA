import os
import pylab
import moments
import itertools
from generate_readme import load_module

def draw_spectrum(data, pop_ids, out_filename, projections=True):
    if data.ndim == 1:
        moments.Plotting.plot_1d_fs(data, fig_num=None, show=True, ax=None,
                                    out=out_filename, markersize=2, lw=1)
        pylab.savefig(out_filename)
        pylab.close('all')
    elif data.ndim == 2:
        pylab.figure(figsize=(3, 2.5))
        moments.Plotting.plot_single_2d_sfs(data, vmin=1.0, vmax=None,
                                            ax=None, pop_ids=pop_ids,
                                            extend='neither', colorbar=True,
                                            cmap=pylab.cm.hsv,
                                            out=out_filename, show=True)
        pylab.savefig(out_filename)
        pylab.close('all')
    elif data.ndim == 3 and not projections:
        moments.Plotting.plot_3d_spectrum(data, fignum=None, vmin=1.0,
                                          vmax=None, pop_ids=pop_ids,
                                          out=out_filename, show=False)
        pylab.close('all')
    else:
        npop = data.ndim
        data.mask[data == 0] = True
        npict = (npop - 1) * npop / 2
        if npop == 3:
            figsize = (6.5, 2)
            rows, cols = 1, 3
        if npop == 4:
            figsize = (7, 4)
            rows, cols = 2, 3
        if npop == 5:
            figsize = (12, 4)
            rows, cols = 2, 5
        f = pylab.figure(None, figsize=figsize)
        pylab.clf()
    #    extend = _extend_mapping[vmin <= datamin, vmax >= datamax]
        cptr = 0
        pairs = list(itertools.combinations(range(npop), 2))
        for i, j in pairs:
            ind = list(range(npop))
            ind.remove(j)
            ind.remove(i)
            marg_data = data
            for xi in reversed(range(npop - 2)):
                marg_data = marg_data.sum(axis=int(ind[xi]))
        
            curr_ids = [pop_ids[j], pop_ids[i]]
    
            ax = pylab.subplot(rows, cols, cptr + 1)
            plot_colorbar = (cptr == npict - 1)
            
            moments.Plotting.plot_single_2d_sfs(marg_data, vmin=1.0,
                                                vmax=None, pop_ids=curr_ids,
                                                extend='neither',
                                                colorbar=plot_colorbar)
            cptr += 1
        f.tight_layout()
        f.savefig(out_filename)
        pylab.close('all')


for dirname in os.listdir():
    if not os.path.isdir(dirname) or dirname.startswith("_") or\
            dirname.startswith("."):
        continue
    npop = int(dirname[0])
    pop_ids = load_module(dirname, "main_script.py").pop_labels

    if npop <= 3:
        filename = os.path.join(dirname, "fs_data.fs")
        data = moments.Spectrum.from_file(filename)
        out = os.path.join(dirname, "fs_plot.png")
        draw_spectrum(data, pop_ids, out, projections=False)
    if npop >= 3:
        filename = os.path.join(dirname, "fs_data.fs")
        data = moments.Spectrum.from_file(filename)
        out = os.path.join(dirname, "fs_plot_projections.png")
        draw_spectrum(data, pop_ids,  out, projections=True)
