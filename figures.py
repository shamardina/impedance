# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import rc
import tables
from parameters import stoich

rc("font", **{"family":"sans-serif", "sans-serif":"Arial"})
rc("text", usetex=True)
rc("text.latex", unicode=True)
plt.ioff()

colors = ["red", "blue", "green", "cyan", "magenta", "black", "orange", "None"]
linestyles = ["-", "--", ":", "", "-.", ""]
linewidths = [1, 2, 4]
markers = ["o", "s", "^", "*", ".", "+", ""]

def make_ticks(dims, step):
    start = min(dims)
    end = max(dims)
    return [step*i+start for i in xrange(-1, int((end-start)/step)+2)]

dim_x = (0.0, 2.0)
dim_y = (0.0, 1.0)
dim_xp = (0.0, 0.2)
dim_yp = (0.0, 0.2)
area = 5.0
scale = 10000.0
Is = [0.1, 0.2, 0.3, 0.4, 0.5]
Js = [j/area*scale for j in Is]
flows = [10.0, 15.0, 200.0]
labels = ["Numeric fit"]*3

for j in xrange(0, len(Js)):
    J = Js[j]
    plot_width = 9
    adjust = 0.2
    plot_height = plot_width*(dim_y[1] - dim_y[0])/(dim_x[1] - dim_x[0]) + adjust
    plt.clf()
    fig = plt.figure(figsize=(plot_width, plot_height))
    fontsize = 18
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=adjust)
    label = u"$Z'$ / $\\Omega$ cm$^2$"
    # TODO: label = u"$\\tilde{J} \\tilde{Z}'$"
    ax.set_xlabel(label, fontsize=fontsize)
    label = u"$-Z''$ / $\\Omega$ cm$^2$"
    # TODO: label = u"$- \\tilde{J} \\tilde{Z}''$"
    ax.set_ylabel(label, fontsize=fontsize)

    spectrum = []
    for i in xrange(0, len(flows)):
        lam = stoich(J, flows[i], area)
        filename = "data/%.1fA-%dml.dat" % (Is[j], int(flows[i]))
        lines = [line.strip("\n").split("\t") for line in list(open(filename))[1:]]
        Z1 = [float(line[2])*area for line in lines]
        Z2 = [float(line[3])*area for line in lines]
        freqs = [float(line[1]) for line in lines]
        label = u"Experiment, $\\lambda=$ %.2f" % lam
        spectrum.append(ax.plot(Z1, Z2,
                                color=colors[i], linestyle=linestyles[-1], linewidth=linewidths[0],
                                marker=markers[0], markerfacecolor=colors[-1], markeredgecolor=colors[i],
                                label=label)[0])
        
        filename = "results/I%.2f/transient-fitting-lambda%.2f.h5" % (J/scale, lam)
        t = tables.open_file(filename, "r")
        Z1 = t.root.found_ImpCCLFastO2_results.Z_1_v[:]
        Z2 = t.root.found_ImpCCLFastO2_results.Z_2_v[:]
        lam_eff = t.root.found_performance._v_attrs.lambda_eff
        t.close()
        label = u"%s, $\\lambda_{eff}=$ %.2f" % (labels[i], lam_eff)
        spectrum.append(ax.plot([z*scale for z in Z1], [z*scale for z in Z2],
                                color=colors[i], linestyle=linestyles[0], linewidth=linewidths[0],
                                marker=markers[-1],
                                label=label)[0])

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    ax.axis("scaled")
    ax.set_xlim(*dim_x)
    ax.set_ylim(*dim_y)
    ax.legend(spectrum, [l.get_label() for l in spectrum], loc="upper right", numpoints=3)
    filename = "results/I%.2f/Nyquist%d.eps" % (J/scale, int(J/10.0))
    fig.savefig(filename)

    ax.legend().set_visible(False)
    ax.set_xlim(*dim_xp)
    ax.set_ylim(*dim_yp)

    filename = "results/I%.2f/Nyquist%d-part.eps" % (J/scale, int(J/10.0))
    fig.savefig(filename)
