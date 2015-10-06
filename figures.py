# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import rc
import tables
from parameters import stoich
from figstyles import *

rc("font", **{"family":"sans-serif", "sans-serif":"Arial"})
rc("text", usetex=True)
rc("text.latex", unicode=True)
plt.ioff()

dim_x = (0.0, 3.0)
dim_y = (0.0, 2.6)
dim_xp = (0.0, 1.0)
dim_yp = (0.0, 1.0)
area = 5.0
scale = 10000.0
Is = [0.1, 0.2, 0.3, 0.4, 0.5][0:3]
Js = [j/area*scale for j in Is]
flows = [10.0, 15.0, 200.0]
labels = ["Fit"]*len(flows)

plot_width = 9
adjust = 0.2
plot_height = plot_width*(dim_y[1] - dim_y[0])/(dim_x[1] - dim_x[0]) + adjust
fontsize = 18
for i in xrange(0, len(Js)):
    plt.clf()
    fig = plt.figure(figsize=(plot_width, plot_height))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=adjust)
    label = u"$Z'$ / $\\Omega$ cm$^2$"
    ax.set_xlabel(label, fontsize=fontsize)
    label = u"$-Z''$ / $\\Omega$ cm$^2$"
    ax.set_ylabel(label, fontsize=fontsize)

    spectrum = []
    for j in xrange(0, len(flows)):
        lam = stoich(Js[i], flows[j], area)
        filename = "data/%.1fA-%dml.dat" % (Is[i], int(flows[j]))
        lines = [line.strip("\n").split("\t") for line in list(open(filename))[1:]]
        Z1 = [float(line[2])*area for line in lines]
        Z2 = [float(line[3])*area for line in lines]
        freqs = [float(line[1]) for line in lines]
        filename = "results/I%.2f/1000-transient-fitting-lambda%.2f.h5" % (Js[i]/scale, lam)
        t = tables.open_file(filename, "r")
        R_Ohm = t.root.found_performance._v_attrs.R_Ohm
        lam_eff = t.root.found_performance._v_attrs.lam_eff
        label = u"Experiment, $\\lambda=$ %.2f" % lam
        spectrum.append(ax.plot([z - R_Ohm*scale for z in Z1], Z2,
                                color=colors[j], linestyle=linestyles[-1], linewidth=linewidths[0],
                                marker=markers[0], markerfacecolor=colors[-1], markeredgecolor=colors[j],
                                label=label)[0])

        Z1 = t.root.found_ImpCCLFastO2_results.Z_1_v[:]
        Z2 = t.root.found_ImpCCLFastO2_results.Z_2_v[:]
        t.close()
        label = u"%s, $\\lambda_{eff}=$ %.2f, $\\lambda_{eff}/\\lambda=$ %.2f" % (labels[j], lam_eff, lam_eff/lam)
        spectrum.append(ax.plot([(z - R_Ohm)*scale for z in Z1], [z*scale for z in Z2],
                                color=colors[j], linestyle=linestyles[0], linewidth=linewidths[0],
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
    # ax.legend().set_visible(False)
    filename = "results/I%.2f/fit-Nyquist.eps" % (Js[i]/scale)
    fig.savefig(filename)

    ax.set_xlim(*dim_xp)
    ax.set_ylim(*dim_yp)
    ax.legend().set_visible(False)
    filename = "results/I%.2f/fit-Nyquist-part.eps" % (Js[i]/scale)
    fig.savefig(filename)
