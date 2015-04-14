# -*- coding: utf-8 -*-
# constant flow velocity

import transientlib as tr
import tables
import stationarylib as st
import parameters as p

default_channels = 100

exper = {"gas_O"     : p.gas_O,
         "T"         : p.T,
         "R"         : p.R,
         "F"         : p.F,
         "c_ref"     : p.c_ref}

param = {"h"     : p.h,
         "l_d"   : p.l_d,
         "sigma" : p.sigma,
         "l_t"   : p.l_t,
         "l_m"   : p.l_m}

area = 5.0
scale = 10000.0
Is = [0.1]#, 0.2, 0.3, 0.4, 0.5]*3
Js = [j/area*scale for j in Is]
flows = [10.0]*5 + [15.0]*5 + [200.0]*5
lams = [p.stoich(Js[i], flows[i], area, scale) for i in xrange(0, len(Is))]

for i in xrange(0, len(Js)):
    exper.update({"forJ"      : Js[i],
                  "jfix"      : Js[i]})
    fuel_cell = st.FCSimple(param, exper)
    impedance = tr.ImpCCLFastO2(fuel_cell, default_channels)
    filename = "results/I%.2f/new-transient-fitting-lambda%.2f.h5" % (Js[i]/scale, lams[i])
    r_calc = impedance.read_experiment("data/%.1fA-%dml.dat" % (Is[i], int(flows[i])), area, scale)
    impedance.freq_v = impedance.exp_freq_v
    rest = {"j_0": p.j_0, "R_Ohm": r_calc}
    fit = {"b": p.b, "sigma_t": p.sigma_t, "Cdl": p.Cdl, "lambda_eff": lams[i], "D_O_GDL": p.D_O_GDL}
    result = tr.leastsqFC(impedance.residuals, fit, rest)
    print result[0]

    # Dump results to HDF5:    
    found = impedance.param_parse(result[0], rest)
    found_Z = impedance.model_fit(found)
    impedance.fc.find_opt()
    print "found alpha =", impedance.fc.express["alpha"]
    print tr.error(impedance.exp_Z_v, impedance.Z_v)
    print impedance.fc.fit
    with tables.open_file(filename, "w") as r:
        impedance.dump_results(r, prefix="found_")

    # Dump initials to HDF5:
    ini = fit.copy()
    ini.update(rest)
    ini_Z = impedance.model_fit(ini)
    impedance.fc.find_opt()
    print "initial alpha =", impedance.fc.express["alpha"]
    print tr.error(impedance.exp_Z_v, impedance.Z_v)
    print impedance.fc.fit
    with tables.open_file(filename, "a") as r:
        impedance.dump_results(r, prefix="initial_")
