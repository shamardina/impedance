#!/usr/bin/python
# -*- coding: utf-8 -*-
# constant flow velocity

import transientlib as tr
import tables
import stationarylib as st
import parameters as p
import argparse

parser = argparse.ArgumentParser(description='Fit impedance data with various models.')
parser.add_argument("--main", dest="fit_mode", action="append_const", const="d",
                    help="Main model with finite difference solver")
parser.add_argument("--RK", dest="fit_mode", action="append_const", const="rk",
                    help="Main model with Runge-Kutta solver")
parser.add_argument("--IL", dest="fit_mode", action="append_const", const="il",
                    help="Main model in the limit of infinite oxygen stoichiometry")
parser.add_argument("--W", dest="fit_mode", action="append_const", const="wl",
                    help="Main model in Warburg limit")
parser.add_argument("--K2015", dest="fit_mode", action="append_const", const="k",
                    help="Kulikovsky 2015 model with the account for D_t")

#fit_mode_list = ["d", "rk", "k", "il", "wl"]
fit_mode_list = parser.parse_args().fit_mode
if fit_mode_list is None:
    parser.exit(0, "No model selected. Nothing to do. Use -h for help.\n")

channelsD     = 1000
channelsRK    = 2000

dropRK = 24

filename_base = "transient-fitting-lambda"
filename_fit_mode = {"d": ("%s-" % channelsD),
                     "rk": "RK-",
                     "k": "K2015-",
                     "il": "il-",
                     "wl": "W-"
                     }

keys_our     = ["b", "j_0", "R_Ohm", "sigma_t", "D_O_GDL", "Cdl", "lam_eff"]
keys_K2015   = ["b", "j_0", "R_Ohm", "sigma_t", "D_O_GDL", "D_O_CCL", "Cdl"]
keys_inf     = ["b", "j_0", "R_Ohm", "sigma_t", "D_O_GDL", "Cdl"]

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
Is = [0.1, 0.2, 0.3]*3
Js = [j/area*scale for j in Is]
flows = [10.0]*3 + [15.0]*3 + [200.0]*3
lams = [p.stoich(Js[i], flows[i], area, scale) for i in xrange(0, len(Is))]

def set_defaults(fit_mode, param, exper):
    if fit_mode == "d":
        fuel_cell = st.FCSimple(param, exper)
        return tr.ImpCCLFastO2(fuel_cell, channelsD, num_method=0), keys_our
    elif fit_mode == "rk":
        fuel_cell = st.FCSimple(param, exper)
        return tr.ImpCCLFastO2(fuel_cell, channelsRK, num_method=1), keys_our
    elif fit_mode == "k":
        fuel_cell = st.FCCCL(param, exper)
        fuel_cell.fit["lam_eff"] = p.eff_inf
        return tr.ImpInfLambdaK2015(fuel_cell), keys_K2015
    elif fit_mode == "il":
        fuel_cell = st.FCSimple(param, exper)
        fuel_cell.fit["lam_eff"] = p.eff_inf
        return tr.ImpInfLambda(fuel_cell), keys_inf
    elif fit_mode == "wl":
        fuel_cell = st.FCSimple(param, exper)
        fuel_cell.fit["lam_eff"] = p.eff_inf
        return tr.ImpInfLambdaWLike(fuel_cell), keys_inf

for fit_mode in fit_mode_list:
    for i in xrange(0, len(Js)):
        exper.update({"forJ": Js[i],
                      "lam_exp": lams[i]})
        impedance, keys = set_defaults(fit_mode, param, exper)
        if fit_mode == "rk":
            r_calc = impedance.read_experiment("data/%.1fA-%dml.dat" % (Is[i], int(flows[i])), area, scale, dropRK)
        else:
            r_calc = impedance.read_experiment("data/%.1fA-%dml.dat" % (Is[i], int(flows[i])), area, scale)
        impedance.freq_v = impedance.exp_freq_v
        rest = {"j_0": p.j_0, "R_Ohm": r_calc}
        if fit_mode in ["d", "rk"]:
            fit = {"b": p.b, "Cdl": p.Cdl, "D_O_GDL": p.D_O_GDL, "sigma_t": p.sigma_t, "lam_eff": lams[i]}
        elif fit_mode == "k":
            fit = {"b": p.b, "Cdl": p.Cdl, "D_O_GDL": p.D_O_GDL, "sigma_t": p.sigma_t, "D_O_CCL": p.D_O_CCL}
        elif fit_mode in ["il", "wl"]:
            fit = {"b": p.b, "Cdl": p.Cdl, "D_O_GDL": p.D_O_GDL, "sigma_t": p.sigma_t}
        result = impedance.leastsqFC(impedance.residuals, fit, rest, keys)
        print result[0]

        # Dump results to HDF5:
        filename = ("results/I%.2f/" + filename_fit_mode[fit_mode] + filename_base + "%.2f.h5") % (Js[i]/scale, lams[i])
        found = impedance.fit_parse(result[0], rest, keys)
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
