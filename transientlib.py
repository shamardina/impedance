# -*- coding: utf-8 -*-
# constant flow velocity

import numpy as np
import sys
from scipy.optimize import leastsq
import math
import cmath
import tables
import stationarylib as st
from parameters import eff_zero, eff_inf

default_channels = 1000

def leastsqFC(func, fit, rest=None):
    """leastsq-wrapper. Arguments: residuals function, fitting
    parameters (dict), other parameters (dict). The order of
    parameters: b, j_0, R_Ohm, sigma_t, D_O_GDL, Cdl, lambda_eff."""
    if rest is None: rest = {}
    kk = fit.keys()
    kk.extend(rest.keys())
    keys = ["b", "j_0", "R_Ohm", "sigma_t", "D_O_GDL", "Cdl", "lambda_eff"]
    if sorted(kk) != sorted(keys):
        print sorted(kk)
        print sorted(keys)
        raise ValueError("Inconsistent performance parameters")
    initial = []
    for k in keys:
        if k in fit:
            initial.append(fit[k])
    init = tuple(initial)
    # return leastsq(func, init, rest, full_output=True)
    return leastsq(func, init, rest, full_output=False, xtol=1.49012e-16, ftol=1.49012e-16)

def error(Z1, Z2):
    if not Z1 or not Z2:
        print "Bad impedances"
        return 0
    rdiff = [(Z1[i].real-Z2[i].real)**2 for i in xrange(0, len(Z1))]
    idiff = [(Z1[i].imag-Z2[i].imag)**2 for i in xrange(0, len(Z1))]
    return math.sqrt(np.average(rdiff)/len(rdiff)) + math.sqrt(np.average(idiff)/len(idiff))

def diffII(X, Y):
    """Returns numerical second derivative of a table function. Corresponding frequencies are to be sorted in ascending order"""
    N = len(X)
    dI = [0.0]*(N - 1)
    dII = [0.0]*(N - 2)
    eps = X[0] - X[1]
    dI[0] = (Y[0] - Y[1])/eps
    for i in xrange(1, N-1):
        eps = X[i] - X[i+1]
        dI[i] = (Y[i] - Y[i+1])/eps
        dII[i-1] = (dI[i-1] - dI[i])/eps
    return dII

def RK_step(function, nz, y, step):
    """Discrete Runge–Kutta method step for a linear function. function should return coefficients."""
    half_step = step/2.0
    coef = function(nz)
    k1 = coef[0] + coef[1]*y
    coef = function(nz + 1)
    k2 = coef[0] + coef[1]*(y + half_step*k1)
    k3 = coef[0] + coef[1]*(y + half_step*k2)
    coef = function(nz + 2)
    k4 = coef[0] + coef[1]*(y + step*k3)
    return y + step/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)


class Impedance(object):
    """Class for EIS simulations"""

    def __init__(self, fc, channels=default_channels):
        self.fc = fc
        self.stationary = None
        self.channels = channels
        self.n_z_mesh = [1.0*z/channels for z in xrange(0, channels+1)]
        self.Z_v = []
        self.freq_v = []
        self.exp_Z_v = []
        self.exp_freq_v = []

    def _Ztotal_all(self):
        raise RuntimeError("should be implemented in child class")

    def read_results(self, h5, prefix=""):
        # TODO: check it! compare to stationarylib
        method = self.__class__.__name__
        results = "%s_results" % (prefix+method)
        res = h5.get_node("/%s" % results)
        self.freq_v = res.freq_v
        Z1 = res.Z_1_v
        Z2 = res.Z_2_v
        self.Z_v = [Z1[i] - 1j*Z2[i] for i in xrange(0, len(Z1))]
        self.channels = res._v_attrs.channels
        self.n_z_mesh = [1.0*z/self.channels for z in xrange(0, self.channels + 1)]

    def dump_results(self, h5, prefix=""):
        if self.stationary.eta_v is not None:
            self.stationary.dump_results(h5, prefix)
        self.fc.dump_vars(h5, prefix)
        method = self.__class__.__name__
        results = "%s_results" % (prefix+method)
        gr = h5.create_group("/", results, "Impedance %s results" % method)
        double_atom = tables.Float64Atom()
        gr._v_attrs.channels = self.channels
        gr._v_attrs.current = self.fc.exper["forJ"]
        a = h5.create_array("/%s" % results, "freq_v", atom=double_atom, shape=(len(self.freq_v),),
                            title="Frequencies f in %s EIS." % method)
        a[:] = self.freq_v[:]
        a = h5.create_array("/%s" % results, "Z_1_v", atom=double_atom, shape=(len(self.Z_v),),
                            title="Dimentional Z1 in %s EIS." % method)
        a[:] = [Z.real for Z in self.Z_v][:]
        a = h5.create_array("/%s" % results, "Z_2_v", atom=double_atom, shape=(len(self.Z_v),),
                            title="Dimentional Z2 in %s EIS." % method)
        a[:] = [-Z.imag for Z in self.Z_v][:]
        nZ = self.normZ(self.Z_v)
        a = h5.create_array("/%s" % results, "n_Z_1_v", atom=double_atom, shape=(len(self.Z_v),),
                            title="Dimentionless Z1 in %s EIS." % method)
        a[:] = [Z.real for Z in nZ][:]
        a = h5.create_array("/%s" % results, "n_Z_2_v", atom=double_atom, shape=(len(self.Z_v),),
                            title="Dimentionless Z2 in %s EIS." % method)
        a[:] = [-Z.imag for Z in nZ][:]
        nJZ = self.normJZ(self.Z_v)
        a = h5.create_array("/%s" % results, "n_J_x_n_Z_1_v", atom=double_atom, shape=(len(self.Z_v),),
                            title="Dimentionless J*Z1 in %s EIS." % method)
        a[:] = [Z.real for Z in nJZ][:]
        a = h5.create_array("/%s" % results, "n_J_x_n_Z_2_v", atom=double_atom, shape=(len(self.Z_v),),
                            title="Dimentionless J*Z2 in %s EIS." % method)
        a[:] = [-Z.imag for Z in nJZ][:]
        a = h5.create_array("/%s" % results, "Omega_v", atom=tables.Float64Atom(), shape=(len(self.Omega_v),),
                            title="Frequencies Omega in %s EIS." % method)
        a[:] = self.Omega_v[:]

    def normZ(self, Z, factor=None):
        if factor is None:
            factor = self.fc.fit["sigma_t"]/self.fc.param["l_t"]
        return [z*factor for z in Z]

    def normJZ(self, Z, factor=None):
        if factor is None:
            factor = self.fc.exper["forJ"]/self.fc.fit["b"]
        return self.normZ(Z, factor)

    def read_experiment(self, fn, area, scale=10000.0, start=0):
        """Returns intersection."""
        lines = [line.strip("\n").split("\t") for line in list(open(fn))[1:]]
        expZ1 = [float(line[2])*area/scale for line in lines]
        expZ2 = [float(line[3])*area/scale for line in lines]
        exp_freq_v = [float(line[1]) for line in lines]
        num = 0
        for i in xrange(0, len(exp_freq_v)):
            if expZ2[i] > 0:
                num = i
                break
        i0 = max(num, start)
        self.exp_freq_v = exp_freq_v[i0:]
        self.exp_Z_v = [(expZ1[i] - 1j*expZ2[i]) for i in xrange(0, len(exp_freq_v))][i0:]
        return expZ1[num-1] + (expZ1[num] - expZ1[num-1])/(1.0 - expZ2[num]/expZ2[num-1])

    def f_from_Omega(self, Omega_v):
        return [om*self.fc.fit["sigma_t"]/2.0/np.pi/self.fc.fit["Cdl"]/self.fc.param["l_t"]**2 for om in Omega_v]

    def Omega_from_f(self, freq_v):
        return [2.0*np.pi*f*self.fc.fit["Cdl"]*self.fc.param["l_t"]**2/self.fc.fit["sigma_t"] for f in freq_v]

    def discrete_Omega(self, freq_1, freq_2, step=1.12):
        Omega_v = []
        om = self.Omega_from_f([freq_1])[0]
        while om < self.Omega_from_f([freq_2])[0]:
            Omega_v.append(om)
            om = step*om
        return Omega_v

    def dump_results_dat(self, dat):
        console_stdout = sys.stdout
        sys.stdout = open(dat, "w", 0)
        Omega = self.Omega_v
        Z = self.Z_v
        nZ = self.normZ(Z)
        nJnZ = self.normJZ(Z)
        print "#" + "\t".join("Omega", "nZ1*nJ", "nZ2*nJ", "f", "Z1", "Z2", "nZ1", "nZ2")
        for i in xrange(0, len(self.Z_v)):
            print "\t".join("%.6f" % x for x in (Omega[i], nJnZ[i].real, -nJnZ[i].imag, self.freq_v[i], Z[i].real, -Z[i].imag, nZ[i].real, -nZ[i].imag))
        sys.stdout.close()
        sys.stdout = console_stdout

    def model_fit(self, new_fit):
        self.update_param(new_fit)
        self.Omega_v = self.Omega_from_f(self.freq_v)
        return self._Ztotal_all()

    def model_calc(self, new_fit, expOm=0):
        self.update_param(new_fit)
        if expOm:
            self.Omega_v = self.discrete_Omega(self.exp_freq_v[-1], self.exp_freq_v[0])
        return self._Ztotal_all()

    def update_param(self, fit):
        self.fc.fit.update(fit)
        self.fc.exper["lambdafix"] = fit["lambda_eff"]/self.fc.exper["jfix"]*self.fc.exper["forJ"]
        self.fc.find_vars()
        self.fc.find_params()

    def param_parse(self, init, rest=None):
        """The order of parameters: b, j_0, R_Ohm, sigma_t, D_O_GDL, Cdl, lambda_eff."""
        if rest is None: rest = {}
        arr_init = list(init)
        keys = ["b", "j_0", "R_Ohm", "sigma_t", "D_O_GDL", "Cdl", "lambda_eff"]
        rc = {}
        rc.update(rest)
        for k in keys:
            if k not in rc:
                rc[k] = arr_init.pop(0)
        return rc

    def residuals(self, init, rest=None):
        pp = self.param_parse(init, rest)
        ll = len(self.exp_freq_v)
        for p in pp.values():
            if p <= 0.0 or math.isnan(p):
                return [eff_inf]*ll
        try:
            fit_v = self.model_fit(pp)
        except:
            print sys.exc_info()[0]
            return [eff_inf]*ll
        if fit_v is None:
            return [eff_inf]*ll
        # TODO: weight (real and imaginary parts individually)? diff = [(self.Z_v[i] - self.exp_Z_v[i])/self.exp_Z_v[i] for i in xrange(0, ll)] ???
        diff = [(self.Z_v[i] - self.exp_Z_v[i]) for i in xrange(0, ll)]
        z1d = [0]*2*ll
        z1d[0::2] = [z.real for z in diff]
        z1d[1::2] = [z.imag for z in diff]
        return z1d


class ImpCCLFastO2(Impedance):
    """Impedance class descendant for EIS with account of CCL with fast oxygen transport. Small perturbations approach."""

    def __init__(self, fc, channels=default_channels, Omega_v=[], num_method=0):
        """0 - finite-difference, 1 - Runge--Kutta"""
        Impedance.__init__(self, fc, channels)
        if num_method == 1:
            self.stationary = st.Tafel(fc, channels*2)
        else:
            self.stationary = st.Tafel(fc, channels)
        self.num_method = num_method
        self.Omega_v = Omega_v
        self.y_v = []
        self.Zloc_v = []

    def _n_Zccl(self, j0n=None, sqrtphin=None):
        """Returns the dimensionless local CCL impedance"""
        # found in maxima:
        return -1.0/cmath.tan(sqrtphin)/sqrtphin

    def _n_Ztotal(self, Omega, j0, c10):
        fc = self.fc
        z_step = 1.0/self.channels
        phi = [-j - 1j*Omega for j in j0]
        psi = cmath.sqrt(-1j*Omega/fc.express["nD_d"])
        nlJfix = fc.express["lJfix"]/fc.express["j_ref"]
        lbmupsi = self.fc.express["nl_d"]*self.fc.express["mu"]*psi
        sqrtphi = [cmath.sqrt(phii) for phii in phi]
        sinsqrtphi = [cmath.sin(s) for s in sqrtphi]
        sinlbmupsi = cmath.sin(lbmupsi)
        coslbmupsi = cmath.cos(lbmupsi)
        tanlbmupsi = cmath.tan(lbmupsi)
        y_v = [0.0]*(self.channels + 1)
        def function_rhs_ch(k):
            # all expressions below found in maxima:
            ABcommon = fc.express["nD_d"]*fc.express["mu"]*sinsqrtphi[k]*sqrtphi[k]*psi/(j0[k]*sinsqrtphi[k]*sqrtphi[k]*sinlbmupsi +
                    c10[k]*fc.express["nD_d"]*fc.express["mu"]*phi[k]*psi*coslbmupsi)
            A = c10[k]*phi[k]*ABcommon
            B = -j0[k]*ABcommon/coslbmupsi+fc.express["nD_d"]*fc.express["mu"]*psi*tanlbmupsi
            return (A/nlJfix, (B - 1j*fc.express["xi2epsilon2"]*Omega)/nlJfix)
        c11peta = [0.0]*(self.channels + 1)
        # TODO: upload maxima script
        # found in maxima:
        c11peta[0] = (sinsqrtphi[0]*sqrtphi[0]*c10[0]*phi[0])/(j0[0]*sinsqrtphi[0]*sqrtphi[0] +
                c10[0]*fc.express["nD_d"]*fc.express["mu"]*phi[0]*psi/tanlbmupsi)
        Zloc = [0.0]*(self.channels + 1)
        # found in maxima:
        denominator = sinsqrtphi[0]*sqrtphi[0]*((c10[0]*phi[0])/(c11peta[0]*j0[0]) - 1.0)
        if denominator == 0.0:
            denominator = eff_zero
        Zloc[0] = -1.0 / denominator + self._n_Zccl(j0[0], sqrtphi[0])
        invZtotal = 1.0/Zloc[0]
        for i in xrange(1, self.channels + 1):
            if self.num_method == 1: # Runge--Kutta method
                y_v[i] = RK_step(function_rhs_ch, (i-1)*2, y_v[i-1], z_step)
                ii = i*2
            else: # finite-difference method
                coef = function_rhs_ch(i)
                y_v[i] = (y_v[i-1] + coef[0]*z_step)/(1.0 - z_step*coef[1])
                ii = i
            # all expressions below found in maxima:
            c11peta[i] = y_v[i]/coslbmupsi - (sinsqrtphi[ii]*sqrtphi[ii]*(y_v[i]*j0[ii]/coslbmupsi -
                    c10[ii]*phi[ii]))/(j0[ii]*sinsqrtphi[ii]*sqrtphi[ii] +
                    c10[ii]*fc.express["nD_d"]*fc.express["mu"]*phi[ii]*psi/tanlbmupsi)
            denominator = sinsqrtphi[ii]*sqrtphi[ii]*((c10[ii]*phi[ii])/(c11peta[i]*j0[ii]) - 1.0)
            if denominator == 0.0:
                denominator = eff_zero
            Zloc[i] = -1.0 / denominator + self._n_Zccl(j0[ii], sqrtphi[ii])
            invZtotal += 1.0/Zloc[i]
        invZtotal = invZtotal*z_step - (1.0/Zloc[0]+1.0/Zloc[-1])*z_step/2.0
        self.y_v = y_v
        self.Zloc_v = Zloc
        return 1.0/invZtotal

    def _Ztotal_all(self):
        eta, j0, c10 = self.stationary_values()
        n_Ztotal_v = [self._n_Ztotal(Omega, j0, c10) for Omega in self.Omega_v]
        self.Z_v = [self.fc.fit["R_Ohm"]+n_z*self.fc.param["l_t"]/self.fc.fit["sigma_t"] for n_z in n_Ztotal_v]
        return self.Z_v

    def y_all(self, Omegas, fit=None):
        if fit:
            self.update_param(fit)
        else:
            self.update_param(self.fc.fit)
        eta, j0, c10 = self.stationary_values()
        rc = {}
        for om in Omegas:
            Z = self._n_Ztotal(om, j0, c10)
            rc[om] = self.y_v
        return rc

    def Z_loc(self, Omegas, nodes, fit=None):
        if fit:
            self.update_param(fit)
        else:
            self.update_param(self.fc.fit)
        eta, j0, c10 = self.stationary_values()
        rc = {x: [] for x in nodes}
        for om in Omegas:
            Z = self._n_Ztotal(om, j0, c10)
            for i in nodes:
                rc[i].append(self.Zloc_v[i]*self.fc.param["l_t"]/self.fc.fit["sigma_t"])
        return rc

    def stationary_values(self):
        print len(self.stationary.n_z_mesh)
        eta = self.stationary.eta()
        if eta is None:
            return None
        j0 = [self.stationary.j_profile(nz)/self.fc.express["j_ref"] for nz in self.stationary.n_z_mesh]
        c10 = [self.stationary.c_t_profile(nz)/self.fc.exper["c_ref"] for nz in self.stationary.n_z_mesh]
        return eta, j0, c10


class ImpInfLambda(ImpCCLFastO2):
    """Impedance class ImpCCLFastO2 descendant for EIS with account of CCL with fast oxygen transport in case of infinite lambda."""

    def _n_Ztotal(self, Omega, nJ):
        return self._n_Zccl(nJ, cmath.sqrt(-nJ - 1j*Omega)) + self._n_Zgdl(Omega, nJ, self.fc.express)

    def _Ztotal_all(self):
        nJ = self.fc.exper["forJ"]/self.fc.express["j_ref"]
        n_Ztotal_v = [self._n_Ztotal(Omega, nJ) for Omega in self.Omega_v]
        self.Z_v = [self.fc.fit["R_Ohm"]+n_z*self.fc.param["l_t"]/self.fc.fit["sigma_t"] for n_z in n_Ztotal_v]
        return self.Z_v

    def _n_Zgdl(self, Omega, nJ, params):
        """Returns the dimensionless GDL+channel impedance"""
        phi = -nJ - 1j*Omega
        sqrtphi = cmath.sqrt(phi)
        psi = cmath.sqrt(-1j*Omega/params["nD_d"])
        return -nJ*cmath.tan(params["mu"]*params["nl_d"]*cmath.tan(psi))/(params["mu"]*psi*(params["nD_d"] - nJ*params["nl_d"])*phi)

    def nZgdl_all(self, nJ=None, params=None):
        if params is None:
            self.fc.find_vars()
            self.fc.find_params()
            params =  self.fc.express.copy()
        if nJ is None:
            nJ = self.fc.exper["forJ"]/self.fc.express["j_ref"]
        return [self._n_Zgdl(Omega, nJ, params) for Omega in self.Omega_v]


class ImpGDLCh(ImpCCLFastO2):
    """Impedance class ImpCCLFastO2 descendant for the impedance of GDL and channel only."""

    def _n_Zccl(self, j0n=None, sqrtphin=None):
        """Returns the dimensionless local CCL impedance (DC limit)"""
        return 1.0/3.0 + 1.0/j0n


def main():
    scale = 10000.0
    import parameters as p
    exper = {"gas_O"     : p.gas_O,
             "T"         : p.T,
             "R"         : p.R,
             "F"         : p.F,
             "c_ref"     : p.c_ref,
             "forJ"      : 600.0,
             "jfix"      : 600.0,
             "lambdafix" : 3.0}

    param = {"h"     : p.h,
             "l_d"   : p.l_d,
             "sigma" : p.sigma,
             "l_t"   : p.l_t,
             "l_m"   : p.l_m}
    fuel_cell = st.FCSimple(param, exper)
    fuel_cell.fit = {"j_0"        : p.j_0,
                     "R_Ohm"      : 0.000008,
                     "b"          : p.b,
                     "sigma_t"    : p.sigma_t,
                     "Cdl"        : p.Cdl,
                     "lambda_eff" : exper["lambdafix"],
                     "D_O_GDL"    : p.D_O_GDL}
    Omega_v = []
    om = 1.0e-5
    while om < 10.0:
        om = 1.12*om
        Omega_v.append(om)
    impedance = ImpCCLFastO2(fuel_cell, default_channels, 0)
    impedance.Omega_v = Omega_v
    found_Z = impedance.model_calc(impedance.fc.fit)
    impedance.freq_v = impedance.f_from_Omega(impedance.Omega_v)
    filename = "results/I%.2f/transient-lambda%.2f.h5" % (impedance.fc.exper["forJ"]/scale, impedance.fc.express["lam"])
    with tables.open_file(filename, "w") as r:
        impedance.dump_results(r, "found_")

    from matplotlib import rc
    import matplotlib.pyplot as plt
    rc("font", **{"family":"sans-serif", "sans-serif":"Arial"})
    rc("text", usetex=True)
    rc("text.latex", unicode=True)
    plt.ioff()
    dim_x = (0.0, 2.0)
    dim_y = (0.0, 1.0)
    plot_width = 9
    adjust = 0.2
    plot_height = plot_width*(dim_y[1] - dim_y[0])/(dim_x[1] - dim_x[0]) + adjust
    plt.clf()
    fig = plt.figure(figsize=(plot_width, plot_height))
    fontsize = 18
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=adjust)
    label = u"$Z'$ / $\\Omega$ cm$^2$"
    ax.set_xlabel(label, fontsize=fontsize)
    label = u"$-Z''$ / $\\Omega$ cm$^2$"
    ax.set_ylabel(label, fontsize=fontsize)
    ax.plot([z.real*scale for z in impedance.Z_v], [-z.imag*scale for z in impedance.Z_v], color="red", linestyle="-", linewidth=1)
    ax.axis("scaled")
    ax.set_xlim(*dim_x)
    ax.set_ylim(*dim_y)
    fig.savefig("results/I%.2f/Nyquist%d.eps" % (impedance.fc.exper["forJ"]/scale, impedance.fc.express["lam"]))

if __name__ == "__main__":
    main()
