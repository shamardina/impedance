#!/usr/bin/python
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

default_channels = 100
default_keys = ["b", "j_0", "R_Ohm", "sigma_t", "D_O_GDL", "Cdl", "lam_eff"]
keys_K2015   = ["b", "j_0", "R_Ohm", "sigma_t", "D_O_GDL", "D_O_CCL", "Cdl"]


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
        self.fc.find_params()
        self.stationary = None
        self.channels = channels
        self.n_z_mesh = [1.0*z/channels for z in xrange(0, channels+1)]
        self.Z_v = []
        self.freq_v = []
        self.Omega_v = []
        self.exp_Z_v = []
        self.exp_freq_v = []

    def _Ztotal_all(self):
        nJ = self.fc.exper["forJ"]/self.fc.express["j_ref"]
        n_Ztotal_v = [self._n_Ztotal(Omega, nJ) for Omega in self.Omega_v]
        self.Z_v = [self.fc.fit["R_Ohm"]+n_z*self.fc.param["l_t"]/self.fc.fit["sigma_t"] for n_z in n_Ztotal_v]
        return self.Z_v

    def _n_Ztotal(self, Omega, nJ):
        raise NotImplementedError("Should be implemented in child class.")

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
        self.fc.find_vars()

    def param_parse(self, fit, rest, keys):
        """The order of parameters: b, j_0, R_Ohm, sigma_t, D_O_GDL, D_O_CCL, Cdl, lam_eff.
        fit and rest are dictionaries, returns a list of initial values in correct order."""
        if rest is None: rest = {}
        kk = fit.keys()
        kk.extend(rest.keys())
        if sorted(kk) != sorted(keys):
            print sorted(kk)
            print sorted(keys)
            raise ValueError("Inconsistent performance parameters")
        initial = []
        for k in keys:
            if k in fit:
                initial.append(fit[k])
        return initial

    def fit_parse(self, init, rest, keys):
        """The order of parameters: b, j_0, R_Ohm, sigma_t, D_O_GDL, D_O_CCL, Cdl, lam_eff.
        init is a list, rest is a dictionary, returns a full dictionary of performance parameters."""
        if rest is None: rest = {}
        if len(init)+len(rest) != len(keys):
            raise ValueError("Inconsistent parameters number in performance results: %s + %s is not %s" % (len(init), len(rest), len(keys)))
        arr_init = list(init)
        rc = {}
        rc.update(rest)
        for k in keys:
            if k not in rc:
                rc[k] = arr_init.pop(0)
        return rc

    def residuals(self, init, rest, keys):
        pp = self.fit_parse(init, rest, keys)
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

    def leastsqFC(self, func, fit, rest=None, keys=default_keys):
        """leastsq-wrapper. Arguments: residuals function, fitting
        parameters (dict), other parameters (dict), keys (list) of the performance parameters dictionary.
        The order of parameters: b, j_0, R_Ohm, sigma_t, D_O_GDL, D_O_CCL, Cdl, lam_eff."""
        init = self.param_parse(fit, rest, keys)
        # return leastsq(func, init, rest, full_output=True)
        return leastsq(func, init, args=(rest, keys), full_output=False, xtol=1.49012e-16, ftol=1.49012e-16)

    def _n_Zccl(self, j0n, sqrtphin):
        """Returns the dimensionless local CCL impedance"""
        # found in maxima:
        return -1.0/cmath.tan(sqrtphin)/sqrtphin


class ImpCCLFastO2(Impedance):
    """Impedance class descendant for EIS with account of CCL with fast oxygen transport. Small perturbations approach."""

    def __init__(self, fc, channels=default_channels, num_method=0):
        """0 - finite-difference, 1 - Runge--Kutta"""
        Impedance.__init__(self, fc, channels)
        if num_method == 1:
            self.stationary = st.Tafel(fc, channels*2)
        else:
            self.stationary = st.Tafel(fc, channels)
        self.num_method = num_method
        self.y_v = []
        self.Zloc_v = []

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
        j0, c10 = self.stationary_values()
        n_Ztotal_v = [self._n_Ztotal(Omega, j0, c10) for Omega in self.Omega_v]
        self.Z_v = [self.fc.fit["R_Ohm"]+n_z*self.fc.param["l_t"]/self.fc.fit["sigma_t"] for n_z in n_Ztotal_v]
        return self.Z_v

    def y_all(self, Omegas, fit=None):
        if fit:
            self.update_param(fit)
        else:
            self.update_param(self.fc.fit)
        j0, c10 = self.stationary_values()
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
        j0, c10 = self.stationary_values()
        rc = {x: [] for x in nodes}
        for om in Omegas:
            Z = self._n_Ztotal(om, j0, c10)
            for i in nodes:
                rc[i].append(self.Zloc_v[i]*self.fc.param["l_t"]/self.fc.fit["sigma_t"])
        return rc

    def stationary_values(self):
        print len(self.stationary.n_z_mesh)
        eta = self.stationary.eta()
        if eta is None: return None
        j0 = [self.stationary.j_profile(nz)/self.fc.express["j_ref"] for nz in self.stationary.n_z_mesh]
        c10 = [self.stationary.c_t_profile(nz)/self.fc.exper["c_ref"] for nz in self.stationary.n_z_mesh]
        return j0, c10

    def dump_results(self, h5, prefix=""):
        self.stationary.dump_results(h5, prefix)
        Impedance.dump_results(self, h5, prefix)


#class ImpInfLambda(ImpCCLFastO2):
class ImpInfLambda(Impedance):
    """Impedance class descendant for EIS with account of CCL with fast oxygen transport in case of infinite lambda."""

    def _n_Ztotal(self, Omega, nJ):
        return self._n_Zccl(nJ, cmath.sqrt(-nJ - 1j*Omega)) + self._n_Zgdl(Omega, nJ, self.fc.express)

    def _Ztotal_all(self):
       nJ = self.fc.exper["forJ"]/self.fc.express["j_ref"]
       n_Ztotal_v = [self._n_Ztotal(Omega, nJ) for Omega in self.Omega_v]
       self.Z_v = [self.fc.fit["R_Ohm"]+n_z*self.fc.param["l_t"]/self.fc.fit["sigma_t"] for n_z in n_Ztotal_v]
       return self.Z_v

    def _n_Zgdl(self, Omega, nJ, params):
        """Returns the dimensionless GDL impedance"""
        psi = cmath.sqrt(-1j*Omega/params["nD_d"])
        return nJ*cmath.tan(params["mu"]*params["nl_d"]*psi)/(params["mu"]*psi*(params["nD_d"] - nJ*params["nl_d"])*(nJ + 1j*Omega))

    def nZgdl_all(self, nJ=None, params=None):
        if params is None:
            self.fc.find_vars()
            self.fc.find_params()
            params =  self.fc.express.copy()
        if nJ is None:
            nJ = self.fc.exper["forJ"]/self.fc.express["j_ref"]
        return [self._n_Zgdl(Omega, nJ, params) for Omega in self.Omega_v]


class ImpInfLambdaWLike(ImpInfLambda):
    """Impedance class ImpInfLambda descendant for Warburg-like impedance of a GDL."""

    def _n_Zgdl(self, Omega, nJ, params):
        """Returns the dimensionless Warburg-like GDL impedance"""
        psi = cmath.sqrt(-1j*Omega/params["nD_d"])
        return cmath.tan(params["mu"]*params["nl_d"]*psi)/(params["mu"]*psi*(params["nD_d"] - nJ*params["nl_d"]))


class ImpInfLambdaW(ImpInfLambda):
    """Impedance class ImpInfLambda descendant for Warburg impedance of a GDL."""

    def _n_Zgdl(self, Omega, nJ, params):
        """Returns the dimensionless Warburg GDL impedance"""
        return params["Rm"]*cmath.tanh(params["mu"]*params["nl_d"]*cmath.sqrt(1j*Omega/params['nD_d']))/cmath.sqrt(1j*Omega*params['nD_d'])
#        return params["Rm"]*cmath.tan(params["mu"]*params["nl_d"]*cmath.sqrt(-1j*Omega/params['nD_d']))/cmath.sqrt(-1j*Omega*params['nD_d'])/params["mu"]


class ImpGDLCh(ImpCCLFastO2):
    """Impedance class ImpCCLFastO2 descendant for the impedance of GDL and channel only."""

    def _n_Zccl(self, j0n, sqrtphin):
        """Returns the dimensionless local CCL impedance (DC limit)"""
        return 1.0/3.0 + 1.0/j0n


class ImpInfLambdaK2015(Impedance):
    """Impedance from A. A. Kulikovsky, Journal of the Electrochemical Society 162, F217 (2015). The implementation literally reproduces the paper."""
    @staticmethod
    def _p(Omega, c10, j0, Dt, mu2):
        return (2.0*1j*Omega*c10*j0*(Dt*c10 - 1.0)*(Dt - mu2)
                - Omega**2 * c10**2 * (Dt - mu2)**2
                + j0**2 * (1.0 - Dt*c10)**2)

    @staticmethod
    def _t(Omega, c10, j0, Dt, mu2):
        return (-Omega**2 * c10**2 * (mu2**2 + Dt**2)
                + 2.0*Dt*Omega*c10**2 * (Omega*mu2 - 1j*j0*mu2 + 1j*Dt*j0)
                + 2.0*1j*Omega*c10*j0*(mu2 - Dt)
                + j0**2 * (1 + Dt*c10)**2)

    @staticmethod
    def _ym(Omega, c10, j0, Dt, mu2, loc_t):
        return Dt*c10*(1j*Omega*c10*mu2 + 1j*Dt*Omega*c10
                       + c10*j0*Dt + j0 - cmath.sqrt(loc_t))

    @staticmethod
    def _yp(Omega, c10, j0, Dt, mu2, loc_t):
        return Dt*c10*(1j*Omega*c10*mu2 + 1j*Dt*Omega*c10
                       + c10*j0*Dt + j0 + cmath.sqrt(loc_t))

    @staticmethod
    def _gsp(Omega, c10, c11, j0, Dt, mu2, loc_t, loc_yp):
        return cmath.sqrt(2.0*loc_yp)*c11*(j0**2 * (Dt*c10 + 1.0)**2 -
                                           Omega**2 * c10**2 * (Dt - mu2)**2 -
                                           1j*Omega*c10*(Dt - mu2)*(cmath.sqrt(loc_t) + 2.0*j0 - 2.0*Dt*c10*j0) -
                                           cmath.sqrt(loc_t)*j0*(Dt*c10 - 1.0))

    @staticmethod
    def _gsm(Omega, c10, c11, j0, Dt, mu2, loc_t, loc_ym):
        return cmath.sqrt(2.0*loc_ym)*c11*(j0**2 * (Dt*c10 + 1.0)**2 -
                                           Omega**2 * c10**2 * (Dt - mu2)**2 +
                                           1j*Omega*c10*(Dt - mu2)*(cmath.sqrt(loc_t) - 2.0*j0 + 2.0*Dt*c10*j0) +
                                           cmath.sqrt(loc_t)*j0*(Dt*c10 - 1.0))

    @staticmethod
    def _gcp(Omega, c10, j0, Dt, mu2, loc_t):
        return -2.0*cmath.sqrt(loc_t)*c10*(1j*Omega*c10*(mu2 - Dt) +
                                           j0*(1.0 - Dt*c10) +
                                           cmath.sqrt(loc_t))

    @staticmethod
    def _gcm(Omega, c10, j0, Dt, mu2, loc_t):
        return 2.0*cmath.sqrt(loc_t)*c10*(1j*Omega*c10*(mu2 - Dt) +
                                           j0*(1.0 - Dt*c10) -
                                           cmath.sqrt(loc_t))

    @staticmethod
    def _N11(Omega, c10, j0, Dt, loc_t, loc_ym, loc_yp, loc_gcm, loc_gcp, loc_gsm, loc_gsp):
        twoDtc10 = 2.0*Dt*c10
        return (2.0*cmath.sqrt(2.0*loc_t)*c10*j0*(cmath.sqrt(loc_ym)*cmath.sinh(cmath.sqrt(2.0*loc_ym)/twoDtc10) -
                                                  cmath.sqrt(loc_yp)*cmath.sinh(cmath.sqrt(2.0*loc_yp)/twoDtc10))/
                (loc_gcm*cmath.cosh(cmath.sqrt(2.0*loc_ym)/twoDtc10) +
                 loc_gcp*cmath.cosh(cmath.sqrt(2.0*loc_yp)/twoDtc10) +
                 loc_gsm*cmath.sinh(cmath.sqrt(2.0*loc_ym)/twoDtc10) +
                 loc_gsp*cmath.sinh(cmath.sqrt(2.0*loc_yp)/twoDtc10)))

    @staticmethod
    def _bcs(Omega, c10, c11, j0, N11, Dt, mu2, loc_t, loc_ym, loc_yp):
        return 2.0*j0*loc_ym*cmath.sqrt(loc_yp)*((1j*Omega*c10*(mu2 - Dt) +
                                                 j0*(1.0 - Dt*c10))*cmath.sqrt(loc_t) + loc_t -
                                                2.0*cmath.sqrt(loc_t)*Dt*c11*j0*N11)

    @staticmethod
    def _bsc(Omega, c10, c11, j0, N11, Dt, mu2, loc_t, loc_ym, loc_yp):
        return -2.0*j0*loc_yp*cmath.sqrt(loc_ym)*((1j*Omega*c10*(mu2 - Dt) +
                                                 j0*(1.0 - Dt*c10))*cmath.sqrt(loc_t) - loc_t -
                                                2.0*cmath.sqrt(loc_t)*Dt*c11*j0*N11)

    @staticmethod
    def _q(Omega, c10, c11, j0, N11, Dt, mu2, loc_t, loc_p, loc_ym, loc_yp):
        return -2.0*Dt*cmath.sqrt(2.0*loc_yp*loc_ym)*((loc_p - loc_t)*c10*j0 +
                                                     (2.0*Dt*c10*j0**2 +
                                                      loc_p - loc_t)*(j0*(1.0 - Dt*c10) +
                                                              1j*Omega*c10*(mu2 - Dt))*c11*N11)

    @staticmethod
    def _alss(Omega, c10, c11, j0, N11, Dt, mu2, loc_t, loc_p, loc_ym, loc_yp):
        return math.sqrt(2.0)*Dt*c10*j0*(-(loc_ym + loc_yp)*(loc_p - loc_t) +
                                         2.0*((1j*Omega*c10*mu2 - j0*Dt*c10 -
                                               1j*Omega*Dt*c10 + j0)*(loc_ym + loc_yp) -
                                              (loc_ym - loc_yp)*cmath.sqrt(loc_t))*N11*Dt*c11*j0)

    @staticmethod
    def _alcc(Omega, c10, c11, j0, N11, Dt, mu2, loc_t, loc_p, loc_ym, loc_yp):
        return 2.0*cmath.sqrt(2.0*loc_yp*loc_ym)*Dt*c10*j0*(loc_p + loc_t -
                                                            2.0*1j*(1j*Dt*c10*j0 +
                                                                    Omega*c10*mu2 -
                                                                    Dt*Omega*c10 -
                                                                    1j*j0)*N11*Dt*c11*j0)
    @staticmethod
    def _c11(Omega, Dd, mu, ld):
        # N11 = 1.0
        return (-(1.0 + 1j)/(mu*math.sqrt(2.0*Dd*Omega))*
                cmath.tan((1.0 - 1j)*mu*ld*math.sqrt(Omega)/math.sqrt(2.0*Dd)))

    @staticmethod
    def _n_Z(Omega, c10, Dt, loc_q, loc_ym, loc_yp, loc_alcc, loc_alss, loc_bsc, loc_bcs):
        arg1 = cmath.sqrt(2.0*loc_yp)/2.0/Dt/c10
        arg2 = cmath.sqrt(2.0*loc_ym)/2.0/Dt/c10
        return ((loc_alcc*cmath.cosh(arg1)*cmath.cosh(arg2) +
                 loc_alss*cmath.sinh(arg1)*cmath.sinh(arg2) + loc_q)/
                (loc_bsc*cmath.sinh(arg1)*cmath.cosh(arg2) +
                 loc_bcs*cmath.cosh(arg1)*cmath.sinh(arg2)))

    def _n_Ztotal(self, Omega, j0, c10):
        Dt = self.fc.express["nD_t"]
        Dd = self.fc.express["nD_d"]
        mu = self.fc.express["mu"]
        mu2 = self.fc.express["mu2"]
        ld = self.fc.express["nl_d"]
        loc_p = self._p(Omega, c10, j0, Dt, mu2)
        loc_t = self._t(Omega, c10, j0, Dt, mu2)
        loc_ym = self._ym(Omega, c10, j0, Dt, mu2, loc_t)
        loc_yp = self._yp(Omega, c10, j0, Dt, mu2, loc_t)
        c11 = self._c11(Omega, Dd, mu, ld)
        loc_gsp = self._gsp(Omega, c10, c11, j0, Dt, mu2, loc_t, loc_yp)
        loc_gsm = self._gsm(Omega, c10, c11, j0, Dt, mu2, loc_t, loc_ym)
        loc_gcp = self._gcp(Omega, c10, j0, Dt, mu2, loc_t)
        loc_gcm = self._gcm(Omega, c10, j0, Dt, mu2, loc_t)
        N11 = self._N11(Omega, c10, j0, Dt, loc_t, loc_ym, loc_yp, loc_gcm, loc_gcp, loc_gsm, loc_gsp)
        loc_bcs = self._bcs(Omega, c10, c11, j0, N11, Dt, mu2, loc_t, loc_ym, loc_yp)
        loc_bsc = self._bsc(Omega, c10, c11, j0, N11, Dt, mu2, loc_t, loc_ym, loc_yp)
        loc_q = self._q(Omega, c10, c11, j0, N11, Dt, mu2, loc_t, loc_p, loc_ym, loc_yp)
        loc_alss = self._alss(Omega, c10, c11, j0, N11, Dt, mu2, loc_t, loc_p, loc_ym, loc_yp)
        loc_alcc = self._alcc(Omega, c10, c11, j0, N11, Dt, mu2, loc_t, loc_p, loc_ym, loc_yp)
        return self._n_Z(Omega, c10, Dt, loc_q, loc_ym, loc_yp, loc_alcc, loc_alss, loc_bsc, loc_bcs)

    def _Ztotal_all(self):
        j0 = self.fc.exper["forJ"]/self.fc.express["j_ref"]
        c10 = 1.0 - self.fc.exper["forJ"]/self.fc.express["j_lim"]
        n_Ztotal_v = [self._n_Ztotal(Omega, j0, c10) for Omega in self.Omega_v]
        self.Z_v = [self.fc.fit["R_Ohm"]+n_z*self.fc.param["l_t"]/self.fc.fit["sigma_t"] for n_z in n_Ztotal_v]
        return self.Z_v


def main():
    scale = 10000.0
    import parameters as p
    exper = {"gas_O"   : p.gas_O,
             "T"       : p.T,
             "R"       : p.R,
             "F"       : p.F,
             "c_ref"   : p.c_ref,
             "forJ"    : 600.0,
             "lam_exp" : 3.0}

    param = {"h"     : p.h,
             "l_d"   : p.l_d,
             "sigma" : p.sigma,
             "l_t"   : p.l_t,
             "l_m"   : p.l_m}
    fuel_cell = st.FCSimple(param, exper)
    fuel_cell.fit = {"j_0"     : p.j_0,
                     "R_Ohm"   : 0.000008,
                     "b"       : p.b,
                     "sigma_t" : p.sigma_t,
                     "Cdl"     : p.Cdl,
                     "lam_eff" : exper["lam_exp"],
                     "D_O_GDL" : p.D_O_GDL}
    Omega_v = []
    om = 1.0e-5
    while om < 10.0:
        om = 1.12*om
        Omega_v.append(om)
    impedance = ImpCCLFastO2(fuel_cell, default_channels, 0)
    impedance.Omega_v = Omega_v
    found_Z = impedance.model_calc(impedance.fc.fit)
    impedance.freq_v = impedance.f_from_Omega(impedance.Omega_v)
    filename = "results/I%.2f/transient-lambda%.2f.h5" % (impedance.fc.exper["forJ"]/scale, impedance.fc.exper["lam_exp"])
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
    fig.savefig("results/I%.2f/Nyquist%d.eps" % (impedance.fc.exper["forJ"]/scale, impedance.fc.exper["lam_exp"]))

if __name__ == "__main__":
    main()
