# -*- coding: utf-8 -*-
# constant flow velocity

import numpy as np
import sys
from scipy.optimize import leastsq
import math
import cmath
import tables
import stationarylib as st

default_channels = 100

def leastsqFC(func, fit, rest={}):
    """leastsq-wrapper. Arguments: residuals function, fitting
    parameters (dict), other parameters (dict). The order of
    parameters: b, j_0, R_Ohm, sigma_t, D_O_GDL, Cdl, lambda_eff."""
    kk = fit.keys()
    kk.extend(rest.keys())
    keys = ['b', 'j_0', 'R_Ohm', 'sigma_t', 'D_O_GDL', 'Cdl', 'lambda_eff']
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


class Impedance(object):
    """Class for EIS simulations"""
    
    def __init__(self, fc, channels=default_channels):
        self.fc = fc
        self.stationary = None
        self.find_mesh(channels)
        self.Z_v = []
        self.freq_v = []
        self.exp_Z_v = []
        self.exp_freq_v = []

    def Ztotal_all(self):
        raise RuntimeError("should be implemented in child class")

    def find_mesh(self, channels=default_channels):
        self.channels = channels
        self.nodes = channels + 1
        self.z_step = 1.0/channels
        self.n_z_mesh = [1.0*z/channels for z in xrange(0, channels+1)]

    def read_results(self, h5, prefix=""):
        # TODO: check it! compare to stationarylib
        method = self.__class__.__name__
        results = "%s_results" % (prefix+method)
        res = h5.get_node("/%s" % results)
        self.freq_v = res.freq_v
        Z1 = res.Z_1_v
        Z2 = res.Z_2_v
        self.Z_v = [Z1[i] - 1j*Z2[i] for i in xrange(0, len(Z1))]
        self.find_mesh(res._v_attrs.channels)

    def dump_results(self, h5, prefix=""):
        if self.stationary.eta_v is None:
            return None
        self.stationary.dump_results(h5, prefix)
        self.fc.dump_vars(h5, prefix)
        method = self.__class__.__name__
        results = "%s_results" % (prefix+method)
        gr = h5.create_group("/", results, "Impedance %s results" % method)
        double_atom = tables.Float64Atom()
        gr._v_attrs.channels = self.channels
        gr._v_attrs.current = self.fc.exper['forJ']
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
            factor = self.fc.fit['sigma_t']/self.fc.param['l_t']
        return [z*factor for z in Z]

    def normJZ(self, Z, factor=None):
        if factor is None:
            factor = self.fc.exper['forJ']/self.fc.fit['b']
        return self.normZ(Z, factor)

    def read_experiment(self, fn, area, scale=10000.0):
        """Returns intersection."""
        lines = [line.strip('\n').split("\t") for line in list(open(fn))[1:]]
        expZ1 = [float(line[2])*area/scale for line in lines]
        expZ2 = [float(line[3])*area/scale for line in lines]
        self.exp_freq_v = [float(line[1]) for line in lines]
        # TODO: replace with a loop
        self.exp_Z_v = [(expZ1[i] - 1j*expZ2[i])
                        for i in xrange(0, len(self.exp_freq_v)) if expZ2[i] > 0]
        num = len(self.exp_freq_v) - len(self.exp_Z_v)
        self.exp_freq_v = self.exp_freq_v[num:]
        return (expZ1[num-1] + (expZ1[num] - expZ1[num-1])/(1.0 - expZ2[num]/expZ2[num-1]))

    def f_from_Omega(self, Omega_v):
        return [om*self.fc.fit['sigma_t']/2.0/np.pi/self.fc.fit['Cdl']/self.fc.param['l_t']**2 for om in Omega_v]

    def Omega_from_f(self, freq_v):
        return [2.0*np.pi*f*self.fc.fit['Cdl']*self.fc.param['l_t']**2/self.fc.fit['sigma_t'] for f in freq_v]

    def dump_results_dat(self, dat):
        console_stdout = sys.stdout
        sys.stdout = open(dat, "w", 0)
        Omega = self.Omega_v
        Z = self.Z_v
        nZ = self.normZ(Z)
        nJnZ = self.normJZ(Z)
        print "#" + "\t".join('Omega', 'nZ1*nJ', 'nZ2*nJ', 'f', 'Z1', 'Z2', 'nZ1', 'nZ2')
        for i in xrange(0, len(self.Z_v)):
            print "\t".join("%.6f" % x for x in (Omega[i], nJnZ[i].real, -nJnZ[i].imag, self.freq_v[i], Z[i].real, -Z[i].imag, nZ[i].real, -nZ[i].imag))
        sys.stdout.close()
        sys.stdout = console_stdout

    def model_fit(self, new_fit):
        self.fc.fit.update(new_fit)
        self.fc.exper['lambdafix'] = new_fit['lambda_eff']/self.fc.exper['jfix']*self.fc.exper['forJ']
        self.fc.find_vars()
        self.fc.find_params()
        self.Omega_v = self.Omega_from_f(self.freq_v)
        return self.Ztotal_all()

    def param_parse(self, init, rest={}):
        """The order of parameters: b, j_0, R_Ohm, sigma_t, D_O_GDL, Cdl, lambda_eff."""
        arr_init = list(init)
        keys = ['b', 'j_0', 'R_Ohm', 'sigma_t', 'D_O_GDL', 'Cdl', 'lambda_eff']
        rc = {}
        rc.update(rest)
        for k in keys:
            if k not in rc:
                rc[k] = arr_init.pop(0)
        return rc
    
    def residuals(self, init, rest={}):
        pp = self.param_parse(init, rest)
        ll = len(self.exp_freq_v)
        for p in pp.values():
            if p <= 0.0 or math.isnan(p):
                return [10.0**10]*ll
        try:
            fit_v = self.model_fit(pp)
        except:
            print sys.exc_info()[0]
            return [10.0**10]*ll
        if fit_v is None:
            return [10.0**10]*ll
        # TODO: weight (real and imaginary parts individually)? diff = [(self.Z_v[i] - self.exp_Z_v[i])/self.exp_Z_v[i] for i in xrange(0, ll)] ???
        diff = [(self.Z_v[i] - self.exp_Z_v[i]) for i in xrange(0, ll)]
        z1d = [0]*2*ll
        z1d[0::2] = [z.real for z in diff]
        z1d[1::2] = [z.imag for z in diff]
        return z1d


class ImpCCLFastO2(Impedance):
    """Impedance class descendant for EIS with account of CCL with fast oxygen transport. Small perturbations approach."""
    
    def __init__(self, fc, channels=default_channels, Omega_v=[]):
        Impedance.__init__(self, fc, channels)
        self.stationary = st.Tafel(fc, channels)
        self.Omega_v = Omega_v

    def n_Ztotal(self, Omega, j0, c10):
        fc = self.fc
        z_step = self.z_step
        phi = [-j - 1j*Omega for j in j0]
        psi = cmath.sqrt(-1j*Omega/fc.express['nD_d'])
        nlJfix = fc.express['lJfix']/fc.express['j_ref']
        lbmupsi = self.fc.express['nl_d']*self.fc.express['mu']*psi
        sqrtphi = [cmath.sqrt(phii) for phii in phi]
        y_v = [0.0]*self.nodes
        c11peta = [0.0]*self.nodes
        # TODO: upload maxima script
        # found in maxima:
        c11peta[0] = (cmath.sin(sqrtphi[0])*sqrtphi[0]*c10[0]*phi[0]*cmath.sin(lbmupsi))/(cmath.cos(lbmupsi)*(j0[0]*cmath.sin(sqrtphi[0])*sqrtphi[0]*cmath.tan(lbmupsi)+c10[0]*fc.express['nD_d']*fc.express['mu']*phi[0]*psi))
        Zloc = [0.0]*self.nodes
        # found in maxima:
        Zloc[0] = -1.0/(cmath.sin(sqrtphi[0])*sqrtphi[0]*((c10[0]*phi[0])/(c11peta[0]*j0[0])-1.0)) - cmath.cos(sqrtphi[0])/(cmath.sin(sqrtphi[0])*sqrtphi[0])
        invZtotal = 1.0/Zloc[0]
        for i in xrange(1, self.nodes):
            # all expressions below found in maxima:
            ABcommon = fc.express['nD_d']*fc.express['mu']*cmath.sin(sqrtphi[i])*sqrtphi[i]*psi/(j0[i]*cmath.sin(sqrtphi[i])*sqrtphi[i]*cmath.sin(lbmupsi)+c10[i]*fc.express['nD_d']*fc.express['mu']*phi[i]*psi*cmath.cos(lbmupsi))
            A = c10[i]*phi[i]*ABcommon
            B = -j0[i]*ABcommon/cmath.cos(lbmupsi)+fc.express['nD_d']*fc.express['mu']*psi*cmath.tan(lbmupsi)
            y_v[i] = (y_v[i-1] + A*self.z_step/nlJfix)/(1.0 + self.z_step/nlJfix*(1j*fc.express['xi2epsilon2']*Omega - B))
            c11peta[i] = y_v[i]/cmath.cos(lbmupsi) - (cmath.sin(sqrtphi[i])*sqrtphi[i]*(y_v[i]*j0[i]/cmath.cos(lbmupsi) - c10[i]*phi[i])*cmath.sin(lbmupsi))/(cmath.cos(lbmupsi)*(j0[i]*cmath.sin(sqrtphi[i])*sqrtphi[i]*cmath.tan(lbmupsi) + c10[i]*fc.express['nD_d']*fc.express['mu']*phi[i]*psi))
            Zloc[i] = -1.0/(cmath.sin(sqrtphi[i])*sqrtphi[i]*((c10[i]*phi[i])/(c11peta[i]*j0[i]) - 1.0))-cmath.cos(sqrtphi[i])/(cmath.sin(sqrtphi[i])*sqrtphi[i])
            invZtotal += 1.0/Zloc[i]
        invZtotal = invZtotal*z_step - (1.0/Zloc[0]+1.0/Zloc[-1])*z_step/2.0
        return 1.0/invZtotal
    
    def Ztotal_all(self):
        eta = self.stationary.eta()
        if eta is None:
            return None
        self.stationary.profiles(eta/self.fc.fit['b'])
        j0 = [a/self.fc.express['j_ref'] for a in self.stationary.j_v]
        c10 = [c/self.fc.exper['c_ref'] for c in self.stationary.c_t_v]
        n_Ztotal_v = [self.n_Ztotal(Omega, j0, c10) for Omega in self.Omega_v]
        self.Z_v = [self.fc.fit['R_Ohm']+n_z*self.fc.param['l_t']/self.fc.fit['sigma_t'] for n_z in n_Ztotal_v]
        return self.Z_v
