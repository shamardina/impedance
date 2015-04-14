# -*- coding: utf-8 -*-
# constant flow velocity

import math
import tables

default_channels = 100

class Stationary(object):
    """Class for stationary one point simulations"""

    def __init__(self, fc, channels=default_channels):
        self.fc = fc
        self.find_mesh(channels)

    def find_mesh(self, channels=default_channels):
        self.channels = channels
        self.nodes = channels + 1
        self.z_step = 1.0/channels
        self.n_z_mesh = [1.0*z/channels for z in xrange(0, channels+1)]

    def read_results(self, h5, prefix=""):
        method = self.__class__.__name__
        results = "%s_results" % (prefix+method)
        res = h5.get_node("/%s" % results)
        self.c_h_v = []
        self.j_v = []
        self.c_h_v[:] = res.c_h_v[:]
        self.j_v[:] = res.j_v[:]
        self.eta_v = res._v_attrs.eta_v
        self.find_mesh(res._v_attrs.channels)

    def dump_results(self, h5, prefix=""):
        method = self.__class__.__name__
        results = "%s_results" % (prefix+method)
        gr = h5.create_group("/", results, "Stationary %s results" % method)
        double_atom = tables.Float64Atom()
        gr._v_attrs.eta_v = self.eta_v
        gr._v_attrs.n_eta_v = self.eta_v/self.fc.fit['b']
        gr._v_attrs.channels = self.channels
        a = h5.create_array("/%s" % results, "c_h_v", atom=double_atom, shape=(len(self.c_h_v),),
                            title="Dimentional concentration in %s approximation." % method)
        a[:] = self.c_h_v[:]
        a = h5.create_array("/%s" % results, "n_c_h_v", atom=double_atom, shape=(len(self.c_h_v),),
                            title="Dimentionless concentration in %s approximation (/c_ref)." % method)
        a[:] = [c/self.fc.exper['c_ref'] for c in self.c_h_v][:]
        a = h5.create_array("/%s" % results, "j_v", atom=double_atom, shape=(len(self.j_v),),
                            title="Dimentional current density in %s approximation." % method)
        a[:] = self.j_v[:]
        a = h5.create_array("/%s" % results, "n_j_v", atom=double_atom, shape=(len(self.j_v),),
                            title="Dimentionless current density in %s approximation." % method)
        a[:] = [j/self.fc.express['j_ref'] for j in self.j_v][:]


class Tafel(Stationary):
    """Stationary model with Tafel overpotential"""
    
    def eta(self):
        fc = self.fc
        if fc.express['lam'] <= 1.0:
            print "Tafel overpotential NOT ok"
            self.eta_v = None
            return None
        print "Tafel overpotential ok"
        f_lam_J = -fc.express['lJfix']*math.log(1.0 - 1.0/fc.express['lam'])
        self.eta_v = fc.fit['b']*(math.log(f_lam_J/fc.fit['j_0']) - math.log(1.0 - f_lam_J/fc.express['j_lim']))
        return self.eta_v

    def profiles(self, n_eta):
        if n_eta is None:
            return None
        self.eta_v = n_eta*self.fc.fit['b']
        self.j_v = [-self.fc.express['lJfix']*((1.0-1.0/self.fc.express['lam'])**(n_z))*math.log(1.0-1.0/self.fc.express['lam'])
                    for n_z in self.n_z_mesh]
        self.c_h_v = [self.fc.exper['c_ref']*((1.0-1.0/self.fc.express['lam'])**(n_z))
                      for n_z in self.n_z_mesh]
        self.c_t_v = [self.c_h_v[i]-self.j_v[i]*self.fc.exper['c_ref']/self.fc.express['j_lim']
                      for i in xrange(0, self.nodes)]
        return True


class FCSimple(object):
    """A simple class for FC. Defines or calculates the FC parameters."""
    
    def __init__(self, param={}, exper={}):
        self.param = param.copy()
        self.exper = exper.copy()
        self.fit = {}
        self.express = {}
        
    def find_vars(self):
        """Basic parameters: finds extra variables from demensional ones. Call every time when performance parameters are changed."""
        p = self.exper.copy()
        p.update(self.fit)
        p.update(self.param)
        self.express.update({'j_ref'       : p['sigma_t']*p['b']/p['l_t'],
                             'nD_d'        : 4.0*p['F']*p['D_O_GDL']*p['c_ref']/p['sigma_t']/p['b'],
                             'mu2'         : 4.0*p['F']*p['c_ref']/p['Cdl']/p['b'],
                             'mu'          : math.sqrt(4.0*p['F']*p['c_ref']/p['Cdl']/p['b']),
                             'xi2'         : 8.0*p['F']*p['h']*p['c_ref']*p['j_0']/(p['Cdl']*p['sigma_t']*(p['b']**2)),
                             'epsilon2'    : p['sigma_t']*p['b']/2.0/p['j_0']/p['l_t'],
                             'xi2epsilon2' : 4.0*p['F']*p['h']*p['c_ref']/p['Cdl']/p['b']/p['l_t'],
                             'j_lim'       : 4.0*p['F']*p['D_O_GDL']*p['c_ref']/p['l_d'],
                             'nj_lim'      : 4.0*p['F']*p['D_O_GDL']*p['c_ref']*p['l_t']/p['l_d']/p['sigma_t']/p['b']})

    def find_params(self):
        """Basic parameters: finds extra parameters from initial conditions and parameters. Call once."""
        e = self.exper
        p = self.param
        self.express.update({'nl_d'  : p['l_d']/p['l_t'],
                             'lam'   : e['lambdafix']*e['jfix']/e['forJ'],
                             'lJfix' : e['lambdafix']*e['jfix']})

    def find_opt(self):
        """Finds optional properties. Call once."""
        self.express.update({'alpha' : self.exper['R']*self.exper['T']/self.fit['b']/self.exper['F'],
                             'kappa' : self.param['l_m']/(self.fit['R_Ohm']-2.0*(self.express['nl_d']*self.param['l_t'])/self.param['sigma'])})

    def find_vvars(self):
        """Dimensionless parameters: finds extra variables from given ones. Call every time when performance parameters are changed."""
        j_ref = self.express['epsilon2']*2.0*self.fit['j_0']
        self.express.update({'j_ref'       : j_ref,
                             'xi2epsilon2' : self.express['xi2']*self.express['epsilon2'],
                             'j_lim'       : self.express['nD_d']/self.express['nl_d']*j_ref,
                             'nj_lim'      : self.express['nD_d']/self.express['nl_d'],
                             'mu'          : math.sqrt(self.express['mu2'])})
        sigma_t = self.express['epsilon2']*2.0*self.fit['j_0']*self.param['l_t']/self.fit['b']
        self.fit.update({'sigma_t' : sigma_t,
                         'D_O_GDL' : self.express['nD_d']*sigma_t*self.fit['b']/4.0/self.exper['F']/self.exper['c_ref'],
                         'Cdl'     : 4.0*self.exper['F']*self.exper['c_ref']/self.express['mu2']/self.fit['b']})
        self.param['h'] = self.express['xi2epsilon2']*self.param['l_t']/self.express['mu2']

    def find_pparams(self):
        """Dimensionless parameters: finds extra parameters from initial conditions and parameters. Call once."""
        self.express['lJfix'] = self.exper['forJ']*self.express['lam']

    def dump_vars(self, h5, prefix=""):
        s = {'exper'   : [self.exper, prefix+"conditions", "Experiment conditions"],
             'param'   : [self.param, prefix+"parameters", "FC parameters"],
             'fit'     : [self.fit, prefix+"performance", "FC performance parameters"],
             'express' : [self.express, prefix+"expressions", "FC dependent parameters"]}
        for v in s.values():
            gr = h5.create_group("/", v[1], v[2])
            for k1, v1 in v[0].items():
                setattr(gr._v_attrs, "%s" % k1, v1)

    def read_fit(self, h5, prefix=""):
        keys = ['b', 'j_0', 'R_Ohm', 'sigma_t', 'D_O_GDL', 'Cdl', 'lambda_eff']
        gr = h5.get_node("/%s" % (prefix+"performance"))
        for k in keys:
            self.fit[k] = getattr(gr._v_attrs, k)
