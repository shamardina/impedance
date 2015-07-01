# -*- coding: utf-8 -*-
# constant flow velocity

import math
import tables

default_channels = 100

class Tafel(object):
    """Stationary model with Tafel overpotential. Class for stationary one point simulations."""

    def __init__(self, fc, channels=default_channels):
        self.fc = fc
        self.channels = channels
        self.n_z_mesh = [1.0*z/channels for z in xrange(0, channels + 1)]
        self.c_h_v = None
        self.c_t_v = None
        self.j_v = None
        self.eta_v = None

    def eta(self):
        fc = self.fc
        if fc.fit["lam_eff"] <= 1.0:
            print "Tafel overpotential NOT ok"
            self.eta_v = None
            return None
        print "Tafel overpotential ok"
        f_lam_J = -fc.express["lJfix"]*math.log(1.0 - 1.0/fc.fit["lam_eff"])
        self.eta_v = fc.fit["b"]*(math.log(f_lam_J/fc.fit["j_0"]) - math.log(1.0 - f_lam_J/fc.express["j_lim"]))
        return self.eta_v

    def profiles_v(self, n_eta):
        if n_eta is None:
            return None
        self.eta_v = n_eta*self.fc.fit["b"]
        self.j_v = [self.j_profile(nz) for nz in self.n_z_mesh]
        self.c_h_v = [self.c_h_profile(nz) for nz in self.n_z_mesh]
        self.c_t_v = [self.c_t_profile(nz) for nz in self.n_z_mesh]
        return True

    def j_profile(self, n_z):
        return -self.fc.express["lJfix"]*((1.0 - 1.0/self.fc.fit["lam_eff"])**(n_z))*math.log(1.0 - 1.0/self.fc.fit["lam_eff"])

    def c_h_profile(self, n_z):
        return self.fc.exper["c_ref"]*((1.0 - 1.0/self.fc.fit["lam_eff"])**(n_z))

    def c_t_profile(self, n_z):
        return self.fc.exper["c_ref"]*((1.0 - 1.0/self.fc.fit["lam_eff"])**(n_z))*(1.0 + self.fc.express["lJfix"]/self.fc.express["j_lim"]*math.log(1.0 - 1.0/self.fc.fit["lam_eff"])) # == self.c_h_profile(n_z) - self.j_profile(n_z)*self.fc.exper["c_ref"]/self.fc.express["j_lim"]

    def dump_results(self, h5, prefix=""):
        method = self.__class__.__name__
        results = "%s_results" % (prefix+method)
        gr = h5.create_group("/", results, "Stationary %s results" % method)
        double_atom = tables.Float64Atom()
        gr._v_attrs.eta_v = self.eta_v
        gr._v_attrs.n_eta_v = self.eta_v/self.fc.fit["b"]
        gr._v_attrs.channels = self.channels
        a = h5.create_array("/%s" % results, "c_h_v", atom=double_atom, shape=(self.channels+1,),
                            title="Dimentional concentration in %s approximation." % method)
        a[:] = [self.c_h_profile(nz) for nz in self.n_z_mesh]
        a = h5.create_array("/%s" % results, "n_c_h_v", atom=double_atom, shape=(self.channels+1,),
                            title="Dimentionless concentration in %s approximation (/c_ref)." % method)
        a[:] = [self.c_h_profile(nz)/self.fc.exper["c_ref"] for nz in self.n_z_mesh]
        a = h5.create_array("/%s" % results, "j_v", atom=double_atom, shape=(self.channels+1,),
                            title="Dimentional current density in %s approximation." % method)
        a[:] = [self.j_profile(nz) for nz in self.n_z_mesh]
        a = h5.create_array("/%s" % results, "n_j_v", atom=double_atom, shape=(self.channels+1,),
                            title="Dimentionless current density in %s approximation." % method)
        a[:] = [self.j_profile(nz)/self.fc.express["j_ref"] for nz in self.n_z_mesh]

    def read_results(self, h5, prefix=""):
        method = self.__class__.__name__
        results = "%s_results" % (prefix+method)
        res = h5.get_node("/%s" % results)
        self.c_h_v = []
        self.j_v = []
        self.c_h_v[:] = res.c_h_v[:]
        self.j_v[:] = res.j_v[:]
        self.eta_v = res._v_attrs.eta_v
        self.channels = res._v_attrs.channels
        self.n_z_mesh = [1.0*z/self.channels for z in xrange(0, self.channels + 1)]


class FCSimple(object):
    """A simple class for FC. Defines or calculates the FC parameters."""

    def __init__(self, param=None, exper=None):
        self.param = {} if param is None else param.copy()
        self.exper = {} if exper is None else exper.copy()
        self.fit = {}
        self.express = {}

    def find_vars(self):
        """Basic parameters: finds extra variables from demensional ones. Call every time when performance parameters are changed."""
        p = self.exper.copy()
        p.update(self.fit)
        p.update(self.param)
        self.express.update({"j_ref"       : p["sigma_t"]*p["b"]/p["l_t"],
                             "nD_d"        : 4.0*p["F"]*p["D_O_GDL"]*p["c_ref"]/p["sigma_t"]/p["b"],
                             "mu2"         : 4.0*p["F"]*p["c_ref"]/p["Cdl"]/p["b"],
                             "mu"          : math.sqrt(4.0*p["F"]*p["c_ref"]/p["Cdl"]/p["b"]),
                             "xi2"         : 8.0*p["F"]*p["h"]*p["c_ref"]*p["j_0"]/(p["Cdl"]*p["sigma_t"]*(p["b"]**2)),
                             "epsilon2"    : p["sigma_t"]*p["b"]/2.0/p["j_0"]/p["l_t"],
                             "xi2epsilon2" : 4.0*p["F"]*p["h"]*p["c_ref"]/p["Cdl"]/p["b"]/p["l_t"],
                             "j_lim"       : 4.0*p["F"]*p["D_O_GDL"]*p["c_ref"]/p["l_d"],
                             "nj_lim"      : 4.0*p["F"]*p["D_O_GDL"]*p["c_ref"]*p["l_t"]/p["l_d"]/p["sigma_t"]/p["b"],
                             "lJfix"       : p["lam_eff"]*p["forJ"]})

    def find_params(self):
        """Basic parameters: finds extra parameters from initial conditions and parameters. Call once."""
        self.express["nl_d"] = self.param["l_d"]/self.param["l_t"]

    def find_opt(self):
        """Finds optional properties. Call once."""
        self.express.update({"alpha" : self.exper["R"]*self.exper["T"]/self.fit["b"]/self.exper["F"],
                             "kappa" : self.param["l_m"]/(self.fit["R_Ohm"]-2.0*(self.express["nl_d"]*self.param["l_t"])/self.param["sigma"])})

    def find_vvars(self):
        """Dimensionless parameters: finds extra variables from given ones. Call every time when performance parameters are changed."""
        j_ref = self.express["epsilon2"]*2.0*self.fit["j_0"]
        self.express.update({"j_ref"       : j_ref,
                             "xi2epsilon2" : self.express["xi2"]*self.express["epsilon2"],
                             "j_lim"       : self.express["nD_d"]/self.express["nl_d"]*j_ref,
                             "nj_lim"      : self.express["nD_d"]/self.express["nl_d"],
                             "mu"          : math.sqrt(self.express["mu2"])})
        sigma_t = self.express["epsilon2"]*2.0*self.fit["j_0"]*self.param["l_t"]/self.fit["b"]
        self.fit.update({"sigma_t" : sigma_t,
                         "D_O_GDL" : self.express["nD_d"]*sigma_t*self.fit["b"]/4.0/self.exper["F"]/self.exper["c_ref"],
                         "Cdl"     : 4.0*self.exper["F"]*self.exper["c_ref"]/self.express["mu2"]/self.fit["b"]})
        self.param["h"] = self.express["xi2epsilon2"]*self.param["l_t"]/self.express["mu2"]

    def find_pparams(self):
        """Dimensionless parameters: finds extra parameters from initial conditions and parameters. Call once."""
        self.express["lJfix"] = self.exper["forJ"]*self.fit["lam_eff"]

    def dump_vars(self, h5, prefix=""):
        s = {"exper"   : [self.exper, prefix+"conditions", "Experiment conditions"],
             "param"   : [self.param, prefix+"parameters", "FC parameters"],
             "fit"     : [self.fit, prefix+"performance", "FC performance parameters"],
             "express" : [self.express, prefix+"expressions", "FC dependent parameters"]}
        for v in s.values():
            gr = h5.create_group("/", v[1], v[2])
            for k1, v1 in v[0].items():
                setattr(gr._v_attrs, "%s" % k1, v1)

    def read_fit(self, h5, prefix=""):
        keys = ["b", "j_0", "R_Ohm", "sigma_t", "D_O_GDL", "Cdl", "lam_eff"]
        gr = h5.get_node("/%s" % (prefix+"performance"))
        for k in keys:
            self.fit[k] = getattr(gr._v_attrs, k)


class FCCCL(FCSimple):
    """A class for FC with finite oxygen transport in CCL."""

    def find_vars(self):
        FCSimple.find_vars(self)
        self.express["nD_t"] = 4.0*self.exper["F"]*self.fit["D_O_CCL"]*self.exper["c_ref"]/self.fit["sigma_t"]/self.fit["b"]

    def find_vvars(self):
        FCSimple.find_vvars(self)
        self.fit["D_O_CCL"] = self.express["nD_t"]*self.fit["sigma_t"]*self.fit["b"]/4.0/self.exper["F"]/self.exper["c_ref"]

    def read_fit(self, h5, prefix=""):
        FCSimple.read_fit(self, h5, prefix)
        gr = h5.get_node("/%s" % (prefix+"performance"))
        self.fit["D_O_CCL"] = gr._v_attrs.D_O_CCL


def main():
    scale = 10000.0
    import parameters as p
    exper = {"gas_O"   : p.gas_O,
             "T"       : p.T,
             "R"       : p.R,
             "F"       : p.F,
             "c_ref"   : p.c_ref,
             "forJ"    : 600.0,
             "lam_exp" : 300.0}

    param = {"h"     : p.h,
             "l_d"   : p.l_d,
             "sigma" : p.sigma,
             "l_t"   : p.l_t,
             "l_m"   : p.l_m}
    fuel_cell = FCSimple(param, exper)
    fuel_cell.fit = {"j_0"     : p.j_0,
                     "R_Ohm"   : 0.000008,
                     "b"       : p.b,
                     "sigma_t" : p.sigma_t,
                     "Cdl"     : p.Cdl,
                     "lam_eff" : exper["lam_exp"],
                     "D_O_GDL" : p.D_O_GDL,
                     "D_O_CCL" : p.D_O_GDL/10.0}
    filename = "results/newSTsimple.h5"
#    with tables.open_file(filename, "w") as r:
#        fuel_cell.dump_vars(r, "found_")
    t = tables.open_file(filename, "w")
    fuel_cell.dump_vars(t, "found_")
    #temp = FCSimple()
    #temp.read_fit(t, "found_")
    #t.close()
    print ["%s=%s" % (k, fuel_cell.fit[k]) for k in sorted(fuel_cell.fit.keys())]
    #print ["%s=%s" % (k, temp.fit[k]) for k in sorted(temp.fit.keys())]
    fuel_cell.find_vars()
    fuel_cell.find_params()
    tafel = Tafel(fuel_cell)
    tafel.profiles_v(1.0)
    tafel.dump_results(t, "found_")
    t.close()


if __name__ == "__main__":
    main()
