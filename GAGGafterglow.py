# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:26:30 2019

@author: Nicola Zampa - INFN Trieste

Change log

Thu Jun 11 2020
  - The gain is obtained as a function of the PMT supply
    voltage V by means of three parameters (A, b, n) from
    the formula    Go = (A*V)**(b*n)
  - Introduced the gain nonlinearity via the formula
                   G = Go*(1+d*exp((I-Io)/Is))
    where I is the anode current, Io [nA] is the current
    at which the relative deviation is d, and Is is a scale
    factor
  - The parameters A, b, d, Io, and s can be obtained
    by fitting the model to the data. The default values
    are: A = 6.80e-3, b = 0.5962, n = 10, d = 0.2, Io = 1e5,
    and Is = 1e4
  - Renamed the methods that retrieve the fit parameters
    by adding _fit after get (e.g. get_Phi -> get_fit_Phi)

Tue Jun 23 2020
  - Added storing and retrival of the fit initial parameters
    and bounds (rors of triplets: lower bound, parameter,
    upper bound)
  - Removed the power law dependence from the fit function
  - Removed useless fit features to speed up the calculation

Wed Sep 30 2020
  - Corrected the filtering procedure to better fit the data
    with a spline.
  - Now the spline is no more monotonic as some temperature
    dependence may create oscillations in the data.

Mon Oct 5 2020
  - Added the initial photon emission rates due to the release
    of trapped charge carriers already filled at the start of
    the test
  - Errors on fluxes are now relative

Tue Oct 6 2020
  - Added the possibility to subtract a wveform from the
    measurements before fitting (the measurement data are
    not modified)
  - Corrected a bug that prevented fitting the data when
    the flux in one of the runs is not positive
"""

import numpy as np
import numpy.polynomial.polynomial as npPoly
import pandas as pd
import scipy.constants as spc
import scipy.interpolate as interp
import scipy.optimize as spopt
import matplotlib.pyplot as plt
import pickle

def rms_func(x, *p):
    return p[0]+p[1]*np.sqrt(x)+p[2]*x

def rms_power(x, *p):
    return p[0]+p[1]*np.power(x, p[2])

# thanks to matehat @ stackoverflow
def window_rms(a, window_size):
  a2 = np.power(a,2)
  window = np.ones(window_size)/float(window_size)
  return np.sqrt(np.convolve(a2, window, 'valid'))

# std() on spline fit residuals wrt signal average in window
def signal_rms(data, errors, window_size):
    n = len(data)//window_size
    rms = np.empty(n, dtype=float)
    sig = np.empty_like(rms)
    for i in range(n):
        rms[i] = errors[i*window_size:(i+1)*window_size].std()
        sig[i] = data[i*window_size:(i+1)*window_size].mean()
    return sig, rms

class gagg:

    def __init__(self, bias=1500.0, A=6.80e-3, b=0.5962, n=10,
                 d=0.2, Io=1e5, sf=0.2, ns=3.0,
                 PMT_eff=0.1, LG_eff=0.4, Tm_err_neg=0,
                 Tm_err_pos=15, scale=1e9):
        self.__verbose = False
        self.Ni = 10
        self.Ny = 5
        self.feval = 0
        self.last_time = 0
        self.color = 0
        self.__active = []
        self.T0 = []
        self.Te = []
        self.Ti = []
        self.Tm = []
        self.Phi = []
        self.Phi_err = []
        self.bias = bias
        self.V = []
        self.A = A
        self.b = b
        self.n = n
        self.d = d
        self.Io = Io
        if sf < 1.0:
            self.sf = sf
        else:
            self.sf = 0.999999
            print("Warning: sf is too large, set to {}".format(self.sf))
        self.__rms_func = rms_power
        self.idx = []
        self.sim_idx = []
        self.ns = ns
        self.__PMT_eff = PMT_eff
        self.__LG_eff = LG_eff
        self.Tm_err_neg = Tm_err_neg
        self.Tm_err_pos = Tm_err_pos
        self.__scale = scale
        self.data = []
        self.runs = 0
        self.Ne = 0
        self.par = None
        self.perr = None
        self.__initial_parameters = None
        self.free = None
        self.ftol = 1e-11
        self.xtol = 1e-11
        self.gtol = 1e-11
        self.max_nfev = 100
        self.sim_Ne = 0
        self.sim_Dn = None
        self.sim_meas_err = None
        self.sim_par = None
        self.t = None
        self.m = None
        self.f = None
        self.r = None
        self.s = None
        self.__free_tau = None
        self.__tau_idx = None
        self.__sort_trap_data = True
        self.fitted = False
        self.simulated = False

    def clear(self):
        self.Ni = 10
        self.Ny = 5
        self.feval = 0
        self.last_time = 0
        self.color = 0
        self.__active = []
        self.T0 = []
        self.Te = []
        self.Ti = []
        self.Tm = []
        self.Phi = []
        self.Phi_err = []
        self.V = []
        self.idx = []
        self.data = []
        self.runs = 0
        self.Ne = 0
        self.par = None
        self.perr = None
        self.__initial_parameters = None
        self.free = None
        self.ftol = 1e-11
        self.xtol = 1e-11
        self.gtol = 1e-11
        self.max_nfev = 100
        self.sim_Ne = 0
        self.sim_Dn = None
        self.sim_meas_err = None
        self.sim_par = None
        self.t = None
        self.m = None
        self.f = None
        self.r = None
        self.s = None
        self.__free_tau = None
        self.__tau_idx = None
        self.fitted = False
        self.simulated = False

    def clear_simulation(self):
        if self.simulated:
            for i in range(self.runs):
                for j in range(len(self.data[i])):
                    del self.data[i][j]['s']
        self.simulated = False

    def set_verbose(self, v=False):
        self.__verbose = bool(v)

    def get_verbose(self):
        return self.__verbose

    def set_gain_parameters(self, A=6.80e-3, b=0.5962, n=10):
        if self.A != A:
            self.A = A
            self.fitted = False
        if self.b != b:
            self.b = b
            self.fitted = False
        if self.n != int(n):
            self.n = int(n)
            self.fitted = False

    def set_nonlinearity_parameters(self, d=0.2, Io=1e5, sf=0.2):
        if self.d != d:
            self.d = d
            self.fitted = False
        if self.Io != Io:
            self.Io = Io
            self.fitted = False
        if sf >= 1.0:
            sf = 0.999999
            print("Warning: sf is too large, set to {}".format(self.sf))
        if self.sf != sf:
            self.sf = sf
            self.fitted = False

    def get_gain_parameters(self):
        return self.A, self.b, self.n
        
    def get_gain(self):
        return (self.A*np.array(self.V))**(self.b*self.n)

    def get_system_gain(self, run=0):
        if run < 0 or run >= self.runs:
            raise ValueError('run should belong to the interval [0..{}]'.format(self.runs-1))
        PMT_gain = (self.A*np.array(self.V[run]))**(self.b*self.n)
        return spc.elementary_charge*PMT_gain*self.__PMT_eff*self.__LG_eff*self.__scale

    def get_nonlinearity_parameters(self):
        return self.d, self.Io, self.sf

    def set_PMT_eff(self, PMT_eff=1.0):
        n = self.runs
        k = self.__PMT_eff/PMT_eff
        if self.fitted:
            self.par[4*n+3+self.Ne:4*n+3+2*self.Ne] *= k
            self.par[4*n+3+2*self.Ne:4*n+3+3*self.Ne] *= k
        if self.simulated:
            self.sim_par[4*n+3+self.Ne:4*n+3+2*self.Ne] *= k
            self.sim_par[4*n+3+2*self.Ne:4*n+3+3*self.Ne] *= k
        self.__PMT_eff = PMT_eff

    def get_PMT_eff(self):
        return self.__PMT_eff

    def set_LG_eff(self, LG_eff=1.0):
        n = self.runs
        k = self.__LG_eff/LG_eff
        if self.fitted:
            self.par[4*n+3+self.Ne:4*n+3+2*self.Ne] *= k
            self.par[4*n+3+2*self.Ne:4*n+3+3*self.Ne] *= k
        if self.simulated:
            self.sim_par[4*n+3+self.Ne:4*n+3+2*self.Ne] *= k
            self.sim_par[4*n+3+2*self.Ne:4*n+3+3*self.Ne] *= k
        self.__LG_eff = LG_eff

    def get_LG_eff(self):
        return self.__LG_eff

    def set_scale(self, scale=1.0):
        n = self.runs
        k = scale/self.__scale
        for i in range(n):
            for j in range(self.data[i]):
                self.data[i][j]['m']['i'] *= k
                self.data[i][j]['m']['f'] *= k
                self.data[i][j]['m']['r'] *= k
        if self.fitted:
            self.m *= k
            self.f *= k
            self.r *= k
            self.s *= k
        if self.simulated:
            for i in range(n):
                for j in range(self.data[i]):
                    self.data[i][j]['s']['t'] *= k
                    self.data[i][j]['s']['i'] *= k
                    self.data[i][j]['s']['f'] *= k
                    self.data[i][j]['s']['r'] *= k
        self.__scale = scale

    def get_scale(self):
        return self.__scale

    def set_rms_function(self, rms_func):
        if not callable(rms_func):
            raise TypeError('The argument must be a function')
        self.__rms_func = rms_func

    def get_rms_function(self):
        return self.__rms_func

    def set_rms_parameters(self, rms_par, Io, r=None, m=None):
        n = self.runs
        if n == 0:
            raise RuntimeError('Please load the measure data')
        try:
            r = int(r)
        except:
            r = None
        try:
            m = int(m)
        except:
            m = None
        if not isinstance(rms_par, (list, tuple, np.ndarray)):
            raise TypeError('rms_par must be a list, a tuple, or a np.ndarray')
        if len(rms_par) < 3:
            raise RuntimeError('the rms_par array must have at least three components')
        try:
            Io = float(Io)
        except:
            raise TypeError('Io must be a float number, or equivalent')
        if r is None:
            for r in range(n):
                for m in range(len(self.data[r])):
                    self.data[r][m]['rms_par'] = [list(rms_par), Io]
        else:
            if r < 0 or r > self.runs-1:
                raise ValueError('r must have a value between 0 and {}'.format(self.runs-1))
            if m is None:
                for m in range(len(self.data[r])):
                    self.data[r][m]['rms_par'] = [list(rms_par), Io]
            else:
                if m < 0 or m > len(self.data[r])-1:
                    raise ValueError('r must have a value between 0 and {}'.format(len(self.data[r])-1))
                self.data[r][m]['rms_par'] = [list(rms_par), Io]

    def get_rms_parameters(self, r, m):
        n = self.runs
        if n == 0:
            raise RuntimeError('Please load the measure data')
        try:
            r = int(r)
        except:
            raise TypeError('r must be an int number, or equivalent')
        if m < 0 or m > len(self.data[r])-1:
            raise ValueError('r must have a value between 0 and {}'.format(len(self.data[r])-1))
        try:
            m = int(m)
        except:
            raise TypeError('m must be an int number, or equivalent')
        if m < 0 or m > len(self.data[r])-1:
            raise ValueError('r must have a value between 0 and {}'.format(len(self.data[r])-1))
        return self.data[r][m]['rms_par']

    def set_sort_trap_data(self, flag):
        if not isinstance(flag, bool):
            raise RuntimeError('The flag must be a boolean value')
        self.__sort_trap_data = flag

    def get_sort_trap_data(self):
        return self.__sort_trap_data

    def load_measure(self, fname, Ti, Tm, Phi, Phi_err, delimiter=' ', header=None, skip=None, keep=None, Ni=10, Ny=5):
        self.Ni = Ni
        self.Ny = Ny
        empty = False
        try:
            data = pd.read_csv(fname, delimiter=delimiter, header=header)
        except:
            empty = True
            meas = np.zeros((0,0), dtype=float)
            T0 = pd.to_datetime(fname, format='%Y%m%d_%H%M%S').value/1e9
        else:
            if data.shape[1] == 3:
                T0 = pd.to_datetime(data[2][0], format='%H:%M:%S_%Y/%m/%d').value/1e9
            else:
                T0 = pd.to_datetime(fname[9:24], format='%Y%m%d_%H%M%S').value/1e9
            meas = np.empty((data.shape[0], 2), dtype=float)
            meas[:, 0] = data.iloc[:, 1]+(Tm+Ti)
            meas[:, 1] = -data.iloc[:, 0]*self.__scale
        T0 -= Tm+Ti # Time origin at the beginning of irradiation
        try:
            skip = float(skip)
        except:
            pass
        else:
            if skip > 0.:
                idx = (meas[:, 0] > meas[0, 0]+skip).nonzero()[0]
                if len(idx) > 0:
                    meas = meas[idx[0]:, :]
                else:
                    meas = meas[:0, :]
        try:
            keep = float(keep)
        except:
            pass
        else:
            if keep > 0.:
                idx = (meas[:, 0] > meas[0, 0]+keep).nonzero()[0]
                if len(idx) > 0:
                    meas = meas[:idx[0], :]
                else:
                    meas = meas[:0, :]
        if self.runs:
            if not empty:
                if T0 > self.last_time:
                    meas[:, 0] += T0-self.T0[0]
                    dlen = 0
                    for d in self.data[-1]:
                        dlen += d['t'].size
                else:
                    raise RuntimeError('The measure must follow already loaded data')
        self.T0.append(T0)
        Te = T0-self.T0[0]+Ti
        self.Te.append(Te)
        self.Ti.append(Ti)
        self.Tm.append(Tm)
        self.Phi.append(Phi)
        self.Phi_err.append(Phi_err)
        self.V.append(self.bias)
        self.data.append([])
        self.data[-1].append({})
        if not empty:
            self.data[-1][-1]['t'] = meas[:, 0]
        else:
            self.data[-1][-1]['t'] = np.zeros(0, dtype=float)
        self.data[-1][-1]['rms_par'] = None
        self.data[-1][-1]['m'] = {}
        if not empty:
            self.data[-1][-1]['m']['i'] = meas[:, 1]
            self.last_time = self.T0[0]+meas[-1, 0]
        else:
            self.data[-1][-1]['m']['i'] = np.zeros(0, dtype=float)
            self.last_time = T0+Ti+Tm
        self.filter_data()
        self.__active.append(self.runs)
        self.runs += 1
        print('data {} loaded'.format(fname))

    def append_measure(self, fname, delimiter=' ', header=None, skip=None, keep=None, Ni=10, Ny=5):
        if self.runs == 0:
            raise RuntimeError('Empty data: you should use load_measure instead')
        self.Ni = Ni
        self.Ny = Ny
        data = pd.read_csv(fname, delimiter=delimiter, header=header)
        T0 = pd.to_datetime(data[2][0], format='%H:%M:%S_%Y/%m/%d').value/1e9
        T0 -= self.T0[0]
        meas = np.empty((data.shape[0], 2), dtype=float)
        meas[:, 0] = data.iloc[:, 1]+T0
        meas[:, 1] = -data.iloc[:, 0]*self.__scale
        if isinstance(skip, int) and skip > 0:
            idx = (meas[:, 0] > meas[0, 0]+skip).nonzero()[0]
            if len(idx) > 0:
                meas = meas[idx[0]:, :]
            else:
                meas = meas[:0, :]
        if isinstance(keep, int) and keep > 0:
            idx = (meas[:, 0] > meas[0, 0]+keep).nonzero()[0]
            if len(idx) > 0:
                meas = meas[:idx[0], :]
            else:
                meas = meas[:0, :]
        self.data[-1].append({})
        self.data[-1][-1]['t'] = meas[:, 0]
        self.data[-1][-1]['rms_par'] = None
        self.data[-1][-1]['m'] = {}
        self.data[-1][-1]['m']['i'] = meas[:, 1]
        self.filter_data()
        self.last_time = self.T0[0]+meas[-1, 0]
        print('file {} loaded'.format(fname))

    def save_data(self, fname):
        data = {}
        data['T0'] = self.T0
        data['Te'] = self.Te
        data['Ti'] = self.Ti
        data['Tm'] = self.Tm
        data['Phi'] = self.Phi
        data['Phi_err'] = self.Phi_err
        data['V'] = self.V
        data['last_time'] = self.last_time
        data['active'] = self.__active
        data['runs'] = self.runs
        data['data'] = self.data
        if self.__rms_func == rms_func:
            data['rms_func'] = 'rms_func'
        elif self.__rms_func == rms_power:
            data['rms_func'] = 'rms_power'
        else:
            data['rms_func'] = 'unknown'
        data['rms_par'] = self.__rms_par
        data['simulated'] = self.simulated
        data['fitted'] = self.fitted
        data['s'] = self.s
        if self.fitted:
            data['t'] = self.t
            data['m'] = self.m
            data['f'] = self.f
            data['r'] = self.r
        try:
            with open(fname, 'wb') as fp:
                pickle.dump(data, fp)
        except Exception:
            print('Error writing data to file {}'.format(fname))

    def load_data(self, fname):
        try:
            with open(fname, 'rb') as fp:
                data = pickle.load(fp)
        except Exception:
            print('Error reading data from file {}'.format(fname))
        else:
            ckeys = set(('T0', 'Te', 'Ti', 'Tm', 'Phi', 'Phi_err', 'V',
                         'last_time', 'active', 'runs', 'data'))
            dkeys = set(data.keys())
            if dkeys == ckeys:
                self.T0 = data['T0']
                self.Te = data['Te']
                self.Ti = data['Ti']
                self.Tm = data['Tm']
                self.Phi = data['Phi']
                self.Phi_err = data['Phi_err']
                self.V = data['V']
                self.last_time = data['last_time']
                self.__active = data['active']
                self.runs = data['runs']
                self.data = data['data']
                self.__rms_par = data['rms_par']
                if data['rms_func'] == 'rms_func':
                    self.__rms_func = rms_func
                elif data['rms_func'] == 'rms_power':
                    self.__rms_func = rms_power
                else:
                    self.__rms_func = lambda x, *p: np.ones_like(x, dtype=float)
                self.simulated = data['simulated']
                self.fitted = data['fitted']
                self.s = data['s']
                if self.fitted:
                    self.t = data['t']
                    self.m = data['m']
                    self.f = data['f']
                    self.r = data['r']
            else:
                print('The file {} does not contain the expected data'.format(fname))
        
    def get_active(self):
        return self.__active

    def set_active(self, active=None):
        if self.runs == 0:
            raise RuntimeError('Please load the measure data')
        if not isinstance(active, (list, tuple, np.ndarray)):
            if active is None:
                active = np.arange(self.runs)
            else:
                raise RuntimeError('active must be a list, a tuple, or a np.ndarray')
        a = np.unique(active)
        a = [m for m in a if 0 <= m < self.runs]
        self.__active = a

    def get_fit_tolerances(self):
        return self.ftol, self.xtol, self.gtol, self.max_nfev

    def set_fit_tolerances(self, ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=100):
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.max_nfev=max_nfev

    def plot(self, dset='m', which='i', runs=None, log=False, new=True, ms=None, unit='nA'):
        if self.runs == 0:
            raise RuntimeError('Please load the measure data')
        if dset not in ('m', 's'):
            raise RuntimeError('Unknown set')
        if dset == 's' and not self.simulated:
            raise RuntimeError('Please simulate a measure set')
        if dset == 'm' and which not in ('i', 'f', 'r', 'p'):
            raise RuntimeError('Unknown plot specification for measurement set')
        if dset == 's' and which not in ('i', 'f', 'r', 't', 's'):
            raise RuntimeError('Unknown plot specification for simulation set')
        if not isinstance(runs, (list, tuple, np.ndarray)):
            if runs is None:
                runs = self.__active
            elif runs == 'all':
                runs = list(range(self.runs))
            else:
                raise RuntimeError('runs must be a list, a tuple, or a np.ndarray')
        plt.rcParams.update({'font.size': 14})
        if new:
            plt.figure(figsize=(10,8))
            self.color = 0
            if log and which != 'r':
                plt.semilogy()
        for i in runs:
            for j in range(len(self.data[i])):
                if dset == 'm' and which == 'p':
                    plt.scatter(self.data[i][j]['m']['x'], self.data[i][j]['m']['y'], color='C{}'.format(self.color), s=ms if ms else 50)
                else:
                    plt.scatter(self.data[i][j]['t'], self.data[i][j][dset][which], color='C{}'.format(self.color), s=ms if ms else 1)
        plt.xlabel('time [s]')
        plt.ylabel('current [{}]'.format(unit))
        plt.grid(True, 'both')
        self.color += 1
        
    def filter_data(self, k=-1, l=-1, w='m'):
        t = self.data[k][l]['t']
        i = self.data[k][l][w]['i']
        if i.size == 0:
            self.data[k][l][w]['x'] = np.zeros(0, dtype=float)
            self.data[k][l][w]['f'] = np.zeros(0, dtype=float)
            self.data[k][l][w]['r'] = np.zeros(0, dtype=float)
            return            
        ii = i.cumsum()-i[0]
        knots = np.empty(self.Ni, dtype=float)
        di = ii[-1]/(self.Ni+1)
        j = 0
        for n, ic in enumerate(ii):
            if ic > di*(j+1):
                knots[j] = t[n]
                j += 1
        tck = interp.splrep(t, i, k=3, t=knots)
        self.data[k][l][w]['x'] = knots
        self.data[k][l][w]['f'] = interp.splev(t, tck)
        self.data[k][l][w]['r'] = self.data[k][l][w]['i']-self.data[k][l][w]['f']
    
    def __afterglow(self, t, *p):
        n = self.runs
        if len(p) > 0:
            self.par[self.free] = p
            par = self.par
            Ne = self.Ne
            m = self.Phi_scale
            idx = self.idx
        else:
            par = self.sim_par
            Ne = self.sim_Ne
            m = self.sim_Phi_scale
            idx = self.sim_idx
        phi = par[:n]                      # fluxes
        ti = np.array(self.Ti)+par[n:2*n]  # durations
        f = (phi*ti).sum()                 # total fluence
        c = par[-2]/f                      # absorption coefficient
        b = par[-1]                        # PMT dark current
        y = np.zeros_like(t)               # output vector
        delay = par[2*n:3*n]               # variations of measurement delays
        bias = par[3*n:4*n]                # PMT bias
        d = par[4*n]                       # parameters for PMT gain determination
        Io = par[4*n+1]
        sf = par[4*n+2]
#        tau_idx = par[4*n+3:4*n+3+Ne].argsort()
#        tau = par[4*n+3:4*n+3+Ne][tau_idx]
#        no = par[4*n+3+Ne:4*n+3+2*Ne][tau_idx]
#        ke = par[4*n+3+2*Ne:4*n+3+3*Ne][tau_idx]
        tau = par[4*n+3:4*n+3+Ne]          # mean lifetimes
        no = par[4*n+3+Ne:4*n+3+2*Ne]      # initial conditions
        ke = par[4*n+3+2*Ne:4*n+3+3*Ne].copy() # n[i]
        dn = par[4*n+3+3*Ne:4*n+3+4*Ne]    # Dn[i]
        s = spc.elementary_charge*m*self.__PMT_eff*self.__LG_eff*self.__scale
        if self.__verbose:
            warn_1 = np.ones(Ne, dtype=bool)
            warn_2 = np.ones(Ne, dtype=bool)
        for i in range(Ne):
            if no[i] > 0.: # apply initial conditions if needed
                y += s/m*no[i]*np.exp(-t/tau[i])/tau[i]
        for i, (Phi, Ti, Te, dT) in enumerate(zip(phi, ti, self.Te, delay)):
            j = idx[i]
            if Phi > 0. and j < len(t):
                tm = t[j:]-Te+dT # measurement times from the end of the irradiation
                for k in range(Ne): # emission from traps
                    df = dn[k]*Phi/f # partial variation in n[i]
                    if df < 0. and ke[k] < -df*Ti:
                        Ti = -ke[k]/df # time at which these traps stop capturing charge carriers
                        if self.__verbose:
                            if warn_1[k]:
                                print('the traps of specie {} get extinguished at run {}'.format(k+1, i+1))
                                warn_1[k] = False
                    if (ke[k] > 0. and Ti > 0.) or (ke[k] == 0. and dn[k] > 0.):
                        N1 = (ke[k]-df*tau[k])*Phi*(1-np.exp(-Ti/tau[k]))
                        N2 = df*Ti*Phi
                        y[j:] += s*(N1+N2)*np.exp(-tm/tau[k]) # photoelectrons emitted by the photocathode of the PMT
                        ke[k] += df*Ti # update n[i]
                    elif self.__verbose:
                        if warn_2[k]:
                            print('n[{}] = {}, Dn[{}] = {}'.format(k+1, ke[k], k+1, df))
                            warn_2[k] = False
        Phi_cum = 0.0
        for i, (j, V, Phi, Ti) in enumerate(zip(idx[:-1], bias, phi, ti)): # account for absorption and bias
            gain = (self.A*V)**(self.b*self.n) # PMT gain
            if j < len(t):
                Phi_cum += Phi*Ti
                transmission = (1-c*Phi_cum)
                k = idx[i+1]
                if k < len(t):
                    y[j:k] *= gain*transmission
                else:
                    y[j:] *= gain*transmission
        gain_lin = (1.0+d*np.exp((y/Io-1)/sf))
        if self.__verbose:
            print('gain_lin.min() = {}, gain_lin.max() = {}.'.format(gain_lin.min(), gain_lin.max()))
        y *= gain_lin             # correct for loss of linearity in the gain
        y += b                    # add the PMT dark current
        self.feval += 1
        return y

    def fit(self, dset='m', which='i', io=None, bias=None,
            nlin=None, Phi=None, DTi=None, delay=None,
            absorp=0.0, tau=None, tau_lo=None, tau_hi=None,
            no=None, n=None, Dn=None, max_abs = 1.0, Ib=0.0,
            Ib_err=0.1, fix_bias=True, fix_nlin=True,
            fix_flux=True, fix_Ti=True, fix_delay=True,
            fix_abs=True, fix_tau=False, fix_init=True,
            fix_rate=False, fix_coef=True, t_max = -1,
            verbose=False):
        if self.runs == 0:
            raise RuntimeError('Please load the measure data')
        k = self.runs
        if not dset in ('m', 's'):
            raise RuntimeError('Unknown set')
        if not which in ('i', 'f'):
            raise RuntimeError('Wrong data specification')
        if dset == 's' and not self.simulated:
            raise RuntimeError('Please run a simulation')
        if bias is not None:
            if not isinstance(bias, (list, tuple, np.ndarray)):
                raise RuntimeError('bias must be a list, a tuple, or a np.ndarray')
            if len(bias) != k:
                raise RuntimeError('len(bias) must be equal to the number of loaded measures')
        if nlin is not None:
            if not isinstance(nlin, (list, tuple, np.ndarray)):
                raise RuntimeError('nlin must be a list, a tuple, or a np.ndarray')
            if len(nlin) != 3:
                raise RuntimeError('len(nlin) must be equal to three')
        if Phi is not None:
            if not isinstance(Phi, (list, tuple, np.ndarray)):
                raise RuntimeError('Phi must be a list, a tuple, or a np.ndarray')
            if len(Phi) != k:
                raise RuntimeError('len(Phi) must be equal to the number of loaded measures')
        if DTi is not None:
            if not isinstance(DTi, (list, tuple, np.ndarray)):
                raise RuntimeError('Ti must be a list, a tuple, or a np.ndarray')
            if len(DTi) != k:
                raise RuntimeError('len(DTi) must be equal to the number of loaded measures')
        if delay is not None:
            if not isinstance(delay, (list, tuple, np.ndarray)):
                raise RuntimeError('delay must be a list, a tuple, or a np.ndarray')
            if len(delay) != k:
                raise RuntimeError('len(delay) must be equal to the number of loaded measures')
        if tau is None:
            raise RuntimeError('you should supply at least the number of trap species')
        try:
            self.Ne = int(tau)
            tau = -1.0
        except:
            if not isinstance(tau, (list, tuple, np.ndarray)):
                raise RuntimeError('tau must be an integer, a list, a tuple, or a np.ndarray')
            self.Ne = len(tau)
        if tau_lo is not None:
            if not isinstance(tau_lo, (list, tuple, np.ndarray)):
                raise RuntimeError('if given, tau_lo must be a list, a tuple, or a np.ndarray')
            if len(tau_lo) != self.Ne:
                raise RuntimeError('len(tau_lo) must be equal to len(tau)')
        if tau_hi is not None:
            if not isinstance(tau_hi, (list, tuple, np.ndarray)):
                raise RuntimeError('if given, tau_hi must be a list, a tuple, or a np.ndarray')
            if len(tau_hi) != self.Ne:
                raise RuntimeError('len(tau_hi) must be equal to len(tau)')
        if no is not None:
            if not isinstance(no, (list, tuple, np.ndarray)):
                raise RuntimeError('if given, no must be a list, a tuple, or a np.ndarray')
            if len(no) != self.Ne:
                raise RuntimeError('len(no) must be equal to len(tau)')
        if n is not None:
            if not isinstance(n, (list, tuple, np.ndarray)):
                raise RuntimeError('if given, n must be a list, a tuple, or a np.ndarray')
            if len(n) != self.Ne:
                raise RuntimeError('len(n) must be equal to len(tau)')
        if Dn is not None:
            if not isinstance(Dn, (list, tuple, np.ndarray)):
                raise RuntimeError('Dn must be a list, a tuple, or a np.ndarray')
            if len(Dn) != self.Ne:
                raise RuntimeError('len(Dn) must be equal to len(tau)')
        self.feval = 0
        self.t = self.get_time()
        self.m = self.get_data(dset, which)
        self.s = np.zeros(0, dtype=float)
        for i in self.__active:
            for d in self.data[i]:
                if d['rms_par'] is None:
                    self.s = None
                    break
                self.s = np.r_[self.s, self.__rms_func(d[dset][which]-d['rms_par'][1], *d['rms_par'][0])]
            if self.s is None:
                break
        if io is not None:
            if not isinstance(io, (list, tuple, np.ndarray)):
                raise RuntimeError('io must be a list, a tuple, or a np.ndarray')
            if len(io) != self.m.size:
                raise RuntimeError('len(io) must be equal to the length of the data to fit')
            self.m -= io
        if t_max > self.t[0]:
            d_idx = (self.t > t_max).nonzero()[0]
            if len(d_idx) > 0:
                d_idx = d_idx[0]
                self.t = self.t[:d_idx]
                self.m = self.m[:d_idx]
            else:
                d_idx = len(self.t)
        else:
            d_idx = len(self.t)
        self.idx = []
        for Te in self.Te:
            i = (self.t > Te).nonzero()[0]
            if len(i) > 0:
                self.idx.append(i[0])
            else:
                self.idx.append(self.t.size)
        self.idx.append(self.t.size)
        self.par = np.zeros(4*k+3+4*self.Ne+2, dtype=float)
        self.perr = np.zeros_like(self.par)
        if Phi is None:
            Phi = self.Phi
        Phi_min = np.max(Phi)
        for phi in Phi:
            if phi > 0. and phi < Phi_min:
                Phi_min = phi
        self.Phi_scale = 10**int(np.log10(Phi_min))
        self.par[:k] = np.array(Phi)/self.Phi_scale
        if DTi is not None:
            self.par[k:2*k] = DTi
        Ti = np.array(self.Ti)
        Tm = np.array(self.Tm)
        xx = self.t[:10]
        yy = np.log(self.m[:10])
        p = npPoly.Polynomial.fit(xx, yy, 1)
        t_res = .1*(self.t[1]-self.t[0])
        tau_min = max(-.25/p.convert().coef[1], 1.1*t_res)
        xx = self.t[-10:]
        yy = np.log(self.m[-10:])
        p = npPoly.Polynomial.fit(xx, yy, 1)
        tau_max = 4.*np.abs(.25/p.convert().coef[1])
        if isinstance(tau, float) and tau < 0.:
            tau = np.logspace(np.log10(tau_min), np.log10(tau_max), num=self.Ne, endpoint=True)
        gain = (self.A*min(self.V))**(self.b*self.n)
        s = spc.elementary_charge*self.Phi_scale*gain*self.__PMT_eff*self.__LG_eff*self.__scale
        if n is None:
            i = np.array(self.Phi).argmax()
            n = self.m[0]*np.exp(Tm[i]/tau)/(s*self.par[i]*(1.-np.exp(-Ti[i]/tau)))/len(tau)
            n_max = 100.*n.max()
        else:
            n_max = 100.*np.array(n).max()
        if verbose:
            print('t_res =', t_res)
            print('tau_min =', tau_min)
            print('tau_max =', tau_max)
            print('n_max =', n_max)
        if delay is not None:
            self.par[2*k:3*k] = delay
        if bias is not None:
            self.par[3*k:4*k] = bias
        else:
            self.par[3*k:4*k] = self.V
        if nlin is not None:
            self.par[4*k:4*k+3] = nlin
        else:
            self.par[4*k:4*k+3] = (self.d, self.Io, self.sf)
        self.par[4*k+3:4*k+3+self.Ne] = tau
        if no is not None:
            self.par[4*k+3+self.Ne:4*k+3+2*self.Ne] = no
        self.par[4*k+3+2*self.Ne:4*k+3+3*self.Ne] = n
        if Dn is not None:
            self.par[4*k+3+3*self.Ne:4*k+3+4*self.Ne] = Dn
        self.par[-2] = absorp
        self.par[-1] = Ib
        low = np.zeros_like(self.par)
        high = np.full_like(self.par, np.inf)
        low[:k] = np.clip((np.array(self.Phi)*(1-self.ns*np.array(self.Phi_err)))/self.Phi_scale, 0., None)
        high[:k] = (np.array(self.Phi)*(1+self.ns*np.array(self.Phi_err)))/self.Phi_scale
        low[k:2*k] = -np.array(self.Ti)*self.ns*np.array(self.Phi_err)
        high[k:2*k] = np.array(self.Ti)*self.ns*np.array(self.Phi_err)
        low[2*k:3*k] = self.Tm_err_neg
        high[2*k:3*k] = self.Tm_err_pos
        low[3*k:4*k] = 0.5*self.par[3*k:4*k]
        high[3*k:4*k] = 1.5*self.par[3*k:4*k]
        low[4*k] = -1.0
        high[4*k] = 1.0
        low[4*k+1] = self.par[4*k+1]/100.
        high[4*k+1] = 100.*self.par[4*k+1]
        low[4*k+2] = 0.
        high[4*k+2] = 1.
        if tau_lo is None:
            low[4*k+3:4*k+3+self.Ne] = min(t_res, .1*np.min(tau))
        else:
            low[4*k+3:4*k+3+self.Ne] = tau_lo
        if tau_hi is None:
            high[4*k+3:4*k+3+self.Ne] = 100.*max(tau_max, np.max(tau))
        else:
            high[4*k+3:4*k+3+self.Ne] = tau_hi
        ir = np.array(0, dtype=float)
        for i in self.__active:
            ir = self.get_data(r=i)
            if len(ir):
                break
        if len(ir):
            io_max = ir.max()
        else:
            io_max = np.inf
        no_max = io_max*tau_max*self.Phi_scale/s
        low[4*k+3+self.Ne:4*k+3+2*self.Ne] = 0.
        high[4*k+3+self.Ne:4*k+3+2*self.Ne] = no_max
        low[4*k+3+2*self.Ne:4*k+3+3*self.Ne] = 0.
        high[4*k+3+2*self.Ne:4*k+3+3*self.Ne] = n_max
        low[4*k+3+3*self.Ne:4*k+3+4*self.Ne] = -100*n_max
        high[4*k+3+3*self.Ne:4*k+3+4*self.Ne] = 100*n_max
        low[-2] = 0
        high[-2] = max_abs
        if Ib == 0.0:
            low[-1] = -Ib_err
            high[-1] = Ib_err
        else:
            low[-1] = Ib*(1-Ib_err)
            high[-1] = Ib*(1+Ib_err)
        if isinstance(fix_flux, bool):
            if fix_flux:
                free_phi = np.zeros(k, dtype=bool)
            else:
                free_phi = np.ones(k, dtype=bool)
        elif isinstance(fix_flux, int):
            free_phi = np.ones(k, dtype=bool)
            if 0 <= fix_flux < k:
                free_phi[fix_flux] = False
        elif isinstance(fix_flux, (list, tuple, np.ndarray)):
            free_phi = np.ones(k, dtype=bool)
            for i in fix_flux:
                if 0 <= i < k:
                    free_phi[i] = False
        if isinstance(fix_Ti, bool):
            if fix_Ti:
                free_Ti = np.zeros(k, dtype=bool)
            else:
                free_Ti = np.ones(k, dtype=bool)
        elif isinstance(fix_Ti, int):
            free_Ti = np.ones(k, dtype=bool)
            if 0 <= fix_Ti < k:
                free_Ti[fix_Ti] = False
        elif isinstance(fix_Ti, (list, tuple, np.ndarray)):
            free_Ti = np.ones(k, dtype=bool)
            for i in fix_Ti:
                if 0 <= i < k:
                    free_Ti[i] = False
        if isinstance(fix_delay, bool):
            if fix_delay:
                free_delay = np.zeros(k, dtype=bool)
            else:
                free_delay = np.ones(k, dtype=bool)
        elif isinstance(fix_delay, int):
            free_delay = np.ones(k, dtype=bool)
            if 0 <= fix_delay < k:
                free_delay[fix_delay] = False
        elif isinstance(fix_delay, (list, tuple, np.ndarray)):
            free_delay = np.ones(k, dtype=bool)
            for i in fix_delay:
                if 0 <= i < k:
                    free_delay[i] = False
        if isinstance(fix_bias, bool):
            if fix_bias:
                free_bias = np.zeros(k, dtype=bool)
            else:
                free_bias = np.ones(k, dtype=bool)
        elif isinstance(fix_bias, int):
            free_bias = np.ones(k, dtype=bool)
            if 0 <= fix_bias < k:
                free_bias[fix_bias] = False
        elif isinstance(fix_bias, (list, tuple, np.ndarray)):
            free_bias = np.ones(k, dtype=bool)
            for i in fix_bias:
                if 0 <= i < k:
                    free_bias[i] = False
        if isinstance(fix_nlin, bool):
            if fix_nlin:
                free_nlin = np.zeros(3, dtype=bool)
            else:
                free_nlin = np.ones(3, dtype=bool)
        elif isinstance(fix_nlin, int):
            free_nlin = np.ones(3, dtype=bool)
            if 0 <= fix_nlin < 3:
                free_nlin[fix_nlin] = False
        elif isinstance(fix_nlin, (list, tuple, np.ndarray)):
            free_nlin = np.ones(3, dtype=bool)
            for i in fix_nlin:
                if 0 <= i < 3:
                    free_nlin[i] = False
        free_abs = False if fix_abs else True
        if isinstance(fix_tau, bool):
            if fix_tau:
                free_tau = np.zeros(self.Ne, dtype=bool)
            else:
                free_tau = np.ones(self.Ne, dtype=bool)
        elif isinstance(fix_tau, int):
            free_tau = np.ones(self.Ne, dtype=bool)
            if 0 <= fix_tau < self.Ne:
                free_tau[fix_tau] = False
        elif isinstance(fix_tau, (list, tuple, np.ndarray)):
            free_tau = np.ones(self.Ne, dtype=bool)
            for i in fix_tau:
                if 0 <= i < self.Ne:
                    free_tau[i] = False
        self.__free_tau = free_tau.copy()
        if isinstance(fix_init, bool):
            if fix_init:
                free_init = np.zeros(self.Ne, dtype=bool)
            else:
                free_init = np.ones(self.Ne, dtype=bool)
        elif isinstance(fix_init, int):
            free_init = np.ones(self.Ne, dtype=bool)
            if 0 <= fix_init < self.Ne:
                free_init[fix_init] = False
        elif isinstance(fix_init, (list, tuple, np.ndarray)):
            free_init = np.ones(self.Ne, dtype=bool)
            for i in fix_init:
                if 0 <= i < self.Ne:
                    free_init[i] = False
        if isinstance(fix_rate, bool):
            if fix_rate:
                free_rate = np.zeros(self.Ne, dtype=bool)
            else:
                free_rate = np.ones(self.Ne, dtype=bool)
        elif isinstance(fix_rate, int):
            free_rate = np.ones(self.Ne, dtype=bool)
            if 0 <= fix_rate < self.Ne:
                free_rate[fix_rate] = False
        elif isinstance(fix_rate, (list, tuple, np.ndarray)):
            free_rate = np.ones(self.Ne, dtype=bool)
            for i in fix_rate:
                if 0 <= i < self.Ne:
                    free_rate[i] = False
        if isinstance(fix_coef, bool):
            if fix_coef:
                free_coef = np.zeros(self.Ne, dtype=bool)
            else:
                free_coef = np.ones(self.Ne, dtype=bool)
        elif isinstance(fix_coef, int):
            free_coef = np.ones(self.Ne, dtype=bool)
            if 0 <= fix_coef < self.Ne:
                free_coef[fix_coef] = False
        elif isinstance(fix_coef, (list, tuple, np.ndarray)):
            free_coef = np.ones(self.Ne, dtype=bool)
            for i in fix_coef:
                if 0 <= i < self.Ne:
                    free_coef[i] = False
        free = np.r_[free_phi, free_Ti, free_delay, free_bias, free_nlin,
                     free_tau, free_init, free_rate, free_coef, free_abs,
                     True]
        if not Ib_err:
            free[-1] = False
        self.free = free.nonzero()[0]
        t = self.t
        m = self.m
        s = self.s
        self.__initial_parameters = np.c_[low[self.free], self.par[self.free], high[self.free]]
        if verbose:
            print('Time')
            print(t)
            print('PMT current')
            print(m)
            print('Parameters')
            print('Phi', 'free_phi')
            print(np.c_[self.par[:k], free_phi])
            print('DTi', 'free_Ti')
            print(np.c_[self.par[k:2*k], free_Ti])
            print('delay', 'free_delay')
            print(np.c_[self.par[2*k:3*k], free_delay])
            print(free_delay)
            print('bias', 'free_bias')
            print(np.c_[self.par[3*k:4*k], free_bias])
            print('gain non linearity', 'free_nlin')
            print(np.c_[self.par[4*k:4*k+3], free_nlin])
            print('tau', 'free_tau')
            print(np.c_[self.par[4*k+3:4*k+3+self.Ne], free_tau])
            print('no', 'free_init')
            print(np.c_[self.par[4*k+3+self.Ne:4*k+3+2*self.Ne], free_init])
            print('n', 'free_rate')
            print(np.c_[self.par[4*k+3+2*self.Ne:4*k+3+3*self.Ne], free_rate])
            print('Dn', 'free_coef')
            print(np.c_[self.par[4*k+3+3*self.Ne:4*k+3+4*self.Ne], free_coef])
            print('absorption coefficient', 'free_abs')
            print(np.c_[self.par[-2], free_abs])
            print('PMT dark current')
            print(self.par[-1])
            print('Initial parameters with limits:')
            print(self.__initial_parameters)
        bounds = (low[self.free], high[self.free])
        kwargs = {}
        kwargs['ftol'] = self.ftol
        kwargs['xtol'] = self.xtol
        kwargs['gtol'] = self.gtol
        kwargs['max_nfev'] = self.max_nfev*len(self.free)
        try:
            popt, pcov = spopt.curve_fit(self.__afterglow, t, m, self.par[self.free], s[:d_idx], bounds=bounds, method='trf', **kwargs)
        except (ValueError, RuntimeError) as err:
            self.fitted = False
            self.Ne = 0
            self.Np = 0
            self.par = None
            self.perr = None
            self.free = None
            self.r = None
            self.t = None
            self.m = None
            self.f = None
            self.__tau_idx = None
            print(err)
            return
        self.fitted = True
        perr = np.sqrt(np.diag(pcov))
        self.par[self.free] = popt
        self.perr[self.free] = perr
#        tau_idx = self.par[4*k+3:4*k+3+self.Ne].argsort()
#        self.par[4*k+3:4*k+3+self.Ne] = self.par[4*k+3:4*k+3+self.Ne][tau_idx]
#        self.par[4*k+3+self.Ne:4*k+3+2*self.Ne] = self.par[4*k+3+self.Ne:4*k+3+2*self.Ne][tau_idx]
#        self.perr[4*k+3:4*k+3+self.Ne] = self.perr[4*k+3:4*k+3+self.Ne][tau_idx]
#        self.perr[4*k+3+self.Ne:4*k+3+2*self.Ne] = self.perr[4*k+3+self.Ne:4*k+3+2*self.Ne][tau_idx]
        self.__tau_idx = self.par[4*k+3:4*k+3+self.Ne].argsort()
        self.f = self.__afterglow(t, *list(self.par[self.free]))
        self.feval -= 1
        self.r = self.m-self.f

    def correct_fluxes(self, w):
        n = self.runs
        if n == 0:
            raise RuntimeError('Please load the measure data')
        if not self.fitted:
            raise RuntimeError('Please fit the measure data')
        Ne = self.Ne
        Phi = self.par[:n]*self.Phi_scale/np.array(w)
        Phi_n = np.array(self.Phi)/np.array(w)
        k = (Phi*Phi_n).sum()/np.square(Phi).sum()
        self.par[:n] *= k
        self.perr[:n] *= k
        self.par[4*n+3+2*Ne:4*n+3+3*Ne] /= k
        self.perr[4*n+3+2*Ne:4*n+3+3*Ne] /= k
        self.par[4*n+3+3*Ne:4*n+3+4*Ne] /= k
        self.perr[4*n+3+3*Ne:4*n+3+4*Ne] /= k
        return k

    def simulate(self, tau, n, bias=None, nlin=None,
                 Phi=None, DTi=None, delay=None, no=None, Dn=None,
                 absorp=None, Td=1e-6, a=1.0, Ib=None):
        if self.runs == 0:
            raise RuntimeError('Please load the measure data')
        k = self.runs
        if not isinstance(tau, (list, tuple, np.ndarray)):
            raise RuntimeError('tau must be a list, a tuple, or a np.ndarray')
        if not isinstance(n, (list, tuple, np.ndarray)):
            raise RuntimeError('n must be a list, a tuple, or a np.ndarray')
        if len(n) != len(tau):
            raise RuntimeError('len(n) must be equal to len(tau)')
        Ne = len(tau)
        self.sim_Ne = Ne
        if bias is not None:
            if not isinstance(bias, (list, tuple, np.ndarray)):
                raise RuntimeError('bias must be a list, a tuple, or a np.ndarray')
            if len(bias) != k:
                raise RuntimeError('len(bias) must equal the number of runs')
        if nlin is not None:
            if not isinstance(nlin, (list, tuple, np.ndarray)):
                raise RuntimeError('nlin must be a list, a tuple, or a np.ndarray')
            if len(nlin) != 3:
                raise RuntimeError('len(nlin) must equal three')
        if Phi is not None:
            if not isinstance(Phi, (list, tuple, np.ndarray)):
                raise RuntimeError('Phi must be a list, a tuple, or a np.ndarray')
            if len(Phi) != k:
                raise RuntimeError('len(Phi) must equal the number of runs')
        if DTi is not None:
            if not isinstance(DTi, (list, tuple, np.ndarray)):
                raise RuntimeError('Ti must be a list, a tuple, or a np.ndarray')
            if len(DTi) != k:
                raise RuntimeError('len(DTi) must equal the number of runs')
        if delay is not None:
            if not isinstance(delay, (list, tuple, np.ndarray)):
                raise RuntimeError('delay must be a list, a tuple, or a np.ndarray')
            if len(delay) != k:
                raise RuntimeError('len(delay) must equal the number of runs')
        if absorp is not None:
            try:
                absorp = float(absorp)
            except:
                raise RuntimeError('absorption must be convertible to an float')
        if no is not None:
            if not isinstance(no, (list, tuple, np.ndarray)):
                raise RuntimeError('no must be a list, a tuple, or a np.ndarray')
            if len(no) != len(tau):
                raise RuntimeError('len(no) must equal the len(tau)')
        if Dn is not None:
            if not isinstance(Dn, (list, tuple, np.ndarray)):
                raise RuntimeError('Dn must be a list, a tuple, or a np.ndarray')
            if len(Dn) != len(tau):
                raise RuntimeError('len(Dn) must equal the len(tau)')
        if Ib is not None:
            try:
                Ib = float(Ib)
            except:
                raise RuntimeError('Ib must be convertible to a float')
        t = self.get_time(active='all')
        self.sim_idx = []
        for Te in self.Te:
            i = (t > Te).nonzero()[0]
            if len(i) > 0:
                self.sim_idx.append(i[0])
            else:
                self.sim_idx.append(t.size)
        self.sim_idx.append(t.size)
        self.sim_par = np.zeros(4*k+3+4*Ne+2, dtype=float)
        if Phi is None:
            Phi = self.Phi
        Phi_min = np.max(Phi)
        for phi in Phi:
            if phi > 0. and phi < Phi_min:
                Phi_min = phi
        self.sim_Phi_scale = 10**int(np.log10(Phi_min))
        self.sim_par[:k] = np.array(Phi)/self.sim_Phi_scale
        if DTi is not None:
            self.sim_par[k:2*k] = DTi
        if delay is not None:
            self.sim_par[2*k:3*k] = delay
        if bias is not None:
            self.sim_par[3*k:4*k] = bias
        else:
            self.sim_par[3*k:4*k] = self.V
        if nlin is not None:
            self.sim_par[4*k:4*k+3] = nlin
        else:
            self.sim_par[4*k:4*k+3] = (self.d, self.Io, self.sf)
        self.sim_par[4*k+3:4*k+3+Ne] = tau
        if no is not None:
            self.sim_par[4*k+3+Ne:4*k+3+2*Ne] = no
        self.sim_par[4*k+3+2*Ne:4*k+3+3*Ne] = n
        if Dn is not None:
            self.sim_par[4*k+3+3*Ne:4*k+3+4*Ne] = Dn
        if absorp is not None:
            self.sim_par[-2] = absorp
        if Ib is not None:
            self.sim_par[-1] = Ib
        y = self.__afterglow(t, *[])
        self.feval -= 1
        l = 0
        for i, run in enumerate(self.data):
            for j, data in enumerate(run):
                m = l+len(data['t'])
                self.data[i][j]['s'] = {}
                if data['rms_par'] is None:
                    y_err = np.zeros(m-l, dtype=float)
                else:
                    y_err = self.__rms_func(y[l:m]-data['rms_par'][1], *data['rms_par'][0])*np.random.randn(m-l)
                self.data[i][j]['s']['i'] = y[l:m]+y_err
                self.data[i][j]['s']['t'] = y[l:m]
#                self.filter_data(i, j, 's')
                l = m
        self.simulated = True

    def plot_fit(self, title=None, band=1., unit='nA', calc=False, log=0, rlog=False):
        if self.runs == 0:
            raise RuntimeError('Please load the measure data')
        if not self.fitted:
            raise RuntimeError('Please fit the measure data')
        if log < 0 or log > 3:
            raise RuntimeError('log must have a value between 0 and 3')
        if (calc):
            self.f = self.__afterglow(self.t, *list(self.par[self.free]))
            self.r = self.m-self.f
        plog = log if rlog else log & 1
        gs_kw = dict(height_ratios=[8, 2])
        plt.rcParams.update({'font.size': 14})
        fig, f_ax = plt.subplots(2, 1, figsize=(10,8), constrained_layout=True, gridspec_kw=gs_kw)
        if log == 0:
            f_ax[0].plot()
        elif log == 1:
            f_ax[0].semilogx()
        elif log == 2:
            f_ax[0].semilogy()
        else:
            f_ax[0].loglog()
        i = 0
        for n in self.__active:
            for d in self.data[n]:
                j = i+d['t'].size
                t = self.t[i:j]
                m = self.m[i:j]
                f = self.f[i:j]
                f_ax[0].scatter(t, m, color='C0', s=0.5)
                f_ax[0].plot(t, f, 'C1', label='fit')
                i = j
        f_ax[0].set_ylabel('current [{}]'.format(unit))
        f_ax[0].grid(True, 'both')
        if isinstance(title, str):
            f_ax[0].set_title(title)
        if plog == 0:
            f_ax[1].plot()
        elif plog == 1:
            f_ax[1].semilogx()
        elif plog == 2:
            f_ax[1].semilogy()
        else:
            f_ax[1].loglog()
        i = 0
        for n in self.__active:
            for d in self.data[n]:
                j = i+d['t'].size
                t = self.t[i:j]
                r = self.r[i:j]
                if self.simulated:
                    true = d['s']['t']-self.f[i:j]
                f_ax[1].scatter(t, r if plog < 2 else np.abs(r), color='C0', s=0.5)
                if self.simulated:
                    f_ax[1].plot(t, true if plog < 2 else np.abs(true), 'k')
                i = j
        yl = f_ax[1].get_ylim()
        i = 0
        if band > 0.:
            for n in self.__active:
                for d in self.data[n]:
                    j = i+d['t'].size
                    t = self.t[i:j]
                    s = band*self.s[i:j]
                    f_ax[1].fill_between(t, -s, s, color='b', alpha=.1)
                    i = j
        f_ax[1].set_ylim(yl)
        f_ax[1].set_ylabel('residual')
        f_ax[1].set_xlabel('time [s]')
        f_ax[1].grid(True, 'both')

    def get_fit_initial_parameters(self):
        if self.runs == 0:
            raise RuntimeError('Please load the measure data')
        if not self.fitted:
            raise RuntimeError('Please fit the measure data')
        ret_val = np.empty_like(self.__initial_parameters)
        ret_val[:] = self.__initial_parameters
        return ret_val
        
    def get_fit_bias(self):
        if self.runs == 0:
            raise RuntimeError('Please load the measure data')
        if not self.fitted:
            raise RuntimeError('Please fit the measure data')
        n = self.runs
        ret_val = np.empty((n, 2), dtype=float)
        ret_val[:, 0] = self.par[3*n:4*n]
        ret_val[:, 1] = self.perr[3*n:4*n]
        return ret_val

    def get_fit_nonlinearity_parameters(self):
        if self.runs == 0:
            raise RuntimeError('Please load the measure data')
        if not self.fitted:
            raise RuntimeError('Please fit the measure data')
        n = self.runs
        ret_val = np.empty((3, 2), dtype=float)
        ret_val[:, 0] = self.par[4*n:4*n+3]
        ret_val[:, 1] = self.perr[4*n:4*n+3]
        return ret_val

    def get_fit_Phi(self):
        if self.runs == 0:
            raise RuntimeError('Please load the measure data')
        if not self.fitted:
            raise RuntimeError('Please fit the measure data')
        n = self.runs
        ret_val = np.empty((n, 2), dtype=float)
        ret_val[:, 0] = self.Phi_scale*self.par[:n]
        ret_val[:, 1] = self.Phi_scale*self.perr[:n]
        return ret_val

    def get_fit_DTi(self):
        if self.runs == 0:
            raise RuntimeError('Please load the measure data')
        if not self.fitted:
            raise RuntimeError('Please fit the measure data')
        n = self.runs
        ret_val = np.empty((n, 2), dtype=float)
        ret_val[:, 0] = self.par[n:2*n]
        ret_val[:, 1] = self.perr[n:2*n]
        return ret_val

    def get_fit_delay(self):
        if self.runs == 0:
            raise RuntimeError('Please load the measure data')
        if not self.fitted:
            raise RuntimeError('Please fit the measure data')
        n = self.runs
        ret_val = np.empty((n, 2), dtype=float)
        ret_val[:, 0] = self.par[2*n:3*n]
        ret_val[:, 1] = self.perr[2*n:3*n]
        return ret_val

    def get_fit_tau(self, sorted=None):
        if self.runs == 0:
            raise RuntimeError('Please load the measure data')
        if not self.fitted:
            raise RuntimeError('Please fit the measure data')
        n = self.runs
        Ne = self.Ne
        if not isinstance(sorted, bool):
            sorted = self.__sort_trap_data
        ret_val = np.empty((Ne, 2), dtype=float)
        if sorted:
            ret_val[:, 0] = self.par[4*n+3:4*n+3+Ne][self.__tau_idx]
            ret_val[:, 1] = self.perr[4*n+3:4*n+3+Ne][self.__tau_idx]
        else:
            ret_val[:, 0] = self.par[4*n+3:4*n+3+Ne]
            ret_val[:, 1] = self.perr[4*n+3:4*n+3+Ne]
        return ret_val

    def get_fit_no(self, sorted=None):
        if self.runs == 0:
            raise RuntimeError('Please load the measure data')
        if not self.fitted:
            raise RuntimeError('Please fit the measure data')
        n = self.runs
        Ne = self.Ne
        if not isinstance(sorted, bool):
            sorted = self.__sort_trap_data
        ret_val = np.empty((Ne, 2), dtype=float)
        if sorted:
            ret_val[:, 0] = self.par[4*n+3+Ne:4*n+3+2*Ne][self.__tau_idx]
            ret_val[:, 1] = self.perr[4*n+3+Ne:4*n+3+2*Ne][self.__tau_idx]
        else:
            ret_val[:, 0] = self.par[4*n+3+Ne:4*n+3+2*Ne]
            ret_val[:, 1] = self.perr[4*n+3+Ne:4*n+3+2*Ne]
        return ret_val

    def get_fit_n(self, sorted=None):
        if self.runs == 0:
            raise RuntimeError('Please load the measure data')
        if not self.fitted:
            raise RuntimeError('Please fit the measure data')
        n = self.runs
        Ne = self.Ne
        if not isinstance(sorted, bool):
            sorted = self.__sort_trap_data
        ret_val = np.empty((Ne, 2), dtype=float)
        if sorted:
            ret_val[:, 0] = self.par[4*n+3+2*Ne:4*n+3+3*Ne][self.__tau_idx]
            ret_val[:, 1] = self.perr[4*n+3+2*Ne:4*n+3+3*Ne][self.__tau_idx]
        else:
            ret_val[:, 0] = self.par[4*n+3+2*Ne:4*n+3+3*Ne]
            ret_val[:, 1] = self.perr[4*n+3+2*Ne:4*n+3+3*Ne]
        return ret_val

    def get_fit_Dn(self, sorted=None):
        if self.runs == 0:
            raise RuntimeError('Please load the measure data')
        if not self.fitted:
            raise RuntimeError('Please fit the measure data')
        n = self.runs
        Ne = self.Ne
        if not isinstance(sorted, bool):
            sorted = self.__sort_trap_data
        ret_val = np.empty((Ne, 2), dtype=float)
        if sorted:
            ret_val[:, 0] = self.par[4*n+3+3*Ne:4*n+3+4*Ne][self.__tau_idx]
            ret_val[:, 1] = self.perr[4*n+3+3*Ne:4*n+3+4*Ne][self.__tau_idx]
        else:
            ret_val[:, 0] = self.par[4*n+3+3*Ne:4*n+3+4*Ne]
            ret_val[:, 1] = self.perr[4*n+3+3*Ne:4*n+3+4*Ne]
        return ret_val

    def get_fit_absorption(self):
        if self.runs == 0:
            raise RuntimeError('Please load the measure data')
        if not self.fitted:
            raise RuntimeError('Please fit the measure data')
        ret_val = np.empty((1, 2), dtype=float)
        ret_val[0, 0] = self.par[-2]
        ret_val[0, 1] = self.perr[-2]
        return ret_val

    def get_fit_Ib(self):
        if self.runs == 0:
            raise RuntimeError('Please load the measure data')
        if not self.fitted:
            raise RuntimeError('Please fit the measure data')
        ret_val = np.empty((1, 2), dtype=float)
        ret_val[0, 0] = self.par[-1]
        ret_val[0, 1] = self.perr[-1]
        return ret_val

    def get_time(self, r=None, m=None, active=None):
        n = self.runs
        if n == 0:
            raise RuntimeError('Please load the measure data')
        try:
            r = int(r)
        except:
            r = None
        try:
            m = int(m)
        except:
            m = None
        if r is None:
            if active is None:
                a = self.__active
            elif active == 'all':
                a = np.arange(self.runs, dtype=int)
            elif isinstance(active, (list, tuple, np.ndarray)):
                a = np.unique(active)
                if (a < 0).any() or (a > self.runs-1).any():
                    raise RuntimeError('Wrong run number')
            else:
                raise RuntimeError('active must be a list, a tuple, or a np.ndarray')
            times = np.zeros(0, dtype=float)
            for i in a:
                for d in self.data[i]:
                    times = np.r_[times, d['t']]
        elif -1 < r < n:
            if m is None:
                times = np.zeros(0, dtype=float)
                for d in self.data[r]:
                    times = np.r_[times, d['t']]
            elif -1 < m < len(self.data[r]):
                return self.data[r][m]['t']
            else:
                raise RuntimeError('Wrong measure number')
        else:
            raise RuntimeError('Wrong run number')
        return times.copy()

    def get_data(self, dset='m', which='i', r=None, m=None, active=None):
        n = self.runs
        if n == 0:
            raise RuntimeError('Please load the measure data')
        try:
            r = int(r)
        except:
            r = None
        try:
            m = int(m)
        except:
            m = None
        if r is None:
            if active is None:
                a = self.__active
            elif active == 'all':
                a = np.arange(self.runs, dtype=int)
            elif isinstance(active, (list, tuple, np.ndarray)):
                a = np.unique(active)
                if (a < 0).any() or (a > self.runs-1).any():
                    raise RuntimeError('Wrong run number')
            else:
                raise RuntimeError('active must be a list, a tuple, or a np.ndarray')
            data = np.zeros(0, dtype=float)
            for i in a:
                for d in self.data[i]:
                    data = np.r_[data, d[dset][which]]
        elif -1 < r < n:
            if m is None:
                data = np.zeros(0, dtype=float)
                for d in self.data[r]:
                    data = np.r_[data, d[dset][which]]
            elif -1 < m < len(self.data[r]):
                return self.data[r][m][dset][which]
            else:
                raise RuntimeError('Wrong measure number')
        else:
            raise RuntimeError('Wrong run number')
        return data.copy()

    def get_rms(self, r=None, m=None):
        n = self.runs
        if n == 0:
            raise RuntimeError('Please load the measure data')
        if not self.fitted:
            raise RuntimeError('Please fit the measure data')
        try:
            r = int(r)
        except:
            r = None
        try:
            m = int(m)
        except:
            m = None
        if r is None:
            data = self.s.copy()
        elif r in self.__active:
            data = self.s[self.idx[r]:self.idx[r+1]]
            if m is not None:
                if m < len(self.data[r]):
                    skip = 0
                    for i in range(m):
                        skip += len(self.data[r][i]['m']['i'])
                    keep = len(self.data[r][m]['m']['i'])
                    data = data[skip:skip+keep].copy()
                else:
                    raise RuntimeError('Wrong measure number')
        else:
            raise RuntimeError('Wrong run number')
        return data.copy()

    def get_fit(self, r=None, m=None):
        n = self.runs
        if n == 0:
            raise RuntimeError('Please load the measure data')
        if not self.fitted:
            raise RuntimeError('Please fit the measure data')
        try:
            r = int(r)
        except:
            r = None
        try:
            m = int(m)
        except:
            m = None
        if r is None:
            data = self.f.copy()
        elif r in self.__active:
            data = self.f[self.idx[r]:self.idx[r+1]]
            if m is not None:
                if m < len(self.data[r]):
                    skip = 0
                    for i in range(m):
                        skip += len(self.data[r][i]['m']['i'])
                    keep = len(self.data[r][m]['m']['i'])
                    data = data[skip:skip+keep].copy()
                else:
                    raise RuntimeError('Wrong measure number')
        else:
            raise RuntimeError('Wrong run number')
        return data.copy()

    def get_residuals(self, r=None, m=None):
        n = self.runs
        if n == 0:
            raise RuntimeError('Please load the measure data')
        if not self.fitted:
            raise RuntimeError('Please fit the measure data')
        try:
            r = int(r)
        except:
            r = None
        try:
            m = int(m)
        except:
            m = None
        if r is None:
            data = self.r.copy()
        elif r in self.__active:
            data = self.r[self.idx[r]:self.idx[r+1]]
            if m is not None:
                if m < len(self.data[r]):
                    skip = 0
                    for i in range(m):
                        skip += len(self.data[r][i]['m']['i'])
                    keep = len(self.data[r][m]['m']['i'])
                    data = data[skip:skip+keep].copy()
                else:
                    raise RuntimeError('Wrong measure number')
        else:
            raise RuntimeError('Wrong run number')
        return data.copy()
