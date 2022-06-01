### Package implementing the formalism of Honeychurch et al, 2019
#author: jks
#v3: implemented FFT, and proper finite-difference differentiation

import numpy as np
from scipy.special import jv,jvp
import scipy as scipy
from numpy.polynomial.legendre import leggauss
from numpy.linalg import inv
from scipy.integrate import quad
#from numpy.polynomial.chebyshev import Chebyshev as cheb
from scipy.linalg import expm
#import numpy.polynomial.chebyshev
import math
import time
import PadeDecomp
from numba import njit, prange, jit
from scipy.interpolate import CubicSpline
#matrix structure of internal variables: #aux, aux , ... T, omega, [device space matrix]
import os, psutil
process = psutil.Process(os.getpid())
##print(process.memory_info().rss/2**30)  # in GB
class timescale:
    Gamma_L =0
    Gamma_R =0
    H = 0
    H_dT = 0
    mu_L = 0
    Temperature = 0
    mu_R = 0
    Delta_L = 0
    Delta_R = 0
    Delta = 0
    device_dim = 0

    #variables
    T = np.array([0]).reshape(-1,1,1,1)
    omega = np.array([0]).reshape(-1,1,1)
    omega_weights =np.array([0]).reshape(-1,1,1)

    tau = 0
    tau_weights = 0

    Vbias = lambda t:0
    Vbias_dT = lambda t:0

    #self-energies
    Sigma_R_less  = 0
    Sigma_L_less  = 0
    Sigma_L_R  = 0
    Sigma_L_A  = 0
    Sigma_R_R  = 0
    Sigma_R_A  = 0
    Sigma_L_less_dT  = 0
    Sigma_R_less_dT  = 0
    Sigma_L_R_dT  = 0
    Sigma_L_A_dT  = 0
    Sigma_R_R_dT  = 0
    Sigma_R_A_dT  = 0
    Sigma_L_less_dw  = 0
    Sigma_R_less_dw  = 0
    Sigma_L_R_dw  = 0
    Sigma_L_A_dw  = 0
    Sigma_R_R_dw  = 0
    Sigma_R_A_dw  = 0
    Sigma_R  = 0
    Sigma_A  = 0
    Sigma_less  = 0
    Sigma_less_dT  = 0
    Sigma_less_dw  = 0
    Sigma_R_dT  = 0
    Sigma_A_dT  = 0
    Sigma_R_dw = 0
    Sigma_A_dw = 0

    G0_R = 0
    G0_A = 0
    G0_less = 0 
    G0_R_dT = 0
    G0_R_dw = 0 
    G0_A_dT = 0
    G0_A_dw = 0 
    G0_less_dT = 0 
    G0_less_dw = 0 
    G1_R = 0 
    G1_A = 0
    G1_less = 0 

    expint_R = None
    expint_L = None

    def __init__(self,T=None,omega_params=None,Gamma_L=(lambda x : np.eye(2)),Gamma_R=(lambda x : np.eye(2)),H=None,
                 Temperature=0.1,mu_L = 0, mu_R = 0,Delta_L = 0, Delta_R = 0, Delta = None,
                 Vbias=(lambda t : 0),Vbias_dT=None,Vbias_int=None,
                 use_aux_modes=False,Fermi_params=[],Lorentz_params_L=[],Lorentz_params_R=[],eta=None):
        #T: np.array of times
        #omega: list or nparray with syntax [omega_min, omega_max, N_omega]
        if Delta is None:
            print('error in __init__: Delta must be specified')
            assert 1==0
        device_dim = np.shape(Delta)[-1]
        self.device_dim = device_dim
        self.Delta_L = Delta_L 
        self.Delta_R = Delta_R 
        self.Delta = Delta

        self.Gamma_L = Gamma_L
        self.Gamma_R = Gamma_R
        if callable(Gamma_L):
            self.WBL_L = False
        else:
            self.WBL_L = True
        if callable(Gamma_R):
            self.WBL_R = False
        else:
            self.WBL_R = True

        self.Temperature = Temperature 
        self.mu_L = mu_L 
        self.mu_R = mu_R 
        

        T=np.array(T).reshape(-1,1,1,1)
        self.T=T
        if H is not None:
            self.H = H + Delta*Vbias(T)
            self.f_H = lambda t : (H + Delta*Vbias(t))
        if Vbias_dT is None: #Vbias_dT may be optionally specifieself. If it is not, it is calculated here w/ finite difference.
            if np.all(Delta == 0):
                Vbias_dT = lambda t : 0
            else:
                Vbias_dT = lambda t : self.FD(Vbias,t) #numerical derivative w/ finite difference
        self.H_dT = Delta*Vbias_dT(T)

        self.Vbias = Vbias
        self.Vbias_dT = Vbias_dT
        self.Vbias_int = Vbias_int
        self.use_aux_modes = use_aux_modes
        self.Fermi_params = Fermi_params
        self.Lorentz_params_L = Lorentz_params_L
        self.Lorentz_params_R = Lorentz_params_R


        if omega_params is not None: 
            omega_min, omega_max, N_omega = omega_params
            #omega, omega_weights = self.get_integration_var(N_omega,omega_min,omega_max)
            omega = np.linspace(omega_min,omega_max,N_omega)
            omega_weights = np.ones(N_omega)*(omega[1]-omega[0])
            omega_weights[0] = omega_weights[0]/2
            omega_weights[-1] = omega_weights[-1]/2
                #                     T, w  matrix
            self.omega = omega.reshape(1,-1,   1,1)
            self.omega_weights = omega_weights.reshape(1,-1,1,1)
            self.dw = (omega_max - omega_min)/(N_omega -1)
            if eta is None:
                eta = self.dw
            self.eta = eta



    def get_physical_params(self,Test): #Gets physical parameters (hamiltonian, chemical potentials, temperature, self-energy etc) from the TimedependentTransport object Test
        print('Running get_physical_params. Remember Delta_R, Delta_L, Delta and Vbias must be specified manually')
        Temperature_L, Temperature_R = Test.kT_i
        self.Temperature = Temperature_L
        if Temperature_L != Temperature_R:
            print('Error in get_physical_params: Electrodes must have same temperature')
            assert 1==0
            return
        if len(Test.kT_i) != 2:
           print('Error in get_physical_params: System must contain exactly 2 electrodes')
           assert 1==0
           return
        mu_L, mu_R = Test.mu_i
        self.mu_L = mu_L
        self.mu_R = mu_R
        self.H = Test.Hdense[0,0] + self.potential(self.T)
        self.device_dim = np.shape(self.H)[-1]
        self.f_H = lambda t : (Test.Hdense[0,0] + self.potential(t))


        if self.use_aux_modes:
            self.get_Lorentz_and_Fermi_params(Test)
        else: #get self-energy from TBTrans (stored in Test)
            sidx = Test.sampling_idx[0]
            TBT_Gam_sparse_L = Test.Nonortho_Gammas[0].get_e_subset(sidx)
            E  = Test.Contour[sidx]
            if len(E) == 1: #spline interpolation cannot be based on one point; use Lorentz instead
                print('Warning: only one point in contour. Using Lorentzian expansion of self energies.')
                self.get_Lorentz_and_Fermi_params(Test)
                return
            TBT_Gam_L = Test.bs2np(TBT_Gam_sparse_L)[0]
            self.coupling_index_L = (np.sum(np.abs(TBT_Gam_L),axis=0) != 0) #array of size nxn containing True if site n couples to left lead, and no if not.
            TBT_Gam_sparse_R = Test.Nonortho_Gammas[1].get_e_subset(sidx)
            TBT_Gam_R = Test.bs2np(TBT_Gam_sparse_R)[0]
            self.coupling_index_R = (np.sum(np.abs(TBT_Gam_R),axis=0) != 0) #array of size nxn containing True if site n couples to left lead, and no if not.
            def Gamma_L(eps):
                cs = CubicSpline(E.real,TBT_Gam_L,extrapolate=False)
                eps = eps.reshape(1,-1)
                res = cs(eps)
                res[np.isnan(res)] = 0
                return res + self.eta*self.coupling_index_L

            def Gamma_R(eps):
                cs = CubicSpline(E.real,TBT_Gam_R,extrapolate=False)
                eps = eps.reshape(1,-1)
                res = cs(eps)
                res[np.isnan(res)] = 0
                return res + self.eta*self.coupling_index_R

            self.Gamma_L = Gamma_L
            self.Gamma_R = Gamma_R




    def get_Lorentz_and_Fermi_params(self,Test): #Test is a TimeDependentTransport object
        Lorentz_L = Test.fitted_lorentzians[0]
        Lorentz_L_np = Test.bs2np(Lorentz_L)[0]
        self.coupling_index_L = (np.sum(np.abs(Lorentz_L_np),axis=0) != 0) #array of size nxn containing True if site n couples to left lead, and no if not.
        ei_L,wi_L = Lorentz_L.ei, Lorentz_L.gamma
        ei_L = ei_L.reshape(-1,1,1)
        wi_L = wi_L.reshape(-1,1,1)

        Lorentz_R = Test.fitted_lorentzians[1]
        Lorentz_R_np = Test.bs2np(Lorentz_R)[0]
        self.coupling_index_R = (np.sum(np.abs(Lorentz_R_np),axis=0) != 0) #array of size nxn containing True if site n couples to left lead, and no if not.
        ei_R,wi_R = Lorentz_R.ei, Lorentz_R.gamma
        ei_R = ei_R.reshape(-1,1,1)
        wi_R = wi_R.reshape(-1,1,1)

        Lorentz_poles_L = np.concatenate((ei_L + 1j*wi_L,ei_L - 1j*wi_L))
        Lorentz_ccoefs_L = np.concatenate((wi_L**2*Lorentz_L_np/(2j*wi_L),wi_L**2*Lorentz_L_np/(-2j*wi_L)))
        if self.eta != 0:
            #add extra, very wide Lorentzian to avoid numerical issues. 
            #currently the width is set to 100 times the frequency range, and is centered at the fermi energy
            W = 100*(self.omega.max() - self.omega.min())
            Wide_Lor_poles = np.array([self.mu_L + 1j*W, self.mu_L - 1j*W]).reshape(-1,1,1)
            Wide_Lor_ccoefs = np.array([W*self.eta*self.coupling_index_L/(2j),W*self.eta*self.coupling_index_L/(-2j)])
            Lorentz_poles_L = np.concatenate((Lorentz_poles_L,Wide_Lor_poles))
            Lorentz_ccoefs_L = np.concatenate((Lorentz_ccoefs_L,Wide_Lor_ccoefs))
        self.Lorentz_params_L = (Lorentz_poles_L, Lorentz_ccoefs_L)

        Lorentz_poles_R = np.concatenate((ei_R + 1j*wi_R,ei_R - 1j*wi_R))
        Lorentz_ccoefs_R = np.concatenate((wi_R**2*Lorentz_R_np/(2j*wi_R),wi_R**2*Lorentz_R_np/(-2j*wi_R)))
        if self.eta != 0:
            #add extra, very wide Lorentzian to avoid numerical issues. 
            #currently the width is set to 100 times the frequency range, and is centered at the fermi energy
            W = 100*(self.omega.max() - self.omega.min())
            Wide_Lor_poles = np.array([self.mu_R + 1j*W, self.mu_R - 1j*W]).reshape(-1,1,1)
            Wide_Lor_ccoefs = np.array([W*self.eta*self.coupling_index_R/(2j),W*self.eta*self.coupling_index_R/(-2j)])
            Lorentz_poles_R = np.concatenate((Lorentz_poles_R,Wide_Lor_poles))
            Lorentz_ccoefs_R = np.concatenate((Lorentz_ccoefs_R,Wide_Lor_ccoefs))
        self.Lorentz_params_R = (Lorentz_poles_R, Lorentz_ccoefs_R)


        def Gamma_L(eps):
            roots,complex_coefs = self.Lorentz_params_L
            NL = len(roots)
            res = 0
            for i in range(NL):
                res = res + complex_coefs[i]/(eps - roots[i])# - complex_coefs[i]/(eps - np.conjugate(roots[i]))
            return res
        def Gamma_R(eps):
            roots,complex_coefs = self.Lorentz_params_R
            NL = len(roots)
            res = 0
            for i in range(NL):
                res = res + complex_coefs[i]/(eps - roots[i])# - complex_coefs[i]/(eps - np.conjugate(roots[i]))
            return res
        self.Gamma_L = Gamma_L
        self.Gamma_R = Gamma_R

        PDroots, PDcoefs = PadeDecomp.Hu_poles(Test.num_poles)
        PDroots = np.concatenate((PDroots,PDroots.conjugate())) #the function returns only roots w/ pos. imaginary part. 
        PDcoefs = np.concatenate((PDcoefs,PDcoefs))
        self.Fermi_params = (PDroots*self.Temperature, -PDcoefs*self.Temperature)
        #print('obtained Lorentz and fermi params',flush=True)



    def Fermi_pade(self,eps,alpha='L'):
        #eps=eps.reshape(-1,1)
        if alpha=='L':
            mu = self.mu_L
        elif alpha=='R':
            mu = self.mu_R
        else: 
            print('error in Fermi_pade: alpha must be specified')
            assert 1==0
        Fermi_poles, Fermi_res = self.Fermi_params
        res = 1/2
        for i in range(len(Fermi_poles)):
            res = res + Fermi_res[i]/(eps-mu-Fermi_poles[i])
        return res

    def fermi(self,eps,alpha): #fermi-dirac distribution. 
        if alpha == 'L':
            mu = self.mu_L
        elif alpha == 'R':
            mu = self.mu_R
        else: 
            print('error in timescale.fermi: alpha must be either L or R')
            assert 1==0
        return self.exp(-(eps-mu)/self.Temperature)/(1+self.exp(-(eps-mu)/self.Temperature))

    def f_L(self,eps):
        return self.fermi(eps,alpha='L')
    def f_R(self,eps):
        return self.fermi(eps,alpha='R')

    def potential_L(self,T): #Vbias is a function with syntax Vbias(T) that returns the relevant bias.
        bias = self.Vbias(T)
        return self.Delta_L*bias

    def potential_R(self,T):
        bias = self.Vbias(T)
        return self.Delta_R*bias

    def potential(self,T,alpha=0):
        bias = self.Vbias(T)
        if alpha=='L':
            return self.Delta_L*bias
        elif alpha=='R':
            return self.Delta_R * bias
        else:
            return self.Delta*bias

    def potential_dT(self,T,alpha=0):
        bias_dT = self.Vbias_dT(T)
        if alpha=='L':
            return self.Delta_L*bias_dT
        elif alpha=='R':
            return self.Delta_R * bias_dT
        else:
            return self.Delta*bias_dT


    @staticmethod
    def exp(x): #exponential function that does not return inf for large input values
        if np.any(np.iscomplex(x)):
            if hasattr(x, '__iter__'):
                x[np.real(x)>709] = 709 + 1j*np.imag(x[np.real(x)>709])
                #x[x<-710]00 = - 710
            else: 
                x = 709 + 1j*np.imag(x) if np.real(x)>709 else x
                #1
                #x = -700 if x<-700 else x
        else:
            if hasattr(x, '__iter__'):
                x[x>709] = 709
                #x[x<-710]00 = - 710
            else: 
                x = 709 if x>709 else x
                #x = -700 if x<-700 else x
        return np.exp(x)

    @staticmethod
    def get_integration_var(N_int,lower_lim = 0, upper_lim = 1):
        #function to get N_int gauss-legendre quadrature points and weights in the specified interval
        x,w = leggauss(N_int)
        x = (x + 1)/2*(upper_lim-lower_lim) + lower_lim
        w *= (upper_lim - lower_lim)/2
        return x,w

    def int_residues2(self,function, params,halfplane='upper'): 
        #version 2 - handles complex functions. however, now residues and poles in the lower half plane must be explicitly passeself.
        # uses the residue theorem to calculate an integral from -infty to +infty of the function 'function(x) * g(x)'
        #where 'function' is assumed not to contain poles. it is assumed there are no poles on the real axis.
        #The function g(x) is specified in the form (poles, residues).
        poles, residues = params
        if not hasattr(poles, '__iter__'): #put poles in a list so the syntax of the following lines will work.
            poles = np.array(poles)
            residues = np.array(residues)
        if np.any(poles.imag == 0):
            print('error in int_residues2: pole on real axis. saved list of poles as int_residues2_error_poles.npy',flush=True)
            np.save('int_residues2_error_poles',poles)
            assert 1==0
        if halfplane == 'upper':
            indices = np.imag(poles)>0
            poles = poles[indices]
            residues = residues[indices.flatten()]
        elif halfplane == 'lower': 
            indices = np.imag(poles)<0
            poles = poles[indices]
            residues = residues[indices.flatten()]
        else:
            print('error in int_residues2: half-plane improperly specified',flush=True)
            assert 1==0
        res = 0
        for i in range(len(poles)):
            res = res + function(poles[i]) * residues[i]
        res = res*1j*2*np.pi
        if halfplane =='lower':
            res = - res #switch sign to account for the contour having been traversed in the clockwise direction
        return res




    def expint3(self,T,tau,alpha): #T is Nx1, tau is 1xM. alpha is a char specifying the left lead, right lead or central region.
        #Suppose the support of the pulse is contained in the interval [Tmin, Tmax].
        t0=time.time()

        #set the correct potential
        if alpha == 'L':
            pot=self.potential_L
        elif alpha =='R':
            pot = self.potential_R
        else:
            pot = self.potential
            #print('error in expint3: alpha not ==R or L. This should be specified at this point!')
            assert 1==0

        #check if a new calculation is necessary
        if np.array_equal(T,self.T) and np.array_equal(tau,self.tau):
            if alpha == 'L':
                if not np.all(self.expint_L==None):
                    #print('expint3: returning stored value, alpha == L')
                    return self.expint_L #return alreay calculated value; otherwise proceeself.
            elif alpha=='R':
                if not np.all(self.expint_R==None):
                    #print('expint3: returning stored value, alpha == R')
                    return self.expint_R

        n_int=5
        x,w = self.get_integration_var(n_int)

        def int_gauss(f,tmin,tmax,N=n_int):
            t = tmin + x*(tmax-tmin)
            q = w*(tmax-tmin)
            res=0
            for i in range(N):
                res += f(t[i])*q[i]
            return res
        def antiderivative(t,t0=None): #integrates the function pot and gives the antiderivative evaluated as F(t) - F(t0)
            F=[]
            Fcum=0
            if t0 is None:
                t0 = t[0]
            for tt in t:
                Fcum += int_gauss(pot,t0,tt)
                F.append(Fcum)
                t0=tt
            F=np.array(F)
            return F
        T1=T.flatten()
        tau1=tau.flatten()
        F = np.zeros((np.size(T),np.size(tau)))
        for i, TT in enumerate(T1):
            #for every t, calc the integral with bounds T-tau/2 .. T+tau/2 for every value of tau.
            #this is done by calculating the antiderivative for every value of T+tau/2 and every value T-tau/2, then subtracting the two.
                tlist1 = TT+tau1/2
                tlist2 = TT-tau1/2
                F1 = antiderivative(tlist1)
                F2 = antiderivative(tlist2)
                t0_correction = 0
                if tlist1[0] != tlist2[0]: #t0 is not the same in the two derivatives, leading to constant offset. this should fix
                    t01 = tlist1[0]
                    t02 = tlist2[0]
                    t0_correction = quad(pot,t01,t02)[0] #integrate with scipy
                F[i]=F1-F2-t0_correction
        #print('calculated integral of Vbias in %.2f'%(time.time()-t0),flush=True)

        F=F.reshape(np.size(T),np.size(tau),1,1)
        if np.array_equal(T,self.T) and np.array_equal(tau,self.tau):
            if alpha == 'L':
                self.expint_L = F
                #print('setting self.expint_L = F')
            elif alpha=='R':
                self.expint_R = F
                #print('setting self.expint_R = F')
        return F


    @staticmethod #n'th order derivative, Fself.
    def FD(f,x,n=1,dx=1e-3,args=(),order=3): #f a callable function
        if order <=n:
            order = n + 1 #at least n+1 points must be used to calculate n'th derivative
            order = order + (1-order %2) #an odd-number of points must be used to calculate the derivative
        fd = scipy.misc.derivative(f,x,dx,n,args=args,order=order)
        return fd



    @staticmethod
    def fft(array,axis=-1):
        #return np.fft.fft(array,axis)
        return np.fft.fftshift(np.fft.fft(array,axis=axis),axes=axis)

    @staticmethod
    def ifft(array,axis=-1):
        #return np.fft.ifft(array,axis)
        return np.fft.ifft(np.fft.ifftshift(array,axes=axis),axis=axis)

    @staticmethod
    def fftfreq(n,d):
     #return np.fft.fftfreq(n,d)
        return np.fft.fftshift(np.fft.fftfreq(n,d))

    def calc_Sigma_less(self,T=None,omega=None,alpha='L',extension_length=50):
        if np.all(T==None):
            T=self.T
            #print('calc_Sigma_less: set T = self.T',flush=True)
        if np.all(omega==None):
            omega = self.omega
            set_self_tau = 1
        else: 
            set_self_tau = 0
        T=np.array(T).reshape(-1,1,1,1)
        omega=np.array(omega).reshape(-1,1,1)   
        exp = np.exp
        omega_min = omega.min()
        omega_max = omega.max()
        N = np.size(omega)
        if extension_length < N/4:
            extension_length = int(N/4)
        #print('extending internal arrays by ',extension_length)
        dw = (omega_max - omega_min)/(np.size(omega)-1)
        dnu = dw/(2*np.pi)
        extension = dw*np.linspace(1,extension_length,extension_length)
        w = np.concatenate(((omega_min - np.flip(extension)),omega.flatten(),omega_max + extension))
        w = w.reshape(1,-1,1,1)
        tau = self.fftfreq(np.size(w),d=dnu).reshape(1,-1,1,1) #shape: T, tau, n, n
        if set_self_tau:
            self.tau = tau
        dtau = (np.max(tau)-np.min(tau))/(np.size(tau)-1)
        if alpha=='L':
            f = self.f_L
            Gamma = self.Gamma_L
            potential = self.potential_L
            Lorentz_params = self.Lorentz_params_L
            coupling_index = self.coupling_index_L
            Delta_alpha = self.Delta_L
        else:
            f = self.f_R
            Gamma = self.Gamma_R
            potential = self.potential_R
            Lorentz_params = self.Lorentz_params_R
            coupling_index = self.coupling_index_R
            Delta_alpha = self.Delta_R
        if not callable(Gamma): #make gamma callable even in the WBL to make the syntax the same in every case.
            Gamma_mat = Gamma
            Gamma = lambda x : Gamma_mat
            #print('WBL! Gamma was not callable')


        t0=time.time()
        if self.Vbias_int is None:
            expint = np.exp(-1j*self.expint3(T,tau,alpha))
        else:
            expint = np.exp(-1j*Delta_alpha*(self.Vbias_int(T+tau/2) - self.Vbias_int(T-tau/2)))

        if self.use_aux_modes:
            #print('warning! aux modes not yet implemented in v3')
            #assert 1==0
            taup = tau[tau>=0].reshape(1,-1,1,1)
            taum = tau[tau<0].reshape(1,-1,1,1)
            fermi_res_lower = lambda E : Gamma(E)*exp(-1j*(E)*taup)
            Lorentz_res_lower = lambda E : self.Fermi_pade(E,alpha)*exp(-1j*(E)*taup)#f(E)*exp(-1j*(E)*taup)#

            fermi_res_upper = lambda E : Gamma(E)*exp(-1j*(E)*taum)
            Lorentz_res_upper = lambda E : self.Fermi_pade(E,alpha)*exp(-1j*(E)*taum) #f(E)*exp(-1j*(E)*taum)#

            fermi_int_lower = self.int_residues2(fermi_res_lower,self.Fermi_params,halfplane='lower')
            Lorentz_int_lower = self.int_residues2(Lorentz_res_lower,Lorentz_params,halfplane='lower')
            integral_lower = fermi_int_lower + Lorentz_int_lower

            fermi_int_upper = self.int_residues2(fermi_res_upper,self.Fermi_params,halfplane='upper')
            Lorentz_int_upper = self.int_residues2(Lorentz_res_upper,Lorentz_params,halfplane='upper')
            integral_upper = fermi_int_upper + Lorentz_int_upper
            
            Sigma_less_integrand = np.concatenate((integral_upper,integral_lower),axis=1)/(2*np.pi)
            alternate =2*(np.linspace(1,np.size(tau),np.size(tau)).reshape(1,-1,1,1) %2) - 1 #array of 1,-1,1,-1,..etc
            Sigma_less_integrand = np.size(w)*dtau*Sigma_less_integrand * alternate #see https://math.stackexchange.com/questions/688113/approximating-the-fourier-transform-with-dft-fft
            Sigma_less = 1j*self.ifft(expint*Sigma_less_integrand,axis=1)

            t1=time.time()
            #print('sigma_less: using aux mode! Got residues in %.2f'%(t1-t0),flush=True)
        else:
            Sigma_less = np.zeros((len(T),np.size(w),self.device_dim,self.device_dim),dtype=np.complex128)
            fermi = f(w)
            Gam = Gamma(w)
            for i in range(self.device_dim):
                for j in range(self.device_dim):
                    if coupling_index[i,j]:
                        Sigma_less_integrand=self.fft(Gam[:,:,i,j]*fermi[:,:,0,0],axis=1)
                        Sigma_less[:,:,i,j] = 1j*self.ifft(expint[:,:,0,0]*Sigma_less_integrand,axis=1)
            t1=time.time()
            #print('sigma_less: did not use aux mode! Finished in %.2f'%(t1-t0),flush=True)

        #Sigma_less = 1j*self.ifft(expint*Sigma_less_integrand,axis=1)
        Sigma_less = Sigma_less[:,extension_length:-extension_length]

        return Sigma_less


    def calc_Sigma_R(self,T=None,omega=None,alpha='L',extension_length=50):
        if np.all(T==None):
            T=self.T
            #print('calc_Sigma_R: set T = self.T',flush=True)
        if np.all(omega==None):
            omega = self.omega
            set_self_tau = 1
        else: 
            set_self_tau = 0
        T=np.array(T).reshape(-1,1,1,1)
        omega=np.array(omega).reshape(-1,1,1)   
        exp = np.exp
        omega_min = omega.min()
        omega_max = omega.max()
        N = np.size(omega)
        if extension_length < N/4:
            extension_length = int(N/4)
        #print('extending internal arrays by ',extension_length)
        dw = (omega_max - omega_min)/(np.size(omega)-1)
        dnu = dw/(2*np.pi)
        extension = dw*np.linspace(1,extension_length,extension_length)
        w = np.concatenate(((omega_min - np.flip(extension)),omega.flatten(),omega_max + extension))
        w = w.reshape(1,-1,1,1)
        tau = self.fftfreq(np.size(w),d=dnu).reshape(1,-1,1,1) #shape: T, tau, n, n
        if set_self_tau:
            self.tau = tau
        dtau = (np.max(tau)-np.min(tau))/(np.size(tau)-1)
        if alpha=='L':
            Gamma = self.Gamma_L
            potential = self.potential_L
            Lorentz_params = self.Lorentz_params_L
            coupling_index=self.coupling_index_L
            Delta_alpha = self.Delta_L
        else:
            Gamma = self.Gamma_R
            potential = self.potential_R
            Lorentz_params = self.Lorentz_params_R
            coupling_index=self.coupling_index_R
            Delta_alpha = self.Delta_R
        if not callable(Gamma): #make gamma callable even in the WBL to make the syntax the same in every case.
            Gamma_mat = Gamma
            Gamma = lambda x : Gamma_mat
            #print('WBL! Gamma was not callable')


        t0=time.time()
        if self.Vbias_int is None:
            expint = np.exp(-1j*self.expint3(T,tau,alpha))
        else:
            expint = np.exp(-1j*Delta_alpha*(self.Vbias_int(T+tau/2) - self.Vbias_int(T-tau/2)))

        if self.use_aux_modes:
            #print('error! aux modes not yet implemented in v3')
            #assert 1==0
            taup = tau[tau>0].reshape(1,-1,1,1)
            taum = tau[tau<0].reshape(1,-1,1,1)
            Lorentz_res_lower = lambda E : exp(-1j*(E)*taup)#f(E)*exp(-1j*(E)*taup)#

            integral_lower = self.int_residues2(Lorentz_res_lower,Lorentz_params,halfplane='lower')
            integral_upper = np.zeros(np.shape(integral_lower))

            if np.any(tau ==0):
                integral_zero = np.sum(Gamma(w),axis=1).reshape(1,1,self.device_dim,self.device_dim)*dw/2 #factor one half comes from the approximation that theta(0) = 1/2 !
                Sigma_R_integrand = np.concatenate((integral_upper,integral_zero,integral_lower),axis=1)/(2*np.pi)
            else:
                Sigma_R_integrand = np.concatenate((integral_upper,integral_lower),axis=1)/(2*np.pi)
            alternate =2*(np.linspace(1,np.size(tau),np.size(tau)).reshape(1,-1,1,1) %2) - 1 #array of 1,-1,1,-1,..etc
            Sigma_R_integrand = np.size(w)*dtau*Sigma_R_integrand * alternate #see https://math.stackexchange.com/questions/688113/approximating-the-fourier-transform-with-dft-fft
            theta = np.zeros(np.shape(tau))
            theta[tau>0] = 1
            theta[tau == 0] = 1/2
            Sigma_R = -1j*self.ifft(theta*expint*Sigma_R_integrand,axis=1)
            t1=time.time()
            #print('sigma_R: using aux mode! Got residues in %.2f'%(t1-t0),flush=True)
        else:
            Sigma_R = np.zeros((len(T),np.size(w),self.device_dim,self.device_dim),dtype=np.complex128)
            Gam = Gamma(w)
            theta = np.zeros(np.shape(tau))
            theta[tau>0] = 1
            theta[tau == 0] = 1/2
            for i in range(self.device_dim):
                for j in range(self.device_dim):
                    if coupling_index[i,j]:
                        Sigma_R_integrand=self.fft(Gam[:,:,i,j],axis=1)
                        Sigma_R[:,:,i,j] = -1j*self.ifft(theta[:,:,0,0]*expint[:,:,0,0]*Sigma_R_integrand,axis=1)


            t1=time.time()
            #print('sigma_R: did not use aux mode! Finished in %.2f'%(t1-t0),flush=True)

        #Sigma_R = -1j*self.ifft(expint*Sigma_R_integrand,axis=1)
        Sigma_R = Sigma_R[:,extension_length:-extension_length]

        return Sigma_R


    def calc_Sigma(self,T=None,omega=None):
        if T is None:
            T = self.T
        if omega is None:
            omega = self.omega

        if self.WBL_R:
            Sigma_R_R = -1j*np.ones(np.shape(T*omega))*self.Gamma_R/2
            #print('calculated sigma_R_R in the WBL',flush=True)
        else:
            Sigma_R_R = self.calc_Sigma_R(T,omega,alpha='R')
        if self.WBL_L:
            Sigma_L_R = -1j*np.ones(np.shape(T*omega))*self.Gamma_L/2
            #print('calculated sigma_L_R in the WBL',flush=True)
        else:
            Sigma_L_R = self.calc_Sigma_R(T,omega,alpha='L')
        Sigma_L_less = self.calc_Sigma_less(T,omega,alpha='L')
        Sigma_R_less = self.calc_Sigma_less(T,omega,alpha='R')
        Sigma_L_A = np.conjugate(Sigma_L_R)
        Sigma_R_A = np.conjugate(Sigma_R_R)

        #Total self energies
        Sigma_R = Sigma_R_R + Sigma_L_R
        Sigma_A = Sigma_R_A + Sigma_L_A
        Sigma_less = Sigma_L_less + Sigma_R_less

        self.Sigma_R_less  = Sigma_R_less  
        self.Sigma_L_less  = Sigma_L_less  
        self.Sigma_L_R  = Sigma_L_R  
        self.Sigma_L_A  = Sigma_L_A  
        self.Sigma_R_R  = Sigma_R_R  
        self.Sigma_R_A  = Sigma_R_A  
        self.Sigma_R  = Sigma_R  
        self.Sigma_A  = Sigma_A  
        self.Sigma_less  = Sigma_less  
        
        return [Sigma_L_R,Sigma_R_R,Sigma_L_less,Sigma_R_less]




    def calc_density(self):
        if np.all(self.G0_less==0):
            self.calc_current()
        density = np.sum(self.omega_weights*np.imag(self.G0_less+self.G1_less),axis=1)/(2*np.pi)
        return density


    def calc_current(self,T=None,omega=None,side='left'):
        deriv = self.FD
        if T is None:
            T = self.T
        if omega is None:
            omega = self.omega
            omega_weights = self.omega_weights
        else:
            w=omega.flatten()
            omega_weights = (w[1:] - w[:-1]) #calculate dw
            omega_weights[0] = omega_weights[0]/2 #trapez: divide first weight by 2
            omega_weights = list(omega_weights)
            omega_weights.append(omega_weights[-1]/2)
            omega_weights = np.array(omega_weights).reshape(np.shape(omega))
        f_Sigma_L_R = lambda t : self.calc_Sigma_R(t,omega,alpha='L')
        f_Sigma_R_R = lambda t : self.calc_Sigma_R(t,omega,alpha='R')
        f_Sigma_L_less = lambda t : self.calc_Sigma_less(t,omega,alpha='L')
        f_Sigma_R_less = lambda t : self.calc_Sigma_less(t,omega,alpha='R')
        J0_L = np.zeros(len(T))
        J1_L = np.zeros(len(T))
        J0_R = np.zeros(len(T))
        J1_R = np.zeros(len(T))
        G0_less_array = np.zeros((np.size(T),np.size(omega),self.device_dim,self.device_dim),dtype=np.complex128)
        G0_R_array = np.zeros((np.size(T),np.size(omega),self.device_dim,self.device_dim),dtype=np.complex128)
        G1_less_array = np.zeros((np.size(T),np.size(omega),self.device_dim,self.device_dim),dtype=np.complex128)
        G1_R_array = np.zeros((np.size(T),np.size(omega),self.device_dim,self.device_dim),dtype=np.complex128)
        Pi0_L_array = np.zeros((np.size(T),np.size(omega),self.device_dim,self.device_dim),dtype=np.complex128)
        Pi1_L_array = np.zeros((np.size(T),np.size(omega),self.device_dim,self.device_dim),dtype=np.complex128)
        for i in range(len(T)):
            print('loop %d/%d'%(i+1,len(T)),flush=True)
            t=T[i].reshape(1,1,1,1)
            Sigma_L_R = f_Sigma_L_R(t)
            Sigma_R_R = f_Sigma_R_R(t)
            Sigma_L_less = f_Sigma_L_less(t)
            Sigma_R_less = f_Sigma_R_less(t)

            Sigma_L_A = np.conjugate(Sigma_L_R)
            Sigma_R_A = np.conjugate(Sigma_R_R)
            Sigma_R = Sigma_R_R + Sigma_L_R
            #Sigma_A = Sigma_R_A + Sigma_L_A #not used
            Sigma_less = Sigma_L_less + Sigma_R_less

            Sigma_L_less_dT = deriv(f_Sigma_L_less ,t)
            Sigma_R_less_dT = deriv(f_Sigma_R_less ,t)
            Sigma_L_R_dT = deriv(f_Sigma_L_R ,t)
            Sigma_R_R_dT = deriv(f_Sigma_R_R ,t)

            Sigma_L_A_dT = np.conjugate(Sigma_L_R_dT)
            Sigma_R_A_dT = np.conjugate(Sigma_R_R_dT)

            #functions to calculate derivatives wrt omega
            fw_Sigma_L_R = lambda w : self.calc_Sigma_R(t,w,alpha='L')
            fw_Sigma_R_R = lambda w : self.calc_Sigma_R(t,w,alpha='R')
            fw_Sigma_L_less = lambda w : self.calc_Sigma_less(t,w,alpha='L')
            fw_Sigma_R_less = lambda w : self.calc_Sigma_less(t,w,alpha='R')

            Sigma_L_less_dw = deriv(fw_Sigma_L_less ,omega,dx=1e-6)
            Sigma_R_less_dw = deriv(fw_Sigma_R_less ,omega,dx=1e-6)
            Sigma_L_R_dw = deriv(fw_Sigma_L_R ,omega,dx=1e-6)
            Sigma_R_R_dw = deriv(fw_Sigma_R_R ,omega,dx=1e-6)
            
            Sigma_L_A_dw = np.conjugate(Sigma_L_R_dw)
            Sigma_R_A_dw = np.conjugate(Sigma_R_R_dw)

            #Total self energies
            Sigma_less_dT = Sigma_L_less_dT + Sigma_R_less_dT
            Sigma_less_dw = Sigma_L_less_dw + Sigma_R_less_dw

            Sigma_R_dT =  Sigma_R_R_dT + Sigma_L_R_dT
            Sigma_A_dT =  Sigma_R_A_dT + Sigma_L_A_dT
            Sigma_R_dw=  Sigma_R_R_dw + Sigma_L_R_dw
            Sigma_A_dw=  Sigma_R_A_dw + Sigma_L_A_dw
            #print('Calculated derivatives',flush=True)


            H =  self.H[i]
            H_dT = self.H_dT[i]


            G0_R = inv(omega*np.identity(self.device_dim) - H - Sigma_R)
            G0_R_array[i] = G0_R
            G0_A = np.conjugate(G0_R)
            G0_less = G0_R@Sigma_less@G0_A
            G0_less_array[i] = G0_less
            #G0_great = G0_R - G0_A + G0_less

            #print('calculated G0',flush=True)
            #Current matrices - zero order. Eq (35)
            Pi0_L = G0_less@Sigma_L_A + G0_R@Sigma_L_less


            #The integral over omega is calculated as the vector product with the gauss-legendre weights. Eq (34)
            J0_L[i] = 1/np.pi * np.sum(np.trace(np.real(Pi0_L)*omega_weights,axis1=2,axis2=3),axis=1)


            #del(Pi0_L)
            Pi0_R = G0_less@Sigma_R_A + G0_R@Sigma_R_less
            Pi0_L_array[i] = Pi0_L
            J0_R[i] = 1/np.pi * np.sum(np.trace(np.real(Pi0_R)*omega_weights,axis1=2,axis2=3),axis=1)
            #del(Pi0_R)
            #print('calculated J0',flush=True)
            #FIRST ORDER

            #green function derivatives. Eqs (30), (29), (31)
            G0_R_dT = G0_R @ (H_dT+Sigma_R_dT)@G0_R
            G0_A_dT = np.conjugate(G0_R_dT)

            G0_R_dw = -G0_R@G0_R + G0_R@Sigma_R_dw@G0_R 
            G0_A_dw = np.conjugate(G0_R_dw)

            G0_less_dT = G0_R_dT@Sigma_less@G0_A + G0_R@Sigma_less@G0_A_dT + G0_R@Sigma_less_dT@G0_A
            G0_less_dw = G0_R_dw@Sigma_less@G0_A + G0_R@Sigma_less@G0_A_dw + G0_R@Sigma_less_dw@G0_A

            #First order green functions. Eqs (28) and (32)
            G1_R =  1j/2 * G0_R @  (-G0_R_dT - H_dT@G0_R_dw - Sigma_R_dT@G0_R_dw + Sigma_R_dw @ G0_R_dT)
            G1_R_array[i] = G1_R
            G1_A =  1j/2 * G0_A @  (-G0_A_dT - H_dT@G0_A_dw - Sigma_A_dT@G0_A_dw + Sigma_A_dw @ G0_A_dT)

            G1_less = G0_R@Sigma_less@G1_A + 1j/2*G0_R@(
                        -G0_less_dT - H_dT @ G0_less_dw - Sigma_less_dT@G0_A_dw + Sigma_less_dw@G0_A_dT - Sigma_R_dT@G0_less_dw + Sigma_R_dw@G0_less_dT
                        )
            G1_less_array[i] = G1_less
            #print('calculated G1',flush=True)
            #First order current matrices
            Pi1_L = G1_less@Sigma_L_A + G1_R@Sigma_L_less - 1j/2 * G0_less_dT@Sigma_L_A_dw + 1j/2*G0_less_dw@Sigma_L_A_dT -1j/2*G0_R_dT@Sigma_L_less_dw + 1j/2*G0_R_dw@Sigma_L_less_dT
            Pi1_L_array[i] = Pi1_L
            J1_L[i] = 1/np.pi * np.sum(np.trace(np.real(Pi1_L)*omega_weights,axis1=2,axis2=3),axis=1)
            #del(Pi1_L)
            Pi1_R = G1_less@Sigma_R_A + G1_R@Sigma_R_less - 1j/2 * G0_less_dT@Sigma_R_A_dw + 1j/2*G0_less_dw@Sigma_R_A_dT -1j/2*G0_R_dT@Sigma_R_less_dw + 1j/2*G0_R_dw@Sigma_R_less_dT
            J1_R[i] = 1/np.pi * np.sum(np.trace(np.real(Pi1_R)*omega_weights,axis1=2,axis2=3),axis=1)
        self.G0_R = G0_R_array
        self.G0_less = G0_less_array
        self.G1_R = G1_R_array
        self.G1_less = G1_less_array
        self.Pi0_L = Pi0_L_array
        self.Pi1_L = Pi1_L_array
        if side =='left': 
            return J0_L, J1_L
        elif side=='right':
            return J0_R, J1_R
        elif side=='both':
            return [J0_L,J1_L],[J0_R,J1_R]

    def calc_current_ac(self,Omega,T=None,omega=None,side='left',N_bessel_functions=50):
        #deriv = self.FD
        if T is None:
            T = self.T
        if omega is None:
            omega = self.omega
            omega_weights = self.omega_weights
        else:
            w=omega.flatten()
            omega_weights = (w[1:] - w[:-1]) #calculate dw
            omega_weights[0] = omega_weights[0]/2 #trapez: divide first weight by 2
            omega_weights = list(omega_weights)
            omega_weights.append(omega_weights[-1]/2)
            omega_weights = np.array(omega_weights).reshape(np.shape(omega))
        J0_L = np.zeros(len(T))
        J1_L = np.zeros(len(T))
        J0_R = np.zeros(len(T))
        J1_R = np.zeros(len(T))
        G0_less_array = np.zeros((np.size(T),np.size(omega),self.device_dim,self.device_dim),dtype=np.complex128)
        G0_R_array = np.zeros((np.size(T),np.size(omega),self.device_dim,self.device_dim),dtype=np.complex128)
        G1_less_array = np.zeros((np.size(T),np.size(omega),self.device_dim,self.device_dim),dtype=np.complex128)
        G1_R_array = np.zeros((np.size(T),np.size(omega),self.device_dim,self.device_dim),dtype=np.complex128)

        #Get poles and residues
        Lor_poles_L,Lor_res_L = self.Lorentz_params_L
        Lor_poles_R,Lor_res_R = self.Lorentz_params_R
        Fermi_poles, Fermi_res = self.Fermi_params
        dim = self.device_dim
        Fermi_pade = self.Fermi_pade
        Fermi_poles = Fermi_poles.reshape(-1,1,1)
        Fermi_res = Fermi_res.reshape(-1,1,1)
        chi_poles_L = np.concatenate((Fermi_poles,Lor_poles_L))
        chi_res_L = np.concatenate((self.Gamma_L(Fermi_poles)*Fermi_res,Fermi_pade(Lor_poles_L)*Lor_res_L))
        #chi_res_L = np.concatenate((self.Gamma_L(Fermi_poles)*Fermi_res,f_L(Lor_poles_L)*Lor_res_L))
        chi_poles_R = np.concatenate((Fermi_poles,Lor_poles_R))
        chi_res_R = np.concatenate((self.Gamma_R(Fermi_poles)*Fermi_res,Fermi_pade(Lor_poles_R)*Lor_res_R))
        #chi_res_R = np.concatenate((self.Gamma_R(Fermi_poles)*Fermi_res,f_R(Lor_poles_R)*Lor_res_R))
        chi_poles_L = chi_poles_L.reshape(-1,1,1,1,1)
        chi_res_L = chi_res_L.reshape(-1,1,1,dim,dim)
        chi_poles_R = chi_poles_R.reshape(-1,1,1,1,1)
        chi_res_R = chi_res_R.reshape(-1,1,1,dim,dim)
        LHP = (Lor_poles_L.imag < 0).flatten()  #get indices for poles in lower half plane
        Lor_poles_L_LHP = Lor_poles_L[LHP]
        Lor_res_L_LHP = Lor_res_L[LHP]
        LHP_R = (Lor_poles_R.imag < 0).flatten()  #get indices for poles in lower half plane
        Lor_poles_R_LHP = Lor_poles_R[LHP_R]
        Lor_res_R_LHP = Lor_res_R[LHP_R]
        Lor_poles_L_LHP = Lor_poles_L_LHP.reshape(-1,1,1,1,1)
        Lor_res_L_LHP = Lor_res_L_LHP.reshape(-1,1,1,dim,dim)
        Lor_poles_R_LHP = Lor_poles_R_LHP.reshape(-1,1,1,1,1)
        Lor_res_R_LHP = Lor_res_R_LHP.reshape(-1,1,1,dim,dim)
        ###
        for i in range(len(T)):
            t = T[i].reshape(1,1,1,1)
            H =  self.H[i]
            H_dT = self.H_dT[i]
            Psi_L = 2*self.Delta_L/Omega*np.cos(Omega*t)
            Psi_R = 2*self.Delta_R/Omega*np.cos(Omega*t)
            Psi_L_dT = -2*Delta_L *np.sin(Omega*t)
            Psi_R_dT = -2*Delta_R *np.sin(Omega*t)

            Sigma_L_R = 0
            Sigma_R_R = 0
            Sigma_L_less = 0
            Sigma_R_less = 0
            Sigma_L_R_dT = 0
            Sigma_R_R_dT = 0
            Sigma_L_less_dT = 0
            Sigma_R_less_dT = 0
            Sigma_L_R_dw = 0
            Sigma_R_R_dw = 0
            Sigma_L_less_dw = 0
            Sigma_R_less_dw = 0
            for n in range(-N_bessel_functions,N_bessel_functions+1):
                Sigma_L_R = Sigma_L_R + -1j*(-1)**n * jv(n,Psi_L) * np.sum(Lor_res_L_LHP/(omega + n*Omega/2 - Lor_poles_L_LHP),axis=0)
                Sigma_R_R = Sigma_R_R + -1j*(-1)**n * jv(n,Psi_R) * np.sum(Lor_res_R_LHP/(omega + n*Omega/2 - Lor_poles_R_LHP),axis=0)
                Sigma_L_less = Sigma_L_less + 1j*(-1)**n * jv(n,Psi_L) * np.sum(chi_res_L/(omega + n*Omega/2 - chi_poles_L),axis=0)
                Sigma_R_less = Sigma_R_less + 1j*(-1)**n * jv(n,Psi_R) * np.sum(chi_res_R/(omega + n*Omega/2 - chi_poles_R),axis=0) 
                Sigma_L_R_dT = Sigma_L_R_dT + -1j*(-1)**n * Psi_L_dT*jvp(n,Psi_L) * np.sum(Lor_res_L_LHP/(omega + n*Omega/2 - Lor_poles_L_LHP),axis=0)
                Sigma_R_R_dT = Sigma_R_R_dT + -1j*(-1)**n * Psi_R_dT*jvp(n,Psi_R) * np.sum(Lor_res_R_LHP/(omega + n*Omega/2 - Lor_poles_R_LHP),axis=0)
                Sigma_L_less_dT = Sigma_L_less_dT + 1j*(-1)**n * Psi_L_dT*jvp(n,Psi_L) * np.sum(chi_res_L/(omega + n*Omega/2 - chi_poles_L),axis=0)
                Sigma_R_less_dT = Sigma_R_less_dT + 1j*(-1)**n * Psi_R_dT*jvp(n,Psi_R) * np.sum(chi_res_R/(omega + n*Omega/2 - chi_poles_R),axis=0) 
                Sigma_L_R_dw = Sigma_L_R_dw + 1j*(-1)**n * jv(n,Psi_L) * np.sum(Lor_res_L_LHP/(omega + n*Omega/2 - Lor_poles_L_LHP)**2,axis=0) 
                Sigma_R_R_dw = Sigma_R_R_dw + 1j*(-1)**n * jv(n,Psi_R) * np.sum(Lor_res_R_LHP/(omega + n*Omega/2 - Lor_poles_R_LHP)**2,axis=0)
                Sigma_L_less_dw = Sigma_L_less_dw - 1j*(-1)**n * jv(n,Psi_L) * np.sum(chi_res_L/(omega + n*Omega/2 - chi_poles_L)**2,axis=0)
                Sigma_R_less_dw = Sigma_R_less_dw - 1j*(-1)**n * jv(n,Psi_R) * np.sum(chi_res_R/(omega + n*Omega/2 - chi_poles_R)**2,axis=0) 

            Sigma_L_A = np.conjugate(Sigma_L_R)
            Sigma_R_A = np.conjugate(Sigma_R_R)
            Sigma_R = Sigma_L_R + Sigma_R_R
            Sigma_less = Sigma_L_less + Sigma_R_less
            Sigma_A = np.conjugate(Sigma_L_R + Sigma_R_R)

            Sigma_L_A_dT = np.conjugate(Sigma_L_R_dT)
            Sigma_R_A_dT = np.conjugate(Sigma_R_R_dT)
            Sigma_L_A_dw = np.conjugate(Sigma_L_R_dw)
            Sigma_R_A_dw = np.conjugate(Sigma_R_R_dw)

            #Total self energies
            Sigma_less_dT = Sigma_L_less_dT + Sigma_R_less_dT
            Sigma_less_dw = Sigma_L_less_dw + Sigma_R_less_dw

            Sigma_R_dT =  Sigma_R_R_dT + Sigma_L_R_dT
            Sigma_A_dT =  Sigma_R_A_dT + Sigma_L_A_dT
            Sigma_R_dw=  Sigma_R_R_dw + Sigma_L_R_dw
            Sigma_A_dw=  Sigma_R_A_dw + Sigma_L_A_dw


            G0_R = inv(omega*np.identity(self.device_dim) - H - Sigma_R)
            G0_R_array[i] = G0_R
            G0_A = np.conjugate(G0_R)
            G0_less = G0_R@Sigma_less@G0_A
            G0_less_array[i] = G0_less
            #G0_great = G0_R - G0_A + G0_less

            #print('calculated G0',flush=True)
            #Current matrices - zero order. Eq (35)
            Pi0_L = G0_less@Sigma_L_A + G0_R@Sigma_L_less


            #The integral over omega is calculated as the vector product with the gauss-legendre weights. Eq (34)
            J0_L[i] = 1/np.pi * np.sum(np.trace(np.real(Pi0_L)*omega_weights,axis1=2,axis2=3),axis=1)


            #del(Pi0_L)
            Pi0_R = G0_less@Sigma_R_A + G0_R@Sigma_R_less
            J0_R[i] = 1/np.pi * np.sum(np.trace(np.real(Pi0_R)*omega_weights,axis1=2,axis2=3),axis=1)
            #del(Pi0_R)
            #print('calculated J0',flush=True)
            #FIRST ORDER

            #green function derivatives. Eqs (30), (29), (31)
            G0_R_dT = G0_R @ (H_dT+Sigma_R_dT)@G0_R
            G0_A_dT = np.conjugate(G0_R_dT)

            G0_R_dw = -G0_R@G0_R + G0_R@Sigma_R_dw@G0_R 
            G0_A_dw = np.conjugate(G0_R_dw)

            G0_less_dT = G0_R_dT@Sigma_less@G0_A + G0_R@Sigma_less@G0_A_dT + G0_R@Sigma_less_dT@G0_A
            G0_less_dw = G0_R_dw@Sigma_less@G0_A + G0_R@Sigma_less@G0_A_dw + G0_R@Sigma_less_dw@G0_A

            #First order green functions. Eqs (28) and (32)
            G1_R =  1j/2 * G0_R @  (-G0_R_dT - H_dT@G0_R_dw - Sigma_R_dT@G0_R_dw + Sigma_R_dw @ G0_R_dT)
            G1_R_array[i] = G1_R
            G1_A =  1j/2 * G0_A @  (-G0_A_dT - H_dT@G0_A_dw - Sigma_A_dT@G0_A_dw + Sigma_A_dw @ G0_A_dT)

            G1_less = G0_R@Sigma_less@G1_A + 1j/2*G0_R@(
                        -G0_less_dT - H_dT @ G0_less_dw - Sigma_less_dT@G0_A_dw + Sigma_less_dw@G0_A_dT - Sigma_R_dT@G0_less_dw + Sigma_R_dw@G0_less_dT
                        )
            G1_less_array[i] = G1_less
            #print('calculated G1',flush=True)
            #First order current matrices
            Pi1_L = G1_less@Sigma_L_A + G1_R@Sigma_L_less - 1j/2 * G0_less_dT@Sigma_L_A_dw + 1j/2*G0_less_dw@Sigma_L_A_dT -1j/2*G0_R_dT@Sigma_L_less_dw + 1j/2*G0_R_dw@Sigma_L_less_dT
            J1_L[i] = 1/np.pi * np.sum(np.trace(np.real(Pi1_L)*omega_weights,axis1=2,axis2=3),axis=1)
            #del(Pi1_L)
            Pi1_R = G1_less@Sigma_R_A + G1_R@Sigma_R_less - 1j/2 * G0_less_dT@Sigma_R_A_dw + 1j/2*G0_less_dw@Sigma_R_A_dT -1j/2*G0_R_dT@Sigma_R_less_dw + 1j/2*G0_R_dw@Sigma_R_less_dT
            J1_R[i] = 1/np.pi * np.sum(np.trace(np.real(Pi1_R)*omega_weights,axis1=2,axis2=3),axis=1)
        self.G0_R = G0_R_array
        self.G0_less = G0_less_array
        self.G1_R = G1_R_array
        self.G1_less = G1_less_array
        if side =='left': 
            return J0_L, J1_L
        elif side=='right':
            return J0_R, J1_R
        elif side=='both':
            return [J0_L,J1_L],[J0_R,J1_R]


####################################### end class
