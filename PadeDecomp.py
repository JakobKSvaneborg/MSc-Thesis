import numpy as np
# The functions found here are either from the Croy 2009 paper
# or the Jie Hu paper from 2011. Read these to understand these functions.
# They simply return the poles and weights of an expansion of the fermi function 
# in terms of simple poles




def Pade_poles_and_coeffs(N_F):
    x = Pade_Poles(N_F)
    return x, np.ones(len(x))

def Pade_Poles(N_F):
    Z = np.zeros((N_F, N_F), dtype = np.complex128)
    for i in range(N_F):
        for j in range(N_F): 
            I = i+1
            if j == i+1:
                Z[i,j] = 2 * I * ( 2 * I -1)
            if i == N_F - 1:
                Z[i,j] = -2 * N_F * (2 * N_F -1)
    
    eig, v = np.linalg.eig(Z)
    x = 2 * np.sqrt(eig)
    xr = x.real
    xi = x.imag
    x[xi<0] *= -1
    return x

def FD_expanded(E, xp, beta , mu = 0.0, coeffs = None):
    Xpp = mu  + xp/beta
    Xpm = mu  - xp/beta
    if coeffs is None:
        coeffs = np.ones(len(xp))
    diffs =  (1 / beta) * (1 / np.subtract.outer(E , Xpp)  + 1 / np.subtract.outer(E , Xpm)) * coeffs
    return 1 / 2 - diffs.sum(axis = 1)

def FD(E, beta, mu = 0.0):
    return 1 / (1 + np.exp((E - mu) * beta))

def diff(E, xp, beta, mu = 0.0):
    return FD(E, beta, mu = mu) - FD_expanded(E, xp, beta, mu = mu)

def Hu_b(m):
    return 2 * m -1
def Hu_RN(N):
    return 1/(4 * (N+1) *Hu_b(N+1))

def Hu_Gamma(M):
    Mat = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            I = i+1
            J = j+1
            if  i == j+1 or i == j-1:
                Mat[i,j] = 1/np.sqrt(Hu_b(I) * Hu_b(J))
    return Mat
def Hu_roots_Q(N):
    M = 2 * N
    e,v = np.linalg.eig(Hu_Gamma(M))
    #print(e)
    e = np.sort(e[e>1e-15])[::-1]
    return 2/e
def Hu_roots_P(N):
    M = 2 * N
    e,v = np.linalg.eig(Hu_Gamma(M)[1:, 1:])
    e = e[e>1e-15]
    e = 2/e
    return e

def Hu_coeffs(N):
    Const =  N * Hu_b(N+1)/2
    Qx = Hu_roots_Q(N)
    Px = Hu_roots_P(N)
    coeffs = []
    for i in range(N):
        p1 = Qx**2 - Qx[i]**2
        p1[np.abs(p1)<1e-15] = 1.0
        p1 = np.prod(p1)
        p2 = np.prod(Px ** 2 - Qx[i]**2 )
        coeffs += [Const*p2/p1]
    return np.array(coeffs)

def Hu_poles(N):
    return 1j * Hu_roots_Q(N), Hu_coeffs(N)
