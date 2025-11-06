from math import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import *
from numpy.linalg import solve

kappa = 1/15

# NT scaling

t = 3/(array([5/3,10/3,10/3])@array([23,11,10]))
Xprimalnt = [array([5/3,10/3,10/3])]
Xdualnt = [array([5/3,10/3,10/3])]
# S = [array([3,1,0])] # already at optimal
Sprimalnt = [array([23,11,10])]
Sdualnt = [array([23,11,10])]
# S = [array([601,300,299])] # close to central path
count = 1
y = [array([-12])]
tol = 1e-6

A = array([[2,1,1]])
b = array([10])
c = array([-1,-1,-2])
m, n = A.shape

while count < 1000:
    t_plus = t / (1 - kappa/sqrt(3))
    gprimal  = -1.0/Xprimalnt[-1]
    gdual  = -1.0/Sdualnt[-1]
    
    # NT scaling factor
    wprimal = (1.0/sqrt(t_plus))*sqrt(Xprimalnt[-1]/Sprimalnt[-1])
    wdual = (1.0/sqrt(t_plus))*sqrt(Sdualnt[-1]/Xdualnt[-1])
    
    Hprimal  = diag(1.0/(wprimal**2))
    Hdual  = diag(1.0/(wdual**2))

    r_c_primal = -Sprimalnt[-1] - (1.0/t_plus) * gprimal
    r_c_dual = -Xdualnt[-1] - (1.0/t_plus) * gdual

    Kprimal = block([
        [(1.0/t_plus)*Hprimal,  zeros((n,m)), eye(n)],
        [A,               zeros((m,m)), zeros((m,n))],
        [zeros((n,n)), A.T,             eye(n)]
    ])
    
    Kdual = block([
        [eye(n),       zeros((n,m)), (1.0/t_plus)*Hdual],
        [A,            zeros((m,m)), zeros((m,n))],
        [zeros((n,n)), A.T,             eye(n)]
    ])
    
    rhsprimal = concatenate([r_c_primal, zeros(m), zeros(n)])
    rhsdual = concatenate([r_c_dual, zeros(m), zeros(n)])
    
    zprimal = solve(Kprimal, rhsprimal)
    zdual = solve(Kdual, rhsdual)
    # dx = z[:n]; dy = z[n:n+m]; ds = z[n+m:]
    
    # Dx.append(dx)
    # Dy_bar.append(dy)
    # Ds_bar.append(ds)
    
    Xprimalnt.append(Xprimalnt[-1] + zprimal[:n])
    Sprimalnt.append(Sprimalnt[-1] + zprimal[n+m:])
    Xdualnt.append(Xdualnt[-1] + zdual[:n])
    Sdualnt.append(Sdualnt[-1] + zdual[n+m:])
    y.append(y[-1]+zprimal[n:n+m])
    
    if Xprimalnt[-1] @ Sprimalnt[-1] < tol:
        print(count)
        break
    
    t = t_plus
    count += 1