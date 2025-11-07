from math import *
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle
from numpy.linalg import solve, inv
from scipy.linalg import sqrtm
import cvxpy as cp

kappa = 1/15

def LPdemo(kappa = 1/15, plot = False):

    t = 3/(array([5/3,10/3,10/3])@array([23,11,10]))
    X = [array([5/3,10/3,10/3])]
    # S = [array([3,1,0])]  # already at optimal
    S = [array([23,11,10])]
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
        g  = -1.0/X[-1]
        
        # NT scaling factor
        w = (1.0/sqrt(t_plus))*sqrt(X[-1]/S[-1])
        
        H = diag(1.0/(w**2))

        r_c= -S[-1] - (1.0/t_plus) * g

        K = block([
            [(1.0/t_plus)*H,  zeros((n,m)), eye(n)],
            [A,               zeros((m,m)), zeros((m,n))],
            [zeros((n,n)), A.T,             eye(n)]
        ])
        
        rhs = concatenate([r_c, zeros(m), zeros(n)])
        
        z = solve(K, rhs)
        
        X.append(X[-1] + z[:n])
        S.append(S[-1] + z[n+m:])
        y.append(y[-1]+z[n:n+m])
        
        if X[-1] @ S[-1] < tol:
            break
        
        t = t_plus
        count += 1

    if plot:
        plotLP(X=X)
    
    return X, S, y, count


def plotLP(X):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xrange = linspace(0, 10, 100)
    yrange = linspace(0, 10, 100)

    Xplot, Yplot = meshgrid(xrange, yrange)

    Zplot = -2 * Xplot - Yplot + 10

    Z_masked = where(
        (Zplot < 0) | (Zplot > 10),
        nan,
        Zplot
    )

    plane = ax.plot_surface(Xplot, Yplot, Z_masked, cmap='viridis', alpha=0.8)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.zaxis.labelpad=-2.7
    ax.set_zlim(0, 10)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.view_init(elev=25, azim=225)
    ax.set_box_aspect((1,1,1))

    ntpts = vstack(X)
    xcoor_nt, ycoor_nt, zcoor_nt = ntpts.T

    ax.scatter(xcoor_nt, ycoor_nt, zcoor_nt, color='red', s=1)

    plane_proxy = Patch(facecolor="blue", edgecolor="k", alpha=0.3)
    sc_proxy  = Line2D([0], [0], linestyle="none", marker="o", color="red", markersize=3)

    ax.legend([plane_proxy, sc_proxy], ["Feasible region", "Iterates"], bbox_to_anchor=(0.95, 0.85))
    plt.title("LPdemo problem primal trajectory", y=1.04)

    plt.show()

    return


def LPcvxpy():
    x = cp.Variable(3)

    A = array([[2,1,1]])
    b = array([10])
    c = array([-1,-1,-2])


    objective = cp.Minimize(c @ x)
    constraints = [A @ x == b, x >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    return x.value, prob.value


def SDPdemo(kappa = 1/15):
    A_list = [
        array([
            [-1, 0],
            [0, 0]
        ]),
        array([
            [1, 0],
            [0, 1]
        ])
    ]

    b_list = [None, 1]

    X_list = [0.5 * eye(2)]
    y_list = [-2]
    S_list = [A_list[0] - y_list[0]*A_list[1]]
    t_list = [2 / trace(X_list[-1] @ S_list[-1])]
    tol = 1e-6

    count = 1

    while count < 1000:
        t_list.append(t_list[-1] / (1 - kappa/sqrt(2)))
        
        V = 1.0/sqrt(t_list[-1]) * inv(sqrtm(S_list[-1])) @ sqrtm(sqrtm(S_list[-1]) @ X_list[-1] @ sqrtm(S_list[-1])) @ inv(sqrtm(S_list[-1]))
        
        XAX = V @ A_list[1] @ V
        RHS = V @ S_list[-1] @ V - 1.0/t_list[-1]* V @ inv(X_list[-1]) @ V
        M = array([
            [-1, 0, 0, 0, XAX[0][0]],
            [0, -1, 0, 0, XAX[0][1]],
            [0, 0, -1, 0, XAX[1][0]],
            [0, 0, 0, -1, XAX[1][1]],
            [A_list[1][0][0], A_list[1][0][1], A_list[1][1][0], A_list[1][1][1], 0]
        ])
        v = solve(M, array([RHS[0][0], RHS[0][1], RHS[1][0], RHS[1][1], 0]))
        dy = v[-1]
        
        dS = -A_list[1] * dy

        dX = V @ (-t_list[-1] * S_list[-1] - t_list[-1] * dS + inv(X_list[-1])) @ V

        y_list.append(y_list[-1] + dy)
        X_list.append(X_list[-1] + dX)
        S_list.append(S_list[-1] + dS)
        
        if trace(S_list[-1] @ X_list[-1]) < tol:
            break

        count += 1

    return X_list, S_list, y_list, count
    

def SDPcvxpy():
    A0 = array([[-1.0, 0.0], [0.0, 0.0]])
    A1 = array([[1.0, 0.0], [0.0, 1.0]])
    X = cp.Variable((2, 2), symmetric = True)

    prob = cp.Problem(cp.Minimize(cp.trace(A0 @ X)),
                    [cp.trace(A1@X) == 1, X >> 0])
    prob.solve(solver=cp.SCS)

    return X.value, prob.value


def SOCPdemo(kappa = 1/15, plot = False):
    A = array([[1.0, 0.0, 0.0]])
    b = array([1.0])
    c = array([0.0, 1.0, 2.0])

    # Alternative initial conditions

    # x0 = array([1.0, 0.2, 0.4])
    # y0 = -3.0
    # s0 = array([-y0, 1.0, 2.0])

    # x0 = array([1.0, 0.0, 0.0])
    # y0 = -sqrt(5.0)-0.1
    # s0 = array([sqrt(5.0)+0.1, 1.0, 2.0])

    x0 = array([1.0, 0.0, 0.0])
    y0 = -5.0
    s0 = array([5.0, 1.0, 2.0])
    t = 2/(x0[0] * s0[0] - x0[1] * s0[1] - x0[2] * s0[2])
    tol = 1e-6

    x_list = [x0]
    y_list = [y0]
    s_list = [s0]

    count = 1

    while count < 1000:
        t_plus = t / (1 - kappa/sqrt(2))

        cx, xbar = x_list[-1][0], x_list[-1][1:]
        cs, sbar = s_list[-1][0], s_list[-1][1:]

        phix, phis = cx * cx - xbar @ xbar, cs * cs - sbar @ sbar
        if phix <= 0 or phis <= 0:
            break
    
        # calculate w
        gamma = sqrt(1 + (cx * cs - xbar @ sbar) / sqrt(phix * phis))
        cv = 1.0/sqrt(t_plus) * (1/gamma) * (cx/sqrt(phix) + cs/sqrt(phis))
        vbar = 1.0/sqrt(t_plus) * (1/gamma) * (xbar/sqrt(phix) - sbar/sqrt(phis))
        phiv = cv * cv - vbar @ vbar
        
        
        H = block([
        [cv**2 + vbar @ vbar,          -2 * cv * vbar],
        [-2 * cv * vbar[:, None],      phiv * eye(vbar.shape[0]) + 2 * outer(vbar, vbar)]
        ])
        
        lhs = A @ inv(H) @ A.T
        rhs = A @ inv(H) @ (s_list[-1] + 2/(t_plus*phix)*hstack((-cx, xbar)))
        dy = (rhs / lhs).item()   
        
        ds = (-dy * A.T).ravel()
        
        lhs = 1.0/t_plus * 2.0/(phix**2) * H
        rhs = -s_list[-1] - ds - 2.0/(t_plus * phix) * hstack((-cx, xbar))
        dx = solve(lhs, rhs).ravel()
        
        x_list.append(x_list[-1] + dx)
        y_list.append(y_list[-1] + dy)
        s_list.append(s_list[-1] + ds)

        if x_list[-1][0] * s_list[-1][0] - x_list[-1][1] * s_list[-1][1] - x_list[-1][2] * s_list[-1][2] < tol:
            break
        
        t = t_plus
        count += 1

    if plot:
        plotSOCP(x_list=x_list)

    return x_list, s_list, y_list, count
    

def plotSOCP(x_list):
    fig, ax = plt.subplots()

    unit_circle = Circle((0, 0), radius=1, facecolor='blue', alpha=0.15, edgecolor='blue', linewidth=2)
    edge = Circle((0, 0), radius=1, facecolor='none', edgecolor='blue', linewidth=2)

    ax.add_patch(unit_circle)
    ax.add_patch(edge)

    for arr in x_list:
        ax.plot(arr[1], arr[2], 'o', markersize=2, color='red')
        
    opt, = ax.plot(-1.0/sqrt(5), -2.0/sqrt(5),'o', markersize=5, color='black')

    ax.set_aspect('equal')

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])

    ax.set_title('SOCPdemo problem primal trajectory')

    ax.grid(True)

    sc_proxy  = Line2D([0], [0], linestyle="none", marker="o", color="red", markersize=3)

    plt.legend([unit_circle, opt, sc_proxy],["Feasible region", "Optimal solution", "Iterates"])

    plt.show()

    return


def SOCPcvxpy():

    c = array([1.0, 2.0])
    x = cp.Variable(2)

    prob = cp.Problem(cp.Minimize(c @ x),
                    [cp.norm(x, 2) <= 1])
    prob.solve(solver=cp.SCS)

    return x.value, prob.value
