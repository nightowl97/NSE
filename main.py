import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigs
import scipy.sparse.linalg as la
from scipy import sparse
import os

"""
This program solves the one dimensional time-dependent schrodinger equation 
with an implicit finite difference method. The system is represented by a 2D
lattice with the boundary conditions representing a closed quantum system 
(large potential at x = 0 and x = 1).
Spatial domain                  :   x = [0, 1]
Boundary conditions (Dirichlet) :   Ψ(0, t) = 0
                                    Ψ(1, t) = 0
Potential field                 :   V = 0 everywhere
"""

# TODO: Fix boundary conditions

# Spatial discretisation
N = 500
x = np.linspace(0, 1, N)
dx = x[1] - x[0]


# Time discretisation
K = 300
t = np.linspace(0, .003, K)
dt = t[1] - t[0]

alpha = 1j * dt / (2 * (dx ** 2))
print(alpha)

A = sparse.diags([(N - 3)*[-alpha], (N - 2) * [1 + 2 * alpha], (N - 3) * [-alpha]], [-1, 0, 1], format="csc")
B = sparse.diags([(N - 3)*[ alpha], (N - 2) * [1 - 2 * alpha], (N - 3) * [ alpha]], [-1, 0, 1], format="csc")

# Check for matrix singularity
if not np.isfinite(np.linalg.cond(np.linalg.inv(A.todense()).dot(B.todense()))):
    print("Matrix is singular, aborting..")
    exit(0)

eigenvalues = np.abs(la.eigsh(sparse.linalg.inv(A) * B)[0])
if np.abs(1 - max(eigenvalues)) <= 1e-2:
    print("Looks stable.\nCalculating")
else:
    print("Propagator matrix maximum eigenvalue is:\t {}".format(max(eigenvalues)))
    exit(0)


# Initial and boundary conditions
psi = np.exp((1j * 1000 * x) - (2000 * (x - .25) ** 2))  # [:, None]  # new axis)
b = B.dot(psi[1:-1])
psi[0], psi[-1] = 0, 0

# Time propagation
frames = np.zeros([K, N])
for index, step in enumerate(t):
    # Within the domain
    psi[1:-1] = spsolve(A, b)

    # Enforce boundaries
    # psi[0], psi[N - 1] = 0, 0

    b = B.dot(psi[1:-1])

    frames[index] = np.abs(psi) ** 2

    # print(np.trapz(np.abs(psi) ** 2)) # Check unitarity

print("Solution calculated.")
# Animation
fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)
plt.grid()
plt.title('Time Dependent Schrodinger Equation (TDSE): Re(Ψ(x, t))')
plt.xlabel('X-axis')


# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,


def animate(i):
    line.set_data(x, frames[i])
    return line,


print("Animating..")

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=K, interval=20, blit=True)
anim.save('basic_animation.mp4', fps=60, extra_args=['-vcodec', 'libx264'])

os.system("xdg-open basic_animation.mp4")
