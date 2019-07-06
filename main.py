import scipy as sc
from scipy.constants import constants
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.linalg import toeplitz
from scipy.sparse.linalg import spsolve
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

# mass
m = 9e-30
hbar = constants.hbar

# Spatial discretisation
N = 1000
x = np.linspace(0, 1, N)
dx = x[1] - x[0]


# Time discretisation
K = 300
t = np.linspace(0, 1, K)
dt = t[1] - t[0]

alpha = (1j * hbar * dt) / (2 * m * (dx**2))
print(alpha)
A = toeplitz([2 + 2 * alpha, - alpha, *np.zeros(N-4)])  # 2 less for both boundaries
B = toeplitz([2 - 2 * alpha, alpha, *np.zeros(N-4)])

# Initial and boundary conditions
psi = np.multiply(np.exp(1j * 50 * (x - .5)), np.exp(- 200 * (x - .5) ** 2)) # [:, None]  # new axis)
rhs = B.dot(psi[1:-1])

# Time propagation
frames = np.zeros([K, N], dtype=complex)
for index, step in enumerate(t):
    # Within the domain
    # new_space = np.zeros(N)
    # new_space[1:-1] = psi[1:-1] - alpha * (psi[:-2] - 2 * psi[1:-1] + psi[2:])

    psi[1:-1] = spsolve(A, rhs)
    # Enforce boundaries
    # psi[0], psi[N - 1] = 0, 0

    rhs = B.dot(psi[1:-1])
    frames[index] = np.square(psi)


# Animation
fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)
plt.grid()


# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,


def animate(i):
    # Complex warning because imaginary is null but still exists
    line.set_data(x, frames[i])
    return line,


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)
anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

os.system("xdg-open basic_animation.mp4")
