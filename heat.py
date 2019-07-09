import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.linalg import toeplitz
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigs
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
# TODO: Use a faster method to solve the linear system (i.e.: LU Decomposition)
# TODO: Fix boundary conditions

alpha = 1  # Thermal diffusivity

# Spatial discretisation
N = 100
x = np.linspace(0, 1, N)
dx = x[1] - x[0]


# Time discretisation
K = 1000
t = np.linspace(0, 1, K)
dt = t[1] - t[0]

Fo = - alpha * dt / (2 * (dx ** 2))  # Fourier number Fo
print("Fourier adim number is:\t{}".format(Fo))
# Building Matrices
A = sparse.csc_matrix(toeplitz([1 - 2 * Fo,   Fo, *np.zeros(N-2)]))  # 2 less for both boundaries
B = sparse.csc_matrix(toeplitz([1 + 2 * Fo, - Fo, *np.zeros(N-2)]))

eigenvalues = eigs(sparse.linalg.inv(A).dot(B))[0]
if max(eigenvalues) <= 1:
    print("Looks stable.\nCalculating..")
else:
    print("Unstable matrix eigenvalue:\t{}".format(max(eigenvalues)))

# Initial and boundary conditions
psi = np.array([*np.zeros(int(N / 4)), *np.ones(int(N / 2)), *np.zeros(int(N / 4))])  # [:, None]  # new axis)
# plt.plot(x, psi)
# plt.show()
b = B.dot(psi)
psi[0], psi[-1] = 0, 1

# Time propagation
frames = np.zeros([K, N])
for index, step in enumerate(t):
    # Enforce boundaries
    psi[0], psi[-1] = 0, 1
    # Within the domain
    psi = spsolve(A, b)
    # Right hand side
    b = B.dot(psi)

    frames[index] = psi

# Animation
fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)
plt.grid()
plt.style.use('classic')

im = plt.imshow(line)
plt.colorbar()

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,


def animate(i):
    line.set_data(x, frames[i])
    return line,


print("animating..")

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=K, interval=1000/30, blit=True)
anim.save('basic_animation_heat.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

os.system("xdg-open basic_animation_heat.mp4")
