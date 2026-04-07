# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## The Verlet Algorithm
# The final numerical method for the integration of ODEs that we will implement
# in this class is the *Verlet Algorithm*, which is often stated explicitly in
# term of kinematic state variables and acceleration:
#
# $$x_{n+1} = x_n + v_n\Delta t + \frac{1}{2}a_n\Delta t^2$$
# $$v_{n+1} = v_n + (a_{n+1}+a_n)\Delta t $$
#
# To make use of this algorithm, you will have to manage current acceleration,
# $a_n$, and next acceleration $a_{n+1}$ in order to resolve the second line.
# Note that it is assumed that acceleration is a function of position, so that
# to find the next acceleration you evaluate $a_{n+1}(x_{n+1})$.
#
# Write a function to do the Verlet update. It should accept a function to
# compute force as well as a datastructure *of your choice* that stores $x_n$,
# $v_n$, and $a_n$. At the end of the function's execution the data structure
# should be updated so that $x_n$, $v_n$, and $a_n$ are now $x_{n+1}$,
# $v_{n+1}$, and $a_{n+1}$. Consult the text and give careful thought to how
# the Verlet update will handle periodic boundary conditions on the force
# calculation. 

# %%
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

class VerletODE:

    def __init__(self, dt:float, state:dict, Lx:float | int, Ly:float | int):
        self.dt = dt
        self.state = state
        self.Lx = Lx
        self.Ly = Ly
        self.state['KE'] = 0.0
        self.state['PE'] = 0.0
        self.N = len(state['pos'])

    def getMeanTemp(self, steps) -> float:
        return self.state['KE'] / (self.N*(steps+1))

    def getMeanEnergy(self, steps) -> float:
        if (steps == 0): return 0.0
        return self.state['KE']/steps + self.state['PE']/steps

    def integrate(self, accMethod, dt, potentialMethod=None, *args):
        """Integrates one step using Verlet ODE method with rectangular box (Lx, Ly).

        Args:
            accMethod (callable): Function to compute accelerations. 
                                  Signature: accMethod(positions, Lx, Ly, *args)
            potentialMethod (callable, optional): Function to compute potential energy.
                                  Signature: potentialMethod(positions, Lx, Ly, *args)
            *args: Additional arguments passed to accMethod and potentialMethod.

        Returns:
            tuple: (new_positions, new_velocities, new_accelerations)
        """

        if (dt != self.dt): self.dt = dt

        positions = self.state['pos']
        velocities = self.state['vel']
        accelerations = self.state['acc']

        p_new = positions + velocities * self.dt + 0.5 * accelerations * self.dt**2
        p_new[:, 0] = np.mod(p_new[:, 0] + self.Lx / 2, self.Lx) - self.Lx / 2
        p_new[:, 1] = np.mod(p_new[:, 1] + self.Ly / 2, self.Ly) - self.Ly / 2

        if potentialMethod is not None:
            self.state['PE'] += potentialMethod(p_new, self.Lx, self.Ly, *args)

        a_new = accMethod(p_new, self.Lx, self.Ly, *args)

        v_new = velocities + 0.5 * (accelerations + a_new) * self.dt

        self.state['KE'] += np.sum(v_new ** 2)

        self.state['pos'] = p_new
        self.state['vel'] = v_new
        self.state['acc'] = a_new

        return p_new, v_new, a_new

    def solve_ode(self, dt, tspan:list, method, *args):
        steps = int((tspan[1] - tspan[0]) / dt)
        positions = [self.state['pos']]
        for step in range(steps):
            p_new, v_new, a_new = self.integrate(method, *args)
            positions.append(p_new)

        return np.array(positions)

# %% [markdown]
# ## Distances
# No matter the force we use, it will depend on the distances between
# particles. You've already worked some of this out wen you wrote `n-body`. I'd
# invite you to redo that function again, and try and do better in terms of
# performance. At core, the distances will be distances in the $x$, $y$, and
# $z$ directions, eg $\delta x_{i,j} = x_i = x_j$. The most naive way to write
# this would be:
#
# ```python
# for i in range (N): # N is number of particles:
#     for j in range(i+1,N):
#         dx = x[i] - x[j]
#         dy = y[i] - y[j]
#         dz = z[i] - z[j]
#         # Update periodic BC for image
#         # Store the results in an array
# ```
#
# But this is slow. Based on the data strucure you devised in the previous
# section, write the code to do this "all-pairs" loop as quickly as you can. I
# see two possibilities:
# * Keep the explicit loops and use `numba` to 'compile' the code
# * Use `numpy` to manage arrays of values that are manipulated to find all
# pairs of differences.
#
# I have done both. I have also tried something called `jax` and discovered
# that the overhead of setting up the code makes run times slower than either
# of the above cases, at least for ~100 particles.
#
# You will have to make accomodations here for the periodic boundary
# conditions, which is done by subtracting the length of the box from the
# co-ordinate differences the half length of the box.
#
# So that is easy to compare implementations, call this function
# `distance_matrix` and accept imputs of x,y,z. Run PBC on the distances you
# compute inside the function to implement the *image method* discussed in the
# text.

# %%
def allDistances(positions:NDArray, Lx: float, Ly: float):
    """Get distances between all particles.

    Args:
        positions: List of all particle positions in system.
        Lx: The box size in x.
        Ly: The box size in y.

    Returns:
        NDArray(N, N, d): The vector distance between all particles.
    """
    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]

    diff[:, :, 0] = diff[:, :, 0] - Lx * np.round(diff[:, :, 0] / Lx)
    diff[:, :, 1] = diff[:, :, 1] - Ly * np.round(diff[:, :, 1] / Ly)

    return diff.squeeze()

# pos1 = np.array([0, 1])
# pos2 = np.array([0, -1])
# pos3 = np.array([0, 0])
# print(allDistances(np.array([pos1, pos2, pos3]), 5, 5))

# %% [markdown]
# ## Improved Lennard-Jones
# Finally, using the `distance_matrix` function you just wrote, change the
# Lennard Jones force function you previously wrote to compute the forces for
# $N$ particles. Integrate the call to the function that determines forces
# (Lennard-Jones) into the Verlet algorithm you wrote in the first part of this
# assignment. Upon completion you should have code that can be wired together
# to:
#
# 1. You call a Verlet algorithm to evolve the state (positions and velocities)
# from $y_n$ to $y_{n+1}$. Acceleration is not part of the state, but should be
# maintained in whatever data structure stores state because it is needed in
# Verlet ($a_n$ and $a_{n+1}$)
# 2. After finding the next position, $x_{n+1}$,and updating the periodic
# boundary conditions, the Verlet algorithm calls a force function,
# Lennard-Jones in this case, to get the $a_{n+1}$.
# 3. The Lennard-Jones force function calls the `distance_matrix` function to
# find the distances between all pairs of particles. That function uses the
# *minimum-image approximation* (Figure 8.3) to ensure distances reflect the
# periodic boundary conditions. The `distance_matrix` is the most time
# consuming portion of the process, and deserves careful attention for
# optimization.
# 4. Lennard-Jones uses the distances returned by `distance_matrix` to compute
# the forces and return them.
# 5. Verlet uses the returned forces to find the next acceleration, $a_{n+1}$
# and get the next velocity state, $v_{n+1}$ using the Verlet algorithm.
#
# Do this and have this code base prepared to solve real problems for Tuesday,
# March 31.

# %%
def getLennardPotential(positions, Lx, Ly, sigma=1, epsilon=1):
    r_vec = allDistances(positions, Lx, Ly)
    r = np.linalg.norm(r_vec, axis=-1)

    mask = r > 1e-12
    r_safe = np.where(mask, r, 1.0)

    sig_r = sigma / r_safe
    factor = 4 * epsilon * (sig_r**12 - sig_r**6)

    potential = np.sum(np.where(mask, factor, 0.0))/2
    return potential

def getLennardForce(positions, Lx, Ly, sigma=1, epsilon=1):
    """Computes the force on all particles on each other.

    Args:
        positions (NDArray(N, d)): A list of each particle's position
        Lx (float | int): Size of the box in x.
        Ly (float | int): Size of the box in y.
        sigma (float | int): Lennard force parameter.
        epsilon (float | int): Lennard force parameter.

    Returns:
        NDArray(N,d): A matrix of forces on each particle.
    """
    r_vec = allDistances(positions, Lx, Ly)
    r = np.linalg.norm(r_vec, axis=-1, keepdims=True)

    mask = r > 1e-12
    r_safe = np.where(mask, r, 1.0)

    sig_r = sigma / r_safe
    factor = 24 * epsilon * (2 * sig_r**12 - sig_r**6) / r_safe

    force = np.where(mask, factor * (r_vec / r_safe), 0.0)

    return force.sum(axis=0)
#
# pos1 = np.array([2, 0])
# pos2 = np.array([-2, 0])
# pos3 = np.array([0, 2])
# print(getLennardPotential(np.array([pos1, pos2, pos3]), 10))

# %%
if __name__=="__main__":

    pos1 = np.array([2, 0])
    pos2 = np.array([-2, 0])
    pos3 = np.array([0, 2])
    pos4 = np.array([0, -2])
    zeros = np.zeros(2)
    state = {
        'pos': np.array([pos1, pos2, pos3, pos4]),
        'vel': np.array([zeros, zeros, zeros, zeros]),
        'acc': np.array([zeros, zeros, zeros, zeros])
    }
    L = 20
    verlet = VerletODE(0.01, state, L, L)
    positions = verlet.solve_ode(0.001, [0.0, 75.00], getLennardForce, L, state)
    plt.plot(positions[-1,0,0], positions[-1,0,1], 'o', markersize=5)
    plt.plot(positions[-1,1,0], positions[-1,1,1], 'o', markersize=5)
    plt.plot(positions[-1,2,0], positions[-1,2,1], 'o', markersize=5)
    plt.plot(positions[-1,3,0], positions[-1,3,1], 'o', markersize=5)
    plt.plot(positions[:,0,0], positions[:,0,1])
    plt.plot(positions[:,1,0], positions[:,1,1])
    plt.plot(positions[:,2,0], positions[:,2,1])
    plt.plot(positions[:,3,0], positions[:,3,1])


# %%
