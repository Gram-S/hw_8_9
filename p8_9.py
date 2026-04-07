import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from numpy.typing import NDArray
from verlet import VerletODE, getLennardForce, getLennardPotential
import math

def initLaticePos(L, N):
    n = int(math.sqrt(N))
    half_L = L/2
    dx = L/n
    dy = L/n

    positions = []

    for i in range(n):
        for j in range(n):
            x = dx * i - half_L
            y = dy * j - half_L
            positions.append([x, y])

    return np.array(positions)

def initRectanglePos(Lx, nx, Ly, ny, N, startL=False):
    half_Lx = Lx/2
    half_Ly = Ly/2
    dx = Lx/nx
    dy = Ly/ny
    if (startL):
        dx *= 0.5

    positions = []

    for i in range(nx):
        for j in range(ny):
            x = dx * i - half_Lx
            y = dy * j - half_Ly
            positions.append([x, y])

    return np.array(positions)

def initTriangleLatice(Lx, nx, Ly, ny, N, startL=False):
    half_Lx = Lx/2
    half_Ly = Ly/2
    dx = Lx/nx
    dy = Ly/ny
    if (startL):
        dx *= 0.5

    positions = []

    for i in range(nx):
        for j in range(ny):
            x = dx * i - half_Lx
            y = dy * j - half_Ly
            positions.append([x, y])

    return np.array(positions)

def getParticlesOnLeft(positions):
    pos_x = positions[:,0]
    return len(pos_x[pos_x < 0])

def initVelocities(N, T):
    velocities = np.random.uniform(-.5, .5, size=(N, 2))
    vel_m = np.mean(velocities, axis=0)
    velocities -= vel_m

    scale = T / ( 1/(2*(N-1)) * np.sum(velocities[:,0]**2 + velocities[:,1]**2) )

    velocities *= scale

    return velocities

def getInitState(initialization):
    if initialization == 'two':
        L = 2.9
        dist = 1
        vel = .2
        dt = 0.01
        state = {
            'pos': np.array([
                np.array([-dist,0.]),
                np.array([dist,0.])
            ]),
            'vel': np.array([
                np.array([vel,0.]),
                np.array([-vel,0.])
            ]),
        }
        state['acc'] = getLennardForce(state['pos'], L, L)
        return L, L, state, dt, False

    elif initialization == 'three':
        L = 3.0
        dist = 1
        vel = .2
        dt = 0.01
        state = {
            'pos': np.array([
                np.array([0.,dist*math.sqrt(3)/2]),
                np.array([-dist,0.]),
                np.array([dist,0.])
            ]),
            'vel': np.array([
                np.array([0.,-vel]),
                np.array([vel,0.]),
                np.array([-vel,0.])
            ]),
        }
        state['acc'] = getLennardForce(state['pos'], L, L)
        return L, L, state, dt, False

    elif initialization == 'four':
        L = 3.0
        dist = 1
        vel = .2
        dt = 0.01
        state = {
            'pos': np.array([
                np.array([-dist,0.]),
                np.array([dist,0.]),
                np.array([0.,dist]),
                np.array([0.,-dist])
            ]),
            'vel': np.array([
                np.array([vel,0.]),
                np.array([-vel,0.]),
                np.array([0.,-vel]),
                np.array([0.,vel])
            ]),
        }
        state['acc'] = getLennardForce(state['pos'], L, L)
        return L, L, state, dt, False

    elif initialization == 'p8.3a':
        L = 10.0
        N = 64
        T = 1.0
        dist = 1
        vel = .2
        dt = 0.001
        state = {
            'pos': initLaticePos(L, N),
            'vel': initVelocities(N, T)
        }
        state['acc'] = getLennardForce(state['pos'], L, L)
        return L, L, state, dt, False

    elif initialization == 'p8.3c':
        Lx = 20
        nx = 8
        Ly = 10
        ny = 8
        N = 64
        T = 1.0
        dt = 0.005
        state = {
            'pos': initRectanglePos(Lx, nx, Ly, ny, N, startL=True),
            'vel': initVelocities(N, T)
        }
        state['acc'] = getLennardForce(state['pos'], Lx, Ly)
        return Lx, Ly, state, dt, False

    elif initialization == 'p8.4a':
        L = 10.0
        dt = 0.01
        N = 11

        y_positions = np.linspace(-4.5, 4.5, N)
        state = {
            'pos': np.column_stack((np.zeros(N), y_positions)),
            'vel': np.column_stack((np.ones(N), np.zeros(N)))
        }

        state['acc'] = getLennardForce(state['pos'], L, L)

        return L, L, state, dt, False

    elif initialization == 'p8.4b':
        L = 10.0
        dt = 0.01
        N = 11

        y_positions = np.linspace(-4.5, 4.5, N)
        state = {
            'pos': np.column_stack((np.zeros(N), y_positions)),
            'vel': np.column_stack((np.ones(N), np.zeros(N)))
        }
        state['vel'][5] = np.array([0.99999, 0.00001])
        state['acc'] = getLennardForce(state['pos'], L, L)

        return L, L, state, dt, False

    elif initialization == 'p8.4c':
        L = 10.0
        N = 64
        T = 1.0
        dt = 0.001
        state = {
            'pos': initLaticePos(L, N),
            'vel': initVelocities(N, T)
        }
        state['acc'] = getLennardForce(state['pos'], L, L)
        return L, L, state, dt, True

    elif initialization == 'p8.9a':
        Lx = 8.0
        Ly = 9.0
        nx = 8
        ny = 8
        dx = Lx / nx
        dy = Ly / ny
        T = 1.0
        dt = 0.001
        state = {
            'pos': initLaticePos(L, N),
            'vel': initVelocities(N, T)
        }
        state['acc'] = getLennardForce(state['pos'], L, L)
        return L, L, state, dt, True

    # elif initialization == 'six':
    else:
        L = 5.0
        dist = 1
        vel = .2
        dt = 0.01
        state = {
            'pos': np.array([
                np.array([0.,dist]),
                np.array([0.,-dist]),
                np.array([dist/math.sqrt(2),dist/math.sqrt(2)]),
                np.array([-dist/math.sqrt(2),dist/math.sqrt(2)]),
                np.array([-dist/math.sqrt(2),-dist/math.sqrt(2)]),
                np.array([dist/math.sqrt(2),-dist/math.sqrt(2)]),
            ]),
            'vel': np.array([
                np.array([0.,-vel]),
                np.array([0.,vel]),
                np.array([-vel/math.sqrt(2),-vel/math.sqrt(2)]),
                np.array([vel/math.sqrt(2),-vel/math.sqrt(2)]),
                np.array([vel/math.sqrt(2),vel/math.sqrt(2)]),
                np.array([-vel/math.sqrt(2),vel/math.sqrt(2)]),
            ]),
        }
        state['acc'] = getLennardForce(state['pos'], L, L)
        return L, L, state, dt, False

# pos1 = np.array([2, 0])
# pos2 = np.array([-2, 0])
# pos3 = np.array([0, 2])
# pos4 = np.array([0, -2])
# zeros = np.zeros(2)
# state = {
#     'pos': np.array([pos1, pos2, pos3, pos4]),
#     'vel': np.array([zeros, zeros, zeros, zeros]),
#     'acc': np.array([zeros, zeros, zeros, zeros])
# }
# L = 10

Lx, Ly, state, dt, invert = getInitState('p8.4c')
verlet = VerletODE(dt, state, Lx, Ly)

fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2)
ax_pos = fig.add_subplot(gs[0, 0])
ax_left = fig.add_subplot(gs[0, 1])
ax_energy = fig.add_subplot(gs[1, 0])
ax_temp = fig.add_subplot(gs[1, 1])

ax_pos.set_xlim(-Lx/2, Lx/2)
ax_pos.set_ylim(-Ly/2, Ly/2)
ax_pos.set_aspect('equal')
ax_pos.set_title('Lennard-Jones Particles')

circles = [
    Circle(xy, radius=0.5*2**(1/6), color='green', alpha=0.8)
    for xy in state['pos']
]
for circle in circles:
    ax_pos.add_patch(circle)

# Energy plot
ax_energy.set_title('Total Energy vs Time')
ax_energy.set_xlabel('Time')
ax_energy.set_ylabel('Total Energy')
energies = []
energy_line, = ax_energy.plot([], [], 'b-', lw=2)
ax_energy.set_xlim(0, 10)
ax_energy.set_ylim(0, 10)

# Particles on Left plot
ax_left.set_title('Particles on Left vs Time')
ax_left.set_xlabel('Time')
ax_left.set_ylabel('Particles on Left')
left_particles = []
left_line, = ax_left.plot([], [], 'r-', lw=2)
ax_left.set_xlim(0, 10)
ax_left.set_ylim(0, state['pos'].shape[0] + 5)

# Temperature plot
ax_temp.set_title('Mean Temperature vs Time')
ax_temp.set_xlabel('Time')
ax_temp.set_ylabel('Mean Temperature')
temperatures = []
temp_line, = ax_temp.plot([], [], 'r-', lw=2)
ax_temp.set_xlim(0, 10)
ax_temp.set_ylim(0, 5)

times = []

def update(frame):
    p_new, v_new, a_new = verlet.integrate(getLennardForce, dt if not invert or frame < 1000 else -dt, getLennardPotential)
    state['pos'], state['vel'], state['acc'] = p_new, v_new, a_new

    for circle, pos in zip(circles, state['pos']):
        circle.set_center(pos)

    current_time = frame * dt
    current_energy = verlet.getMeanEnergy(frame)
    current_temp = verlet.getMeanTemp(frame)
    current_left = getParticlesOnLeft(p_new)

    times.append(current_time)
    energies.append(current_energy)
    temperatures.append(current_temp)
    left_particles.append(current_left)

    energy_line.set_data(times, energies)
    left_line.set_data(times, left_particles)
    temp_line.set_data(times, temperatures)

    if len(times) > 5:
        t_max = max(times) * 1.05

        ax_energy.set_xlim(0, t_max)
        ax_energy.set_ylim(min(energies) * 0.98, max(energies) * 1.02)

        ax_left.set_xlim(0, t_max)
        ax_left.set_ylim(min(left_particles) * 0.98, max(left_particles) * 1.02)

        ax_temp.set_xlim(0, t_max)
        ax_temp.set_ylim(min(temperatures) * 0.98, max(temperatures) * 1.02)

    return circles + [energy_line, left_line, temp_line]


ani = animation.FuncAnimation(
    fig, 
    update, 
    interval=dt*100,
    blit=True,
    cache_frame_data=False
)

plt.tight_layout()
plt.show()
