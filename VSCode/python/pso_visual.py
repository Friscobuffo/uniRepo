import numpy as np
import matplotlib.pyplot as plt

def eval(x): # alpine n.2
    x1 = x[0]
    x2 = x[1]
    return -(np.sqrt(np.abs(x1)) * np.sin(x1) * np.sqrt(np.abs(x2)) * np.sin(x2))

swarmsize = 30
iterations = 50
omega = 0.35 # inertia
c1 = 0.5 # cognitive constant (lambda 1)
c2 = 0.5 # social constant (lambda 2)
goal = [7.917, 7.917]

#plot stuff
X = np.linspace(0, 10, 128)
Y = np.linspace(0, 10, 128)
Z = eval((np.meshgrid(X, Y, sparse = True)))
levels = np.linspace(np.min(Z), np.max(Z), 12)

def plot(positions):
    x = positions[:, 0]
    y = positions[:, 1]
    plt.figure("Particle Swarm Optimization")
    plt.clf()
    plt.contour(X, Y, Z, levels=levels)
    plt.scatter(x, y)
    plt.scatter(goal[0], goal[1], s=50, marker="x", color="red")
    plt.pause(0.05)

def pso_visual():
    lb = np.array([0, 0])
    ub = np.array([10, 10])
    
    # set lower and upper bounds to velocities based on position bounds
    upper_bound_velocity = np.abs(ub - lb)
    lower_bound_velocity = -upper_bound_velocity

    # initialize particles positions randomly in the function bounds
    positions = np.random.rand(swarmsize, 2)  # particles position
    positions = lb + positions * (ub - lb)
    best_particles_positions = positions.copy()  # best known position per particle
    evals = np.empty(swarmsize)  # evaluation of each particle
    # evaluating each particle
    for i in range(swarmsize):
        evals[i] = eval(positions[i])
    best_particles_evals = evals.copy()

    i_min = np.argmin(best_particles_evals) # index of best eval
    best_swarm_eval = best_particles_evals[i_min].copy()
    best_swarm_position = best_particles_positions[i_min].copy()

    # initial velocity vector
    velocities = lower_bound_velocity + np.random.rand(swarmsize, 2) * (
            upper_bound_velocity - lower_bound_velocity)

    # algorithm core
    for _ in range(iterations):
        plot(positions)
        
        # update velocity vector with slight randomization to approach minimum
        rp = np.random.uniform(size=(swarmsize, 2))  # relative to personal best
        rg = np.random.uniform(size=(swarmsize, 2))  # relative to global best
        # velocity of each particle
        velocities = omega * velocities + c1 * rp * (best_particles_positions - positions) \
            + c2 * rg * (best_swarm_position - positions)
        # update position vector
        positions = positions + velocities

        # prevent out of bounds
        lower_mask = positions < lb
        upper_mask = positions > ub

        # if particle position out of bounds, it get placed the edge
        positions = positions * (~np.logical_or(lower_mask, upper_mask)) \
            + lb * lower_mask + ub * upper_mask

        # update evaluation of each particle
        for i in range(swarmsize):
            evals[i] = eval(positions[i])

        # update best of each particle
        i_update = evals < best_particles_evals
        best_particles_positions[i_update, :] = positions[i_update, :].copy()
        best_particles_evals[i_update] = evals[i_update]

        # compare swarm best position with global best position
        i_min = np.argmin(best_particles_evals)
        if best_particles_evals[i_min] < best_swarm_eval:
            best_swarm_position = best_particles_positions[i_min].copy()
            best_swarm_eval = best_particles_evals[i_min]
    
    return best_swarm_position

result = pso_visual()
plt.show()