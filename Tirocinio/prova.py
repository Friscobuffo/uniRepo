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
    

    # algorithm core
    for _ in range(iterations):
        plot(positions)
        

result = pso_visual()
plt.show()