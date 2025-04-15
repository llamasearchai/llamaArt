import matplotlib.pyplot as plt


def plot_agent_trajectories(trajectories):
    for traj in trajectories:
        plt.plot(traj)
    plt.show()
