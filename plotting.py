import matplotlib.pyplot as plt

def plot_cost_accuracy(accuracy_list, cost_list):
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.plot(cost_list, color=color)
    ax1.set_xlabel("epoch", color=color)
    ax1.set_ylabel("Cost", color=color)
    ax1.tick_params(axis="y", color=color)
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("accuracy", color=color)
    ax2.set_xlabel("epoch", color=color)
    ax2.plot(accuracy_list, color=color)
    ax2.tick_params(axis="y", color=color)
    fig.tight_layout()
    plt.show()