import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':

    jump = 500
    training_iter = 500000
    indices = [i * jump for i in range(0, int(training_iter / jump))]

    swarm_plot = False
    g_value = "efe_3"
    df = pd.read_csv(f'./data/ActionPicked_{g_value}.csv' if swarm_plot else f'./data/EntropyPriorActions_{g_value}.csv')
    df = df.filter(items=indices, axis=0)

    # Set custom color palette
    colors = ["#d65f5f", "#ee854a", "#4878d0", "#6acc64"]
    sns.set_palette(sns.color_palette(colors))

    sns.set_theme(style="whitegrid", palette="muted")
    if swarm_plot:
        # Draw a categorical scatter plot to show each observation.
        ax = sns.swarmplot(data=df, x="Training iterations", y="Actions", order=["Down", "Right", "Left", "Up"])
        ax.set(ylabel="")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(f"./data/ActionPicked_{g_value}.pdf")
        plt.show()
    else:
        # Draw a categorical scatter plot to show each observation
        ax = sns.lineplot(data=df, x="Training iterations", y="Entropy")
        ax.set(ylabel="Entropy of prior over actions")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(f"./data/EntropyPriorActions_{g_value}.pdf")
        plt.show()
