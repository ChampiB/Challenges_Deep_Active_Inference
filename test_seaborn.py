import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    sns.set_theme(style="whitegrid", palette="muted")

    # Load the penguins dataset
    df = sns.load_dataset("penguins")
    print(type(df))
    print(df)

    # Draw a categorical scatter plot to show each observation
    ax = sns.swarmplot(data=df, x="body_mass_g", y="sex", hue="species")
    ax.set(ylabel="")

    plt.savefig('save_as_a_png.png')
    plt.show()
