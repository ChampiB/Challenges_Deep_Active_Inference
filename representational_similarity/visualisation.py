from representational_similarity import logger
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def save_figure(out_fname, dpi=300, tight=True):
    """ Save a matplotlib figure in an `out_fname` file.

    :param str out_fname: Name of the file used to save the figure.
    :param int dpi: Number of dpi, Default 300.
    :param bool tight: If True, use plt.tight_layout() before saving. Default True.
    """
    if tight is True:
        plt.tight_layout()
    plt.savefig(out_fname, dpi=dpi, transparent=True)
    plt.clf()
    plt.cla()
    plt.close()


def plot_cka(res, save_path):
    # When we have FC/conv + activation function, we only keep the activation function.
    # We also drop activations from dropout and reshape layers as they are not very informative.
    logger.debug("Dataframe before pre-processing: {}".format(res))
    cols_to_keep = {"Encoder_2": "Encoder_1", "Encoder_4": "Encoder_2", "Encoder_6": "Encoder_3",
                    "Encoder_8": "Encoder_4", "Encoder_11": "Encoder_5", "Encoder_12": "Encoder_6",
                    "Encoder_13": "Encoder_7", "Transition_2": "Transition_1", "Transition_4": "Transition_2",
                    "Transition_5": "Transition_3", "Transition_6": "Transition_4",
                    "Critic_2": "Critic_1", "Critic_4": "Critic_2", "Critic_6": "Critic_3",
                    "Critic_7": "Critic_4", "Policy_2": "Policy_1", "Policy_4": "Policy_2", "Policy_6": "Policy_3",
                    "Policy_7": "Policy_4"}
    rows_to_keep = cols_to_keep
    if res.index.name == "DQN":
        rows_to_keep = {"Policy_2": "Policy_1", "Policy_4": "Policy_2", "Policy_6": "Policy_3",
                        "Policy_8": "Policy_4", "Policy_11": "Policy_5", "Policy_14": "Policy_6"}
    if res.columns.name == "DQN":
        cols_to_keep = {"Policy_2": "Policy_1", "Policy_4": "Policy_2", "Policy_6": "Policy_3",
                        "Policy_8": "Policy_4", "Policy_11": "Policy_5", "Policy_14": "Policy_6"}
    df = res.loc[res.index.isin(rows_to_keep.keys()), res.columns.isin(cols_to_keep.keys())]
    df.rename(columns=cols_to_keep, index=rows_to_keep, inplace=True)
    logger.debug("Dataframe after pre-processing: {}".format(df))
    ax = sns.heatmap(df, vmin=0, vmax=1, annot_kws={"fontsize": 13})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    save_figure("{}.pdf".format(save_path))


def plot_corr(res, m1, m2, save_path):
    res = pd.DataFrame(res).T
    # Save csv with m1 layers as header, m2 layers as indexes
    res = res.rename_axis(m2.upper(), axis="columns")
    res = res.rename_axis(m1.upper())
    res.to_csv("{}.tsv".format(save_path), sep="\t")
    ax = sns.heatmap(res, vmin=-1, vmax=1, annot_kws={"fontsize": 13})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    save_figure("{}.pdf".format(save_path))


def plot_distrib(acts, l, save_path):
    df = pd.DataFrame(acts).add_prefix("latent_")
    sns.pairplot(df)
    save_figure("{}_{}.pdf".format(save_path, l))
    df.to_csv("{}_{}.tsv".format(save_path, l), sep="\t", index=False)
