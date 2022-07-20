from representational_similarity import logger
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import pandas as pd
sns.set(rc={'figure.figsize': (12, 10)}, font_scale=2.8)
sns.set_style("whitegrid", {'axes.grid': False, 'legend.labelspacing': 1.2})


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


def plot_cka(res, cfg):
    # When we have FC/conv + activation function, we only keep the activation function.
    # We also drop activations from dropout and reshape layers as they are not very informative.
    logger.debug("Pre-processing: {}".format(res))
    cols_to_keep = {"Encoder_2": "Encoder_1", "Encoder_4": "Encoder_2", "Encoder_6": "Encoder_3",
                    "Encoder_8": "Encoder_4", "Encoder_11": "Encoder_5", "Encoder_12": "Encoder_mean",
                    "Encoder_13": "Encoder_variance", "Transition_2": "Transition_1", "Transition_4": "Transition_2",
                    "Transition_5": "Transition_mean", "Transition_6": "Transition_variance",
                    "Critic_2": "Critic_1", "Critic_4": "Critic_2", "Critic_6": "Critic_3",
                    "Critic_7": "Critic_4", "Policy_2": "Policy_1", "Policy_4": "Policy_2", "Policy_6": "Policy_3",
                    "Policy_7": "Policy_4"}
    rows_to_keep = cols_to_keep
    if res.index.name == "DQN":
        rows_to_keep = {"Policy_2": "Value_1", "Policy_4": "Value_2", "Policy_6": "Value_3",
                        "Policy_8": "Value_4", "Policy_11": "Value_5", "Policy_14": "Value_6"}
    if res.columns.name == "DQN":
        cols_to_keep = {"Policy_2": "Value_1", "Policy_4": "Value_2", "Policy_6": "Value_3",
                        "Policy_8": "Value_4", "Policy_11": "Value_5", "Policy_14": "Value_6"}
    df = res.loc[res.index.isin(rows_to_keep.keys()), res.columns.isin(cols_to_keep.keys())]
    df.rename(columns=cols_to_keep, index=rows_to_keep, inplace=True)
    logger.debug("Dataframe after pre-processing: {}".format(df))
    ax = sns.heatmap(df, vmin=0, vmax=1)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    (ax.set_ylabel("Model={}".format(cfg.a1_name.upper())) if cfg.a1_action is None
     else ax.set_ylabel("Model={}, Action={},\n Gain={}".format(cfg.a1_name.upper(),
                                                              cfg.a1_action.replace("epsilon", '\u03B5'), cfg.a1_gain)))
    (ax.set_xlabel("Model={}".format(cfg.a2_name.upper())) if cfg.a2_action is None
     else ax.set_xlabel("Model={}, Action={},\n Gain={}".format(cfg.a2_name.upper(),
                                                              cfg.a2_action.replace("epsilon", '\u03B5'), cfg.a2_gain)))
    save_figure(cfg.save_file)


def plot_corr(res, m1, m2, save_path):
    res = pd.DataFrame(res).T
    # Save csv with m1 layers as header, m2 layers as indexes
    res = res.rename_axis(m2.upper(), axis="columns")
    res = res.rename_axis(m1.upper())
    res.to_csv("{}.tsv".format(save_path), sep="\t")
    ax = sns.heatmap(res, vmin=-1, vmax=1, annot_kws={"fontsize": 13})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    save_figure("{}.pdf".format(save_path))


def plot_distrib(acts, actions, l, save_path):
    df = pd.DataFrame(acts).add_prefix("Latent variable at index ")
    df["Action"] = actions
    df["Action"] = df["Action"].replace({0: "Down", 1: "Up", 2: "Left", 3: "Right"})
    df = drop_outliers(df)
    df.to_csv("{}_{}.tsv".format(save_path, l), sep="\t", index=False)
    for i in range(acts.shape[1]):
        plt.figure()
        sns.histplot(data=df, x="Latent variable at index {}".format(i), hue="Action", multiple="stack")
        save_figure("{}_{}_latent_{}.pdf".format(save_path, l, i))


def drop_outliers(df, z_thresh=1.5):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', np.number]
    # This a slightly updated version of https://stackoverflow.com/a/56725366
    f = lambda x: np.abs(stats.zscore(x)) < z_thresh
    constrains = df.select_dtypes(include=numerics).apply(f).all(axis=1)
    return df.drop(df.index[~constrains])
