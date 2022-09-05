import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.train_utils import get_filename
from utils.eval_utils import get_metric_name, METRIC_NAME_MAPPING
from scipy.interpolate import interp2d
sns.set_theme()




def plot_uncs_conditional(
    uncs,
    unc_names,
    config,
    unc_range = [0,1], 
    suffix=""
):
    """Plot histograms of unc2|unc1.
    """
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) +  \
        f"_{unc_names[0]}_{unc_names[1]}"\
        + suffix + ".pdf"
    path = os.path.join(save_dir, filename)

    # params to use later for plotting
    bins = 20

    # the two uncertainties
    unc1, unc2 = np.array(uncs[unc_names[0]]), np.array(uncs[unc_names[1]])


    # reset the color cycle style
    # sns.set_theme()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    H, xedges, yedges = np.histogram2d(
        unc1, unc2, range=unc_range,density=True, bins=bins
    )

    H = np.stack(
        [H[i]/H[i].sum() if H[i].sum() > 0 else H[i] for i in range(len(H))]
    )
    # H = np.log(H)
    H = H.T # for display purposes
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H, cmap="Blues")
    ax.set_xlabel(unc_names[0])
    ax.set_ylabel(unc_names[1])
    fig.tight_layout()
    fig.savefig(path)
    print(f"figure saved to:\n{path}")

def plot_uncs_conditional_together(
    uncs_list,
    data_names,
    unc_names,
    config,
    unc_range = [0,1], 
    suffix=""
):
    """Plot histograms of unc2|unc1.
    """
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) +  \
        f"_{unc_names[0]}_{unc_names[1]}"\
        + suffix + "_tog.pdf"
    path = os.path.join(save_dir, filename)

    # params to use later for plotting
    bins = [10,20]
    n = len(data_names)
    fig, axes = plt.subplots(2, n, figsize=(10, 5), sharey="row", sharex="col")
    # the two uncertainties
    for i, ax in enumerate(axes[0]):
        unc1= np.array(uncs_list[i][unc_names[0]])
        unc2= np.array(uncs_list[i][unc_names[1]])
        

        H, xedges, yedges = np.histogram2d(
            unc1, unc2, range=unc_range,density=True, bins=bins
        )

        # approx conditional, normalise by unc1
        # zero if nothing there
        H = np.stack(
            [H[i]/H[i].sum() if H[i].sum() > 0 else H[i] for i in range(len(H))]
        )
        # H = np.log(H)
        H = H.T # for display purposes
        X, Y = np.meshgrid( xedges, yedges
        )

        ax.pcolormesh(
            X, Y, H, 
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidth=0, rasterized=True, vmin=0
        )

        if i == 0:
            ax.set_ylabel(f"{METRIC_NAME_MAPPING[unc_names[1]]} | {METRIC_NAME_MAPPING[unc_names[0]]}")
        ax.set_title(data_names[i].replace("_", " "))
    contours = lambda x,y:x+y
    for i, ax in enumerate(axes[1]):
        unc1 = np.array(uncs_list[i][unc_names[0]])
        unc2 = np.array(uncs_list[i][unc_names[1]])

        sns.kdeplot(
            ax=ax,
            x=unc1,
            y=unc2,
            alpha=0.7,
            levels=7,
            fill=True,
            cut=0
        )
       
        ax.set_xlabel(METRIC_NAME_MAPPING[unc_names[0]])
        if i == 0:
            ax.set_ylabel(
                f"{METRIC_NAME_MAPPING[unc_names[1]]}")
        ax.set_ylim(ymax=1)

        print(data_names[i])

    for i, ax in enumerate(axes[1]):
        x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
        y = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 1000)
        X, Y = np.meshgrid(x, y)
        Z = contours(X, Y)
        # spatially subsample, so contours look nicer but are not even in Z
        x_l = x[199:800:100]
        y_l = y[199:800:100]
        levels = contours(x_l, y_l)
        ax.contour(
            X, Y, Z, 10, alpha=0.9, levels=levels
        )
    fig.tight_layout()
    fig.savefig(path)
    print(f"figure saved to:\n{path}")
