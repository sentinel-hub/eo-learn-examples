import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from eolearn.core import FeatureType

def plot_timestamp_heatmap(timestamps, ax=None, **kwargs):
    if not ax:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = ax.get_figure()

    # create dataframe for complete time interval contained in timestamps
    first = min(timestamps)
    last = max(timestamps)
    month_first = first.replace(day=1)
    month_last = (last.replace(day=1) + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1)
    all_days = pd.date_range(month_first, month_last)

    df = pd.DataFrame(all_days, columns=["days"])
    df["count"] = [0.0] * len(all_days)
    df["days"] = df["days"].dt.round("d")
    df = df.set_index("days")

    # create df for observations
    days = [ts.strftime("%Y-%m-%d") for ts in timestamps]
    uniq, count = np.unique(days, return_counts=True)
    df_counts = pd.DataFrame(zip(uniq.tolist(), count.tolist()), columns=["days", "count"])
    df_counts["days"] = pd.to_datetime(df_counts["days"], format='%Y-%m-%d')
    df_counts = df_counts.set_index("days")
    df.update(df_counts, join="left")

    # group by month in order and extract rows
    g = df.groupby(pd.Grouper(level='days', freq='M'))
    grid = np.zeros((1,31), dtype=float)
    y_labels = []
    for key, val in g:
        y_labels.append(key.strftime("%Y-%m"))
        as_numpy = val.to_numpy(dtype=grid.dtype).transpose()
        padded = np.pad(as_numpy, ((0,0), (0, 31-as_numpy.shape[1])), "constant", constant_values=np.nan)
        grid = np.vstack((grid, padded))
    grid = grid[1:, ...]

    # finally plot
    ax.imshow(grid)

    ax.set_xticks(np.arange(grid.shape[1]), labels=np.arange(grid.shape[1])+1)
    ax.set_yticks(np.arange(grid.shape[0]), labels=y_labels)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, None if np.isnan(grid[i,j]) else int(grid[i, j]), ha="center", va="center", color="w")

    fig.tight_layout()
    return fig


def visualize_time_intervals(timestamps, references, axs=None, **kwargs):
    if not axs:
        fig, axs = plt.subplots(2, 1, **kwargs)
    else:
        fig = axs[0].get_figure()
    plot_timestamp_heatmap(timestamps, ax=axs[0], **kwargs)
    axs[0].set_title("Training data timestamps")
    plot_timestamp_heatmap(references, ax=axs[1], **kwargs)
    axs[1].set_title("Reference data timestamps")
    plt.show()

    return fig


def visualize_timestamp_deltas(diffs, target_delta):
    hist_y, hist_x, bars = plt.hist(diffs, bins=np.arange(min(diffs) - 1, max(diffs) + 1), align="left")
    plt.title("Timestamp deltas")
    plt.xlabel("difference between observations [days]")
    plt.ylabel("count")
    plt.vlines(target_delta.days, 0, max(hist_y), "red", ls="--", label="time_difference")
    plt.legend()
    for y, x, b in zip(hist_y, hist_x, bars):
        if y > 0: plt.annotate(str(int(y)), (x, y + 0.01 * max(hist_y)), ha="center")


def plot_samples(
        eopatches,
        feature_data=(FeatureType.DATA, "reference_data"),
        feature_reference=(FeatureType.DATA, "reference"),
        feature_mask=(FeatureType.MASK, "mask_reference"),
        figscale=5
):
    n_col = 3  # true color, reference, mask
    fig, axs = plt.subplots(len(eopatches), n_col, figsize=(figscale * n_col, figscale * len(eopatches)))
    plt.subplots_adjust(wspace=.3, hspace=.1)
    if len(eopatches) == 1: axs = np.expand_dims(axs, 0)

    for patch, ax_row in zip(eopatches, axs):
        ax_tc, ax_ndvi, ax_mask = ax_row

        if feature_data in patch:
            data = np.flip(patch[feature_data][0, :, :, :3], 2) * 3.5   # rescale true color for realistic appearance
            ax_tc.imshow(data)
            ax_tc.set_title("true color reference")
        if feature_reference in patch:
            reference = patch[feature_reference][0, ...]
            ax_ndvi.imshow(reference, cmap="Greens")
            ax_ndvi.set_title("reference")
        if feature_mask in patch:
            mask = patch[feature_mask][0, ...]
            pcolor = ax_mask.imshow(mask, vmin=0, vmax=1)
            cbar = plt.colorbar(pcolor, ax=ax_mask, aspect=10, fraction=.09,
                                boundaries=[0,1,2], values=[0,1], ticks=[.5, 1.5])
            cbar.ax.set_yticklabels(["0: invalid", "1: valid"])
            ax_mask.set_title("mask")

    return fig
