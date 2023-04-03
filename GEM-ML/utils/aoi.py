from typing import Iterable

import numpy as np
import natsort
from sentinelhub.areas import BBoxSplitter


def split_bbox_list(bbox_splitter: BBoxSplitter,
                    percentages: Iterable[float],
                    mode: str = "v",
                    reversed: bool = False):
    """
    Splits a list of bboxes according to the given percentages. Note that only whole rows/columns are assigned to
    splits, hence resulting percentage may vary from the target percentages.
    :param bbox_splitter: An instance of AreaSplitter
    :param percentages: A list of percentages, defining the expected size of the splits
    :param mode: Defines orientation of a hypothetical splitting line: 'h' = splitting line horizontal, splitting the
        AOI from north to south, 'v' = splitting line vertical, splitting the AOI from west to east
    :param reversed: reverse the splitting direction (west to east, or south to north)
    :return: list of list containing the bounding boxes, in order of the percentages
    """
    if sum(percentages) != 1: raise ValueError("Sum of percentages must be 1!")
    perc_cum = np.cumsum(percentages)

    bbox_list = bbox_splitter.get_bbox_list()
    info_list = bbox_splitter.get_info_list()

    out = []
    start_ind = 0
    if mode == "v":
        key = "index_x"
    elif mode == "h":
        # reorder bounding boxes
        order = natsort.index_natsorted([f'{x["index_y"]}.{x["index_x"]}' for x in info_list])
        bbox_list = np.array(bbox_list)[order].tolist()
        key = "index_y"
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    elements = [info[key] for info in info_list]
    _, counts = np.unique(elements, return_counts=True)

    if reversed:
        counts = counts[::-1]
        bbox_list = bbox_list[::-1]

    cumulative = np.cumsum(counts)
    cumulative_rel = cumulative/np.sum(counts)
    for perc in perc_cum[:-1]:
        ind = np.argmax(cumulative_rel>perc)
        end_ind = cumulative[ind]
        out.append(bbox_list[start_ind:end_ind])
        start_ind = end_ind
    # add last split
    out.append(bbox_list[start_ind:])

    return out
