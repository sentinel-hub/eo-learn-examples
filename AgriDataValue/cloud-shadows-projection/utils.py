from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import label

from eolearn.core import EOPatch, FeatureType
from sentinelhub import bbox_to_resolution


def cv2_disk(radius: int) -> np.ndarray:
    """Recreates the disk structural element from skimage.morphology using OpenCV."""
    return cv2.circle(
        np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8), (radius, radius), radius, color=1, thickness=-1
    )


def dilate_mask(mask: np.ndarray, dilation_size: int) -> np.ndarray:
    """Dilates the mask using a disk structural element."""
    dilation_filter = cv2_disk(dilation_size)
    return cv2.dilate(mask.astype(np.uint8), dilation_filter).astype(bool)


def get_hillshade_mask(eop: EOPatch, t_idx: int, dilation_size: int) -> np.ndarray:
    """Calculates the hillshade mask for the given EOPatch and time index. The hillshade mask can be dilated."""
    dem_feature = (FeatureType.DATA_TIMELESS, "DEM")
    height, width = eop.get_spatial_dimension(*dem_feature)
    resx, resy = bbox_to_resolution(eop.bbox, height=height, width=width)

    dem = eop[dem_feature].squeeze(-1)
    sun_zenith = eop.data["sunZenithAngles"][t_idx].squeeze(-1) * np.pi / 180
    sun_azimuth = (360 - eop.data["sunAzimuthAngles"][t_idx].squeeze(-1) + 90) * np.pi / 180

    grad_x = cv2.Sobel(dem, cv2.CV_32F, 1, 0, ksize=3) / resx / 8
    grad_y = cv2.Sobel(dem, cv2.CV_32F, 0, 1, ksize=3) / resy / 8
    slope = np.arctan((grad_x**2 + grad_y**2) ** 0.5)

    aspect = np.zeros_like(slope)
    aspect[grad_x != 0] = np.arctan2(grad_y[grad_x != 0], -grad_x[grad_x != 0])
    aspect[(grad_x != 0) & (aspect < 0)] += 2 * np.pi
    aspect[(grad_x == 0) & (grad_y > 0)] = np.pi / 2
    aspect[(grad_x == 0) & (grad_y < 0)] = 2 * np.pi - np.pi / 2

    hs = (np.cos(sun_zenith) * np.cos(slope)) + (np.sin(sun_zenith) * np.sin(slope) * np.cos(sun_azimuth - aspect))
    return dilate_mask(hs <= 0, dilation_size)


def get_water_mask(eop: EOPatch, t_idx: int, dilation_size: int, threshold: float = 0.4) -> np.ndarray:
    """Calculates the water mask for the given EOPatch and time index. The water mask can be dilated."""
    green = eop.data["BANDS"][t_idx][..., 2]
    nir = eop.data["BANDS"][t_idx][..., 7]

    water_mask = -1 * np.ones_like(green)
    notna_mask = (green + nir) != 0
    water_mask[notna_mask] = (green[notna_mask] - nir[notna_mask]) / (green[notna_mask] + nir[notna_mask]) > threshold
    return dilate_mask(water_mask, 2 * dilation_size)


def connected_components_filter(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Filters out connected components smaller than the given size."""
    components, ncomponents = label(mask, np.ones((3, 3)))  # 8-connectivity
    sizes = {idx: np.count_nonzero(components == idx) for idx in range(1, ncomponents)}
    cands = np.zeros_like(components).astype(bool)

    for k, v in sizes.items():
        if v > min_size:
            cands[components == k] = True

    return cands


def get_shadow_candidates(eop: EOPatch, t_idx: int, lum_perc_thr: float, min_size: int) -> np.ndarray:
    """Calculates the shadow candidates mask for the given EOPatch and time index."""
    bands = eop.data["BANDS"][t_idx]
    valid = eop.mask["dataMask"][t_idx].squeeze(-1)
    hsm = eop.mask["HSM"][t_idx].squeeze(-1)
    water_mask = eop.mask["WATER_MASK"][t_idx].squeeze(-1)
    clm = eop.mask["CLM"][t_idx].squeeze(-1)

    lum = np.mean(np.square(bands), axis=-1) ** 0.5
    lum_range = np.percentile(lum[valid & ~clm], [1, 99])  # filter to 1-99 percentile
    lum_perc_thr = lum_range[0] + (lum_range[1] - lum_range[0]) * lum_perc_thr  # threshold in lum range
    lum_mask = lum < lum_perc_thr

    filter_mask = ~clm & ~water_mask & ~hsm & valid  # filter out clouds, water, hillshade and invalid pixels
    raw_mask = lum_mask & filter_mask
    raw_mask = connected_components_filter(raw_mask, min_size)  # filter out small connected components
    return raw_mask > 0


def project_cloud_mask(eop: EOPatch, t_idx: int, elevation: float) -> np.ndarray:
    """Projects the cloud mask to the ground level assuming some cloud height and using the sun and view angles."""

    # calculate the image resolution
    dem_feature = (FeatureType.DATA_TIMELESS, "DEM")
    h, w = eop.get_spatial_dimension(*dem_feature)
    resx, resy = bbox_to_resolution(eop.bbox, height=h, width=w)

    # load the necessary data
    clm = eop[(FeatureType.MASK, "CLM")][t_idx].squeeze(-1) == 1
    hsm = eop[(FeatureType.MASK, "HSM")][t_idx].squeeze(-1)
    valid = eop[(FeatureType.MASK, "dataMask")][t_idx].squeeze(-1)

    # obtain the solar and view angles
    mask = clm & valid & ~hsm
    sZ = eop.data["sunZenithAngles"][t_idx].squeeze(-1)[mask] * np.pi / 180
    sA = eop.data["sunAzimuthAngles"][t_idx].squeeze(-1)[mask] * np.pi / 180
    vZ = eop.data["viewZenithMean"][t_idx].squeeze(-1)[mask] * np.pi / 180
    vA = eop.data["viewAzimuthMean"][t_idx].squeeze(-1)[mask] * np.pi / 180
    sZ, sA, vZ, vA = list(map(np.nanmean, [sZ, sA, vZ, vA]))  # calculate the average

    # get pixel indices of the cloud mask and convert them to CRS coordinates
    rows_c, cols_c = np.where(mask)
    xmin, _ = eop.bbox.lower_left
    _, ymax = eop.bbox.upper_right
    xc = xmin + cols_c * resx
    yc = ymax - rows_c * resy

    # project the cloud mask to the ground level in CRS coordinates
    xs = xc - elevation * (-np.tan(vZ) * np.sin(vA) + np.tan(sZ) * np.sin(sA))
    ys = yc - elevation * (np.tan(vZ) * np.cos(vA) + np.tan(sZ) * np.cos(sA))

    # filter out NaN values
    nan_mask = np.isnan(xs) | np.isnan(ys)
    xc, yc, xs, ys = [array[~nan_mask] for array in [xc, yc, xs, ys]]

    # convert the CRS coordinates back to pixel indices
    cols_s = (np.round(xs - xmin, decimals=-1) / resx).astype(int)
    rows_s = (np.round(ymax - ys, decimals=-1) / resy).astype(int)

    # filter out-of-bounds values
    out_of_bounds = (cols_s < 0) | (cols_s >= w) | (rows_s < 0) | (rows_s >= h)
    cols_s = cols_s[~out_of_bounds]
    rows_s = rows_s[~out_of_bounds]

    # create the cloud shadow mask
    shadow_mask = np.zeros_like(mask)
    shadow_mask[(rows_s, cols_s)] = 1
    shadow_mask[~valid] = 0

    return (
        shadow_mask & ~clm & valid,  # remove clouds and invalid pixels
        (np.round(xs[0] - xc[0], decimals=-1) / resx).astype(int),  # projection offset in x
        (np.round(ys[0] - yc[0], decimals=-1) / resy).astype(int),  # projection offset in y
    )


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates the intersection over union between two binary masks."""
    return np.count_nonzero(a & b) / np.count_nonzero(a | b)


def get_edge_mask(cols_off: int, rows_off: int, shape: tuple[int, int]) -> np.ndarray:
    """Function to get the edge area of the mask after projection"""
    edge_offset_mask = np.zeros(shape, dtype=bool)
    if cols_off > 0:
        edge_offset_mask[:, :cols_off] = True
    else:
        edge_offset_mask[:, cols_off:] = True

    if rows_off > 0:
        edge_offset_mask[-rows_off:] = True
    else:
        edge_offset_mask[:-rows_off] = True

    return edge_offset_mask
