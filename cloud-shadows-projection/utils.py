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
    height, width, _ = eop.get_spatial_dimension(*dem_feature)
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
