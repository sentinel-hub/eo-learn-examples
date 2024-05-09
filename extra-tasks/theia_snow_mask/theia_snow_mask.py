"""
Module for snow masking

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see https://github.com/sentinel-hub/eo-learn/blob/master/CREDITS.md.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import itertools
import logging
from typing import Callable

import cv2
import numpy as np

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.types import Feature

LOGGER = logging.getLogger(__name__)


class TheiaSnowMaskTask(EOTask):
    """Task to add a snow mask to an EOPatch. The input data is either Sentinel-2 L1C or L2A level

    Original implementation and documentation available at https://gitlab.orfeo-toolbox.org/remote_modules/let-it-snow

    ATBD https://gitlab.orfeo-toolbox.org/remote_modules/let-it-snow/blob/master/doc/atbd/ATBD_CES-Neige.pdf

    This task computes a snow mask for the input EOPatch. The `data_feature` to be used as input to the
    classifier is a mandatory argument. If all required features exist already, the classifier is run.
    `linear` interpolation is used for resampling of the `data_feature` and cloud probability map, while `nearest`
    interpolation is used to upsample the binary cloud mask.
    """

    B10_THR = 0.015
    DEM_FACTOR = 0.00001

    def __init__(
        self,
        data_feature: Feature,
        band_indices: list[int],
        cloud_mask_feature: Feature,
        dem_feature: Feature,
        dem_params: tuple[float, float] = (100, 0.1),
        red_params: tuple[float, float, float, float, float] = (12, 0.3, 0.1, 0.2, 0.040),
        ndsi_params: tuple[float, float, float] = (0.4, 0.15, 0.001),
        b10_index: int | None = None,
        dilation_size: int = 0,
        undefined_value: int = 0,
        mask_name: str = "SNOW_MASK",
    ):
        """
        :param data_feature: EOPatch feature represented by a tuple in the form of `(FeatureType, 'feature_name')`
            containing the bands B3, B4, and B11

            Example: `(FeatureType.DATA, 'ALL-BANDS')`
        :param band_indices: A list containing the indices at which the required bands can be found in the bands
            feature. If all L1C band values are provided, `band_indices=[2, 3, 11]`. If all L2A band values are
            provided, then `band_indices=[2, 3, 10]`
        :param cloud_mask_feature: EOPatch CLM feature represented by a tuple in the form of
            `(FeatureType, 'feature_name')` containing the cloud mask
        :param dem_feature: EOPatch DEM feature represented by a tuple in the form of `(FeatureType, 'feature_name')`
            containing the digital elevation model
        :param b10_index: Array index where the B10 band is stored in the bands feature. This is used to refine the
            initial cloud mask
        :param dem_params: Tuple with parameters pertaining DEM processing. The first value specifies the bin size
            used to group DEM values, while the second value specifies the minimum snow fraction in an elevation band
            to define z_s. With reference to the ATBD, the tuple is (d_z, f_t)
        :param red_params: Tuple specifying parameters to process the B04 red band. The first parameter defines the
            scaling factor for down-sampling the red band, the second parameter is the maximum value of the
            down-sampled red band for a dark cloud pixel, the third parameter is the minimum value
            to return a non-snow pixel to the cloud mask, the fourth is the minimum reflectance value to pass the 1st
            snow test, and the fifth is the minimum reflectance value to pass the 2nd snow test. With reference to the
            ATBD, the tuple is (r_f, r_d, r_b, r_1, r_2)
        :param ndsi_params: Tuple specifying parameters for the NDSI. First parameter is the minimum value to pass the
            1st snow test, the second parameter is the minimum value to pass the 2nd snow test, and the third parameter
            is the minimum snow fraction in the image to activate the pass 2 snow test. With reference to the
            ATBD, the tuple is (n_1, n_2, f_s)
        """
        self.bands_feature = self.parse_feature(data_feature, allowed_feature_types={FeatureType.DATA})
        self.band_indices = band_indices
        self.dem_feature = self.parse_feature(dem_feature)
        self.clm_feature = self.parse_feature(cloud_mask_feature)
        self.dem_params = dem_params
        self.red_params = red_params
        self.ndsi_params = ndsi_params
        self.b10_index = b10_index
        self.disk_size = 2 * dilation_size + 1
        self.undefined_value = undefined_value
        self.mask_feature = (FeatureType.MASK, mask_name)

    def _resample_red(self, input_array: np.ndarray) -> np.ndarray:
        """Method to resample the values of the red band

        The input array is first down-scaled using bicubic interpolation and up-scaled back using nearest neighbour
        interpolation
        """
        _, height, width = input_array.shape
        size = (height // self.red_params[0], width // self.red_params[0])
        downscaled = resize_images(input_array[..., np.newaxis], new_size=size)
        return resize_images(downscaled, new_size=(height, width)).squeeze()

    def _adjust_cloud_mask(
        self, bands: np.ndarray, cloud_mask: np.ndarray, dem: np.ndarray, b10: np.ndarray | None
    ) -> np.ndarray:
        """Adjust existing cloud mask using cirrus band if L1C data and resampled red band

        Add to the existing cloud mask pixels found thresholding down-sampled red band and cirrus band/DEM
        """
        if b10 is not None:
            clm_b10 = b10 > self.B10_THR + self.DEM_FACTOR * dem
        else:
            clm_b10 = np.full_like(cloud_mask, True)

        criterion = (cloud_mask == 1) & (self._resample_red(bands[..., 1]) > self.red_params[1])
        return criterion | clm_b10

    def _apply_first_pass(
        self, bands: np.ndarray, ndsi: np.ndarray, clm: np.ndarray, dem: np.ndarray, clm_temp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Apply first pass of snow detection"""
        snow_mask_pass1 = ~clm_temp & (ndsi > self.ndsi_params[0]) & (bands[..., 1] > self.red_params[3])

        clm_pass1 = clm_temp | ((bands[..., 1] > self.red_params[2]) & ~snow_mask_pass1 & clm.astype(bool))

        min_dem, max_dem = np.min(dem), np.max(dem)
        dem_edges = np.linspace(min_dem, max_dem, int(np.ceil((max_dem - min_dem) / self.dem_params[0])))
        nbins = len(dem_edges) - 1

        dem_hist_clear_pixels, snow_frac = None, None
        if nbins > 0:
            snow_frac = np.zeros(shape=(bands.shape[0], nbins))
            dem_hist_clear_pixels = np.array([np.histogram(dem[~mask], bins=dem_edges)[0] for mask in clm_pass1])

            for date, nbin in itertools.product(range(bands.shape[0]), range(nbins)):
                if dem_hist_clear_pixels[date, nbin] > 0:
                    dem_mask = (dem_edges[nbin] <= dem) & (dem < dem_edges[nbin + 1])
                    in_dem_range_clear = np.where(dem_mask & ~clm_pass1[date])
                    snow_frac[date, nbin] = (
                        np.sum(snow_mask_pass1[date][in_dem_range_clear]) / dem_hist_clear_pixels[date, nbin]
                    )
        return snow_mask_pass1, snow_frac, dem_edges

    def _apply_second_pass(
        self,
        bands: np.ndarray,
        ndsi: np.ndarray,
        dem: np.ndarray,
        clm_temp: np.ndarray,
        snow_mask_pass1: np.ndarray,
        snow_frac: np.ndarray | None,
        dem_edges: np.ndarray,
    ) -> np.ndarray:
        """Second pass of snow detection"""
        _, height, width, _ = bands.shape
        total_snow_frac = np.sum(snow_mask_pass1, axis=(1, 2)) / (height * width)
        snow_mask_pass2 = np.full_like(snow_mask_pass1, False)
        for date in range(bands.shape[0]):
            if (
                (total_snow_frac[date] > self.ndsi_params[2])
                and snow_frac is not None
                and np.any(snow_frac[date] > self.dem_params[1])
            ):
                z_s = dem_edges[np.max(np.argmax(snow_frac[date] > self.dem_params[1]) - 2, 0)]

                snow_mask_pass2[date, :, :] = (
                    (dem > z_s)
                    & ~clm_temp[date]
                    & (ndsi[date] > self.ndsi_params[1])
                    & (bands[date, ..., 1] > self.red_params[-1])
                )

        return snow_mask_pass2

    def _apply_dilation(self, snow_masks: np.ndarray) -> np.ndarray:
        """Apply binary dilation for each mask in the series"""
        if self.disk_size > 0:
            disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.disk_size, self.disk_size))
            snow_masks = np.array([cv2.dilate(mask.astype(np.uint8), disk) for mask in snow_masks])
        return snow_masks.astype(bool)

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Run multi-pass snow detection"""
        bands = eopatch[self.bands_feature][..., self.band_indices]
        b10 = eopatch[self.bands_feature][..., self.b10_index] if self.b10_index is not None else None
        dem = eopatch[self.dem_feature][..., 0]
        clm = eopatch[self.clm_feature][..., 0]

        with np.errstate(divide="ignore", invalid="ignore"):
            # (B03 - B11) / (B03 + B11)
            ndsi = (bands[..., 0] - bands[..., 2]) / (bands[..., 0] + bands[..., 2])

        ndsi_invalid = ~np.isfinite(ndsi)
        ndsi[ndsi_invalid] = self.undefined_value

        clm_temp = self._adjust_cloud_mask(bands, clm, dem, b10)

        snow_mask_pass1, snow_frac, dem_edges = self._apply_first_pass(bands, ndsi, clm, dem, clm_temp)

        snow_mask_pass2 = self._apply_second_pass(bands, ndsi, dem, clm_temp, snow_mask_pass1, snow_frac, dem_edges)

        snow_mask = self._apply_dilation(snow_mask_pass1 | snow_mask_pass2)

        eopatch[self.mask_feature] = snow_mask[..., np.newaxis].astype(bool)

        return eopatch


def map_over_axis(data: np.ndarray, func: Callable[[np.ndarray], np.ndarray], axis: int = 0) -> np.ndarray:
    """Map function func over each slice along axis.
    If func changes the number of dimensions, mapping axis is moved to the front.

    Returns a new array with the combined results of mapping.

    :param data: input array
    :param func: Mapping function that is applied on each slice. Outputs must have the same shape for every slice.
    :param axis: Axis over which to map the function.

    :example:

    >>> data = np.ones((5,10,10))
    >>> func = lambda x: np.zeros((7,20))
    >>> res = map_over_axis(data,func,axis=0)
    >>> res.shape
    (5, 7, 20)
    """
    # Move axis to front
    data = np.moveaxis(data, axis, 0)
    mapped_data = np.stack([func(data_slice) for data_slice in data])

    # Move axis back if number of dimensions stays the same
    if data.ndim == mapped_data.ndim:
        mapped_data = np.moveaxis(mapped_data, 0, axis)

    return mapped_data


def resize_images(  # type: ignore[no-untyped-def]
    data,
    new_size=None,
    scale_factors=None,
    anti_alias=True,
    interpolation="linear",
):
    """DEPRECATED, please use `eolearn.features.utils.spatially_resize_image` instead.

    Resizes the image(s) according to given size or scale factors.

    To specify the new scale use one of `new_size` or `scale_factors` parameters.

    :param data: input image array
    :param new_size: New size of the data (height, width)
    :param scale_factors: Factors (fy,fx) by which to resize the image
    :param anti_alias: Use anti aliasing smoothing operation when downsampling. Default is True.
    :param interpolation: Interpolation method used for resampling.
                          One of 'nearest', 'linear', 'cubic'. Default is 'linear'.
    """

    inter_methods = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}

    # Number of dimensions of input data
    ndims = data.ndim

    height_width_axis = {2: (0, 1), 3: (0, 1), 4: (1, 2)}

    # Old height and width
    old_size = tuple(data.shape[axis] for axis in height_width_axis[ndims])

    if new_size is not None and scale_factors is None:
        scale_factors = tuple(new / old for old, new in zip(old_size, new_size))
    elif scale_factors is not None and new_size is None:
        new_size = tuple(int(size * factor) for size, factor in zip(old_size, scale_factors))
    else:
        raise ValueError("Exactly one of the arguments new_size, scale_factors must be given.")

    if interpolation not in inter_methods:
        raise ValueError(f"Invalid interpolation method: {interpolation}")

    interpolation_method = inter_methods[interpolation]
    downscaling = scale_factors[0] < 1 or scale_factors[1] < 1

    def _resize2d(image: np.ndarray) -> np.ndarray:
        if downscaling and anti_alias:
            # Sigma computation based on skimage resize implementation
            sigmas = tuple(((1 / s) - 1) / 2 for s in scale_factors)

            # Limit sigma values above 0
            sigma_y, sigma_x = tuple(max(1e-8, sigma) for sigma in sigmas)
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_REFLECT)

        height, width = new_size
        return cv2.resize(image, (width, height), interpolation=interpolation_method)

    # pylint: disable-next=unnecessary-lambda-assignment
    _resize3d = lambda x: map_over_axis(x, _resize2d, axis=2)  # noqa: E731
    # pylint: disable-next=unnecessary-lambda-assignment
    _resize4d = lambda x: map_over_axis(x, _resize3d, axis=0)  # noqa: E731

    # Choose a resize method based on number of dimensions
    resize_methods = {2: _resize2d, 3: _resize3d, 4: _resize4d}

    resize_method = resize_methods[ndims]

    return resize_method(data)
