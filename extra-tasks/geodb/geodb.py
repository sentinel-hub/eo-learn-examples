"""
Module with tasks that integrate with GeoDB

To use tasks from this module you have to install dependencies defined in `requirements-geodb.txt`.

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see https://github.com/sentinel-hub/eo-learn/blob/master/CREDITS.md.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

from typing import Any

import geopandas as gpd

from eolearn.core import EOPatch, EOTask
from eolearn.core.types import Feature
from sentinelhub import CRS, BBox, SHConfig


class GeoDBVectorImportTask(EOTask):
    """A task for importing vector data from `geoDB <https://eurodatacube.com/marketplace/services/edc_geodb>`__
    into EOPatch
    """

    def __init__(
        self,
        feature: Feature,
        geodb_client: Any,
        geodb_collection: str,
        geodb_db: str,
        reproject: bool = True,
        clip: bool = False,
        config: SHConfig | None = None,
        **kwargs: Any,
    ):
        """
        :param feature: A vector feature into which to import data
        :param geodb_client: an instance of GeoDBClient
        :param geodb_collection: The name of the collection to be queried
        :param geodb_db: The name of the database the collection resides in [current database]
        :param reproject: Should the geometries be transformed to coordinate reference system of the requested bbox?
        :param clip: Should the geometries be clipped to the requested bbox, or should be geometries kept as they are?
        :param config: A configuration object with credentials
        :param kwargs: Additional args that will be passed to `geodb_client.get_collection_by_bbox` call
            (e.g. where="id>-1", operator="and")
        """
        self.feature = self.parse_feature(feature, allowed_feature_types=lambda fty: fty.is_vector())
        self.config = config or SHConfig()
        self.reproject = reproject
        self.clip = clip
        self.geodb_client = geodb_client
        self.geodb_db = geodb_db
        self.geodb_collection = geodb_collection
        self.geodb_kwargs = kwargs
        self._dataset_crs: CRS | None = None

    @property
    def dataset_crs(self) -> CRS:
        """Provides a "crs" of dataset, loads it lazily (i.e. the first time it is needed)

        :return: Dataset's CRS
        """
        if self._dataset_crs is None:
            srid = self.geodb_client.get_collection_srid(collection=self.geodb_collection, database=self.geodb_db)
            self._dataset_crs = CRS(f"epsg:{srid}")

        return self._dataset_crs

    def _load_vector_data(self, bbox: BBox | None) -> Any:
        """Loads vector data from geoDB table"""
        prepared_bbox = bbox.transform_bounds(self.dataset_crs).geometry.bounds if bbox else None

        if "comparison_mode" not in self.geodb_kwargs:
            self.geodb_kwargs["comparison_mode"] = "intersects"

        return self.geodb_client.get_collection_by_bbox(
            collection=self.geodb_collection,
            database=self.geodb_db,
            bbox=prepared_bbox,
            bbox_crs=self.dataset_crs.epsg,
            **self.geodb_kwargs,
        )

    def _reproject_and_clip(self, vectors: gpd.GeoDataFrame, bbox: BBox | None) -> gpd.GeoDataFrame:
        """Method to reproject and clip vectors to the EOPatch crs and bbox"""

        if self.reproject:
            if not bbox:
                raise ValueError("To reproject vector data, eopatch.bbox has to be defined!")

            vectors = vectors.to_crs(bbox.crs.pyproj_crs())

        if self.clip:
            if not bbox:
                raise ValueError("To clip vector data, eopatch.bbox has to be defined!")

            bbox_crs = bbox.crs.pyproj_crs()
            if vectors.crs != bbox_crs:
                raise ValueError("To clip, vectors should be in same CRS as EOPatch bbox!")

            extent = gpd.GeoSeries([bbox.geometry], crs=bbox_crs)
            vectors = gpd.clip(vectors, extent, keep_geom_type=True)

        return vectors

    def execute(self, eopatch: EOPatch | None = None, *, bbox: BBox | None = None) -> EOPatch:
        """
        :param eopatch: An existing EOPatch. If none is provided it will create a new one.
        :param bbox: A bounding box for which to load data. By default, if none is provided, it will take a bounding box
            of given EOPatch. If given EOPatch is not provided it will load the entire dataset.
        :return: An EOPatch with an additional vector feature
        """
        if bbox is None and eopatch is not None:
            bbox = eopatch.bbox

        vectors = self._load_vector_data(bbox)
        minx, miny, maxx, maxy = vectors.total_bounds
        final_bbox = bbox or BBox((minx, miny, maxx, maxy), crs=CRS(vectors.crs))

        eopatch = eopatch or EOPatch(bbox=final_bbox)
        if eopatch.bbox is None:
            eopatch.bbox = final_bbox

        eopatch[self.feature] = self._reproject_and_clip(vectors, bbox)

        return eopatch
