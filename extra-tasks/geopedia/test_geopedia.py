"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see https://github.com/sentinel-hub/eo-learn/blob/master/CREDITS.md.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
from geopedia import GeopediaVectorImportTask

from eolearn.core import FeatureType
from sentinelhub import CRS, BBox


@pytest.mark.parametrize(
    argnames="reproject, clip, n_features, bbox, crs",
    ids=["simple", "bbox", "bbox_full", "bbox_smaller"],
    argvalues=[
        (False, False, 193, None, None),
        (False, False, 193, BBox([857000, 6521500, 861000, 6525500], CRS("epsg:2154")), None),
        (True, True, 193, BBox([657089, 5071037, 661093, 5075039], CRS.UTM_31N), CRS.UTM_31N),
        (True, True, 125, BBox([657690, 5071637, 660493, 5074440], CRS.UTM_31N), CRS.UTM_31N),
    ],
)
def test_import_from_geopedia(reproject, clip, n_features, bbox, crs):
    feature = FeatureType.VECTOR_TIMELESS, "lpis_iacs"
    import_task = GeopediaVectorImportTask(feature=feature, geopedia_table=3447, reproject=reproject, clip=clip)
    eop = import_task.execute(bbox=bbox)
    assert len(eop[feature]) == n_features, "Wrong number of features!"
    to_crs = crs or import_task.dataset_crs
    assert eop[feature].crs.to_epsg() == to_crs.epsg
