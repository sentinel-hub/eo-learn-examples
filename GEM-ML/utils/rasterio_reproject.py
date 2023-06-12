### 2022-10-19 ### Michael Engel ### rasterio_reproject.py ###
### adapted from rasterio documentation: https://rasterio.readthedocs.io/en/latest/topics/reproject.html#reprojecting-a-geotiff-dataset ###
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def rasterio_reproject(inputfile, outputfile, crs_target='EPSG:4326', compression=None):
    try:
        with rasterio.open(inputfile) as src:
            transform, width, height = calculate_default_transform(
                src.crs, crs_target, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': crs_target,
                'transform': transform,
                'width': width,
                'height': height
            })
            if compression!=None:
                kwargs.update(compress=compression)
            with rasterio.open(outputfile, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=crs_target,
                        resampling=Resampling.nearest)
                    
    except Exception as e:
        print(e)
        return False
    
    return True