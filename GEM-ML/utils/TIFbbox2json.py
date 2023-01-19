### Michael Engel ### 2022-10-07 ### TIFbbox2json.py ###
import rasterio
import rasterio.features
import rasterio.warp
import json

def TIFbbox2json(inputfile,outputfile,precision=6):
    with rasterio.open(inputfile) as dataset:
        # Read the dataset's valid data mask as a ndarray.
        mask = dataset.dataset_mask()
        # Extract feature shapes and values from the array.
        for geom, val in rasterio.features.shapes(mask,transform=dataset.transform):
            # Transform shapes from the dataset's own coordinate
            # reference system to CRS84 (EPSG:4326).
            geom = rasterio.warp.transform_geom(dataset.crs, 'EPSG:4326', geom, precision=precision)
            # Print GeoJSON shapes to stdout.
            with open(outputfile,"w") as file_: ### TODO: take care of multiple geoms!
                file_.write(json.dumps(geom))
                
if __name__=='__main__':
    raise NotImplementedError(f"Test of {__name__} not implemented yet!")