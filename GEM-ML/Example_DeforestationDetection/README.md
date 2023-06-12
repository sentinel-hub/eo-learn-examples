# DeforestationDetection
In this pipeline, we provide an in-depth example of how the GEM ML framework can be used for segmenting deforested areas using Sentinel-2 imagery as input and the [TMF dataset](https://forobs.jrc.ec.europa.eu/TMF/) as a reference.
The idea is to use a neural network for the analysis. In our case, we show the usage of the Deep-Lab-V3-Plus but thanks to the flexibility of the GEM ML framework, we can easily substitute the model in the future by adjusting only the configuration file.
In the end, we showcase the performance of our trained model for the hypothetic query of a land surveying office for an as less cloudy but as recent as possible analysis of some mining site.

- Area of interest: Amazonia
- split into 5120m times 5120m patches, which means 256x256 pixels at a resolution of 20m
- bands B02, B03, B04, B08, B11 and B12 of Sentinel-2
- maximum cloud coverage of 30% based on s2cloudless mask
- [TMF dataset](https://forobs.jrc.ec.europa.eu/TMF/) of the 2021-12-31 as a reference (downsampled to 20m using nearest neighbor interpolation), more specifically the [N0, W60 tile](https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2021&lat=N0&lon=W60)