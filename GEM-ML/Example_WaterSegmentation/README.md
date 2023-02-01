# WaterSegmentation
Here, we will get a feeling of how the GEM ML framework can be used for the segmentation of water bodies using Sentinel-1 imagery as input and the Sentinel-2 based normalized difference water index (NDWI) as a reference.
During the flood event in Pakistan 2022 temorally close observations of Sentinel-1 and Sentinel-2 were available which made this transfer of knowledge from one sensor to the other possible.
The idea is to use a neural network for the analysis.
In our case, we show the usage of the Deep-Lab-V3-Plus but thanks to the flexibility of the GEM ML framework, we can easily substitute the model in the future by adjusting only the configuration file.
In the end, we showcase the performance of our trained model by the flood events of the Niger river in 2022 where an immediate response is necessary whilst no reference data is available as to too cloudy skys.

- Area of interest: Pakistan
- Showcase: Niger flodd 2022
- split into 5120m times 5120m patches, which means 256x256 pixels at a resolution of 20m
- VV and VH backscatter of Sentinel-1
- NDWI of Sentinel-2 as a reference
- cloud mask based on s2cloudless