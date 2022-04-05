# Forest-Map: Large-scale forest type mapping 

This repository showcases how machine learning and multi-temporal Sentinel-2 satellite imagery can perform a large-scale forest mapping. 

## Background

This demonstrator aims to show how state-of-the-art machine learning models can be applied to multi-temporal satellite imagery with [eo-learn](https://github.com/sentinel-hub/eo-learn). The demonstrator uses multi-spectral Sentinel-2 images as a data source. The core idea is to leverage the temporal dimension, besides the spectral information, to better discriminate between different forest types. We rely on a Convolutional Long-Short-Term-Memory (ConvLSTM) as a machine learning model.

## Overview

The code in the `forest-map.ipynb` notebook allows executing forest segmentation at a large scale using multi-temporal satellite imagery. In this example, Sentinel-2 imagery is used and segmented into coniferous and broadleaved forest types.

The `forest-map.ipynb` notebook demonstrates all steps required to perform a large-scale forest type mapping with machine learning. The notebook consists of three parts. First, it will focus mainly on acquiring satellite data from SentinelHub. It then covers the inference part with the eo-learn library and a pre-trained model from [Vision Impulse GmbH](https://www.vision-impulse.com/). Finally, the notebook demonstrates how to perform inference with a model over a selected AOI using eo-learn.

<p align="center">
<img height="350" src="/Users/benni/VI-Projects/eo-learn-examples/forest-map/figs/example_aoi_s2.png" width="350"/>
<img height="350" src="/Users/benni/VI-Projects/eo-learn-examples/forest-map/figs/example_aoi_prediction.png" width="350"/>
</p>


## Pre-trained model

A pre-trained model is open-sourced and available on the AWS S3 bucket [queryplanet.sentinel-hub.com](http://queryplanet.sentinel-hub.com/index.html). The results shown in the notebook are exemplary and are obtained with models fine-tuned on a subset of the dataset. The models are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.



## Requirements

Before running the notebook, make sure to install the requirements with `pip install -r requirements.txt`.

You need a Sentinel Hub account to run the example notebook, which you can get [here](https://services.sentinel-hub.com/oauth/subscription). Free trial accounts are available.

## Acknowledgements

Project funded by [ESA](https://www.esa.int/About_Us/ESRIN) [Philab](https://philab.phi.esa.int/) through the QueryPlanet 4000124792/18/I-BG grant.

## Questions

If you have any comments or questions, please get in touch at _benjamin.bischke[at]vision-impulse.com_.
