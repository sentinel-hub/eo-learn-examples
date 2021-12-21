# eo-learn-examples

**eo-learn makes extraction of valuable information from satellite imagery easy. This is where we show you how.**

The availability of open Earth observation (EO) data through the Copernicus and Landsat programs represents an unprecedented resource for many EO applications, ranging from ocean and land use and land cover monitoring, disaster control, emergency services and humanitarian relief. Given the large amount of high spatial resolution data at high revisit frequency, techniques able to automatically extract complex patterns in such _spatio-temporal_ data are needed.

**`eo-learn`** _library acts as a bridge between Earth observation/Remote sensing field and Python ecosystem for data science and machine learning._ The library is written in Python and uses NumPy arrays to store and handle remote sensing data. Its aim is to make entry easier for non-experts to the field of remote sensing on one hand and bring the state-of-the-art tools for computer vision, machine learning, and deep learning existing in Python ecosystem to remote sensing experts.

The **`eo-learn-examples`** repository contains example Earth observation workflows that extract valuable information from satellite imagery, giving you hints and ideas how to use the EO data. 

## Sentinel-Hub account

In order to run (some of) the examples you  need a Sentinel Hub account. You can get a trial version [here](https://www.sentinel-hub.com).

Once you have the account set up, login to [Sentinel Hub Configurator](https://apps.sentinel-hub.com/configurator/). By default you will already have the default confoguration with an **instance ID** (alpha-numeric code of length 36). For these examples it is recommended that you create a new configuration (`"Add new configuration"`) and set the configuration to be based on **Python scripts template**. Such configuration will already contain all layers used in these examples. Otherwise you will have to define the layers for your  configuration yourself.

After you have decided which configuration to use, you have two options You can either put configuration's **instance ID** into `sentinelhub` package's configuration file following the [configuration instructions](http://sentinelhub-py.readthedocs.io/en/latest/configure.html) or you can write it down in the example notebooks.

##  Overview

TODO 

## Installation

Generally, examples should run with having latest `eo-learn` installed. In other cases, the example should come with instructions how to set-up environment in order to be able to run it.


## Contributions

We are very curious to see how you use `eo-learn`. If you would like to contribute to `eo-learn-examples`, please check out our [contribution guidelines](./CONTRIBUTING.md).

## Blog posts and papers

 * [Introducing eo-learn](https://medium.com/sentinel-hub/introducing-eo-learn-ab37f2869f5c) (by Devis Peressutti)
 * [Land Cover Classification with eo-learn: Part 1 - Mastering Satellite Image Data in an Open-Source Python Environment](https://medium.com/sentinel-hub/land-cover-classification-with-eo-learn-part-1-2471e8098195) (by Matic Lubej)
 * [Land Cover Classification with eo-learn: Part 2 - Going from Data to Predictions in the Comfort of Your Laptop](https://medium.com/sentinel-hub/land-cover-classification-with-eo-learn-part-2-bd9aa86f8500) (by Matic Lubej)
 * [Land Cover Classification with eo-learn: Part 3 - Pushing Beyond the Point of “Good Enough”](https://medium.com/sentinel-hub/land-cover-classification-with-eo-learn-part-3-c62ed9ecd405) (by Matic Lubej)
 * [Innovations in satellite measurements for development](https://blogs.worldbank.org/opendata/innovations-satellite-measurements-development)
 * [Use eo-learn with AWS SageMaker](https://medium.com/@drewbo19/use-eo-learn-with-aws-sagemaker-9420856aafb5) (by Drew Bollinger)
 * [Spatio-Temporal Deep Learning: An Application to Land Cover Classification](https://www.researchgate.net/publication/333262625_Spatio-Temporal_Deep_Learning_An_Application_to_Land_Cover_Classification) (by Anze Zupanc)
 * [Tree Cover Prediction with Deep Learning](https://medium.com/dataseries/tree-cover-prediction-with-deep-learning-afeb0b663966) (by Daniel Moraite)
 * [NoRSC19 Workshop on eo-learn](https://github.com/sentinel-hub/norsc19-eo-learn-workshop)
 * [Tracking a rapidly changing planet](https://medium.com/@developmentseed/tracking-a-rapidly-changing-planet-bc02efe3545d) (by Development Seed)
 * [Land Cover Monitoring System](https://medium.com/sentinel-hub/land-cover-monitoring-system-84406e3019ae) (by Jovan Visnjic and Matej Aleksandrov)
 * [eo-learn Webinar](https://www.youtube.com/watch?v=Rv-yK7Vbk4o) (by Anze Zupanc)
 * [Cloud Masks at Your Service](https://medium.com/sentinel-hub/cloud-masks-at-your-service-6e5b2cb2ce8a) 
 * [ML examples for Common Agriculture Policy](https://medium.com/sentinel-hub/area-monitoring-concept-effc2c262583) 
   * [High-Level Concept](https://medium.com/sentinel-hub/area-monitoring-concept-effc2c262583)
   * [Data Handling](https://medium.com/sentinel-hub/area-monitoring-data-handling-c255b215364f)
   * [Outlier detection](https://medium.com/sentinel-hub/area-monitoring-observation-outlier-detection-34f86b7cc63)
   * [Similarity Score](https://medium.com/sentinel-hub/area-monitoring-similarity-score-72e5cbfb33b6)
   * [Bare Soil Marker](https://medium.com/sentinel-hub/area-monitoring-bare-soil-marker-608bc95712ae)
   * [Mowing Marker](https://medium.com/sentinel-hub/area-monitoring-mowing-marker-e99cff0c2d08)
   * [Crop Type Marker](https://medium.com/sentinel-hub/area-monitoring-crop-type-marker-1e70f672bf44)
   * [Homogeneity Marker](https://medium.com/sentinel-hub/area-monitoring-homogeneity-marker-742047b834dc)
   * [Parcel Boundary Detection](https://medium.com/sentinel-hub/parcel-boundary-detection-for-cap-2a316a77d2f6)
   * Land Cover Classification (still to come)
   * Minimum Agriculture Activity (still to come)
   * [Combining the Markers into Decisions](https://medium.com/sentinel-hub/area-monitoring-combining-markers-into-decisions-d74f70fe7721)
   * Traffic Light System (still to come)
   * Expert Judgement Application (still to come)
 * [Scale-up your eo-learn workflow using Batch Processing API](https://medium.com/sentinel-hub/scale-up-your-eo-learn-workflow-using-batch-processing-api-d183b70ea237) (by Maxim Lamare) 


## Questions and Issues

Feel free to ask questions about the package and its use cases at [Sentinel Hub forum](https://forum.sentinel-hub.com/) or raise an issue on [GitHub](https://github.com/sentinel-hub/eo-learn/issues).

You are welcome to send your feedback to the package authors, EO Research team, through any of [Sentinel Hub communication channel](https://sentinel-hub.com/develop/communication-channels).


## License

See [LICENSE](https://github.com/sentinel-hub/eo-learn/blob/master/LICENSE).

## Acknowledgements

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreements No. 776115 and No. 101004112.
