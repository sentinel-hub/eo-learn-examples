## Introduction

In **Custom-script** example we show how  to create a Machine Learning (ML) custom-script for water detection.

The example uses [eo-learn](https://eo-learn.readthedocs.io/en/latest/) to process the data and [LightGBM](https://lightgbm.readthedocs.io/en/latest/) to train a ML model for water classification given Seninel-2 band and index values. The resulting custom-script can be used in [the Sentinel Hub EOBrowser](https://www-test.sentinel-hub.com/explore/eobrowser/), in the [multi-temporal instance of Sentinel Playground](https://apps.sentinel-hub.com/sentinel-playground-temporal/?source=S2&lat=40.4&lng=-3.730000000000018&zoom=12&preset=1-NATURAL-COLOR&layers=B04,B03,B02&maxcc=20&gain=1.0&temporal=true&gamma=1.0&time=2015-01-01%7C2019-10-02&atmFilter=&showDates=false) and as evalscript in the [Sentinel Hub process API](https://docs.sentinel-hub.com/api/latest/api/process/).

## Installation

In order to run the example you'll need a Sentinel Hub account.
You can get a trial version [here](https://www.sentinel-hub.com/trial).

Example requires a Python version >= 3.9 and can be set up with:

```
$ pip install -r requirements.txt
```

and run:

```
$ jupyter notebook machine-learning-evalscript.ipynb
```