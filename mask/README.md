## Introduction

In **CloudMaskTask** example we show how to obtain Sentinel-2 cloud masks with eo-learn.

The simplest (and fastest) way is requesting cloud masks (and optionally cloud probabilities) from the Sentinel Hub service. In case provided cloud masks are not sufficient we also show how to use [s2cloudless](https://github.com/sentinel-hub/sentinel2-cloud-detector) to calculate your own.

## Installation

In order to run the example you'll need a Sentinel Hub account.
You can get a trial version [here](https://www.sentinel-hub.com/trial).

Example requires a Python version >= 3.9 and can be set up with:

```
$ pip install -r requirements.txt
```

and run:

```
$ jupyter notebook CloudMaskTask.ipynb
```
.
