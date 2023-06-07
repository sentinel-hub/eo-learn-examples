# Air temperature downscaling in urban areas

Responsible: Nico Bader

This repository demonstrates how to extract the structure of a city's temperature field by using satellite data (Step I). A slide set shows how this gets combined with weather data to generate heatmaps (Step 2). That process will be made available via a meteoblue API that returns a heatmap for the selected hour on a selected day and the notebook will be updated accordingly.


# I) Satellite pre-processing

## Getting started

**1. Clone the repository to your local machine.**
```
cd existing_repo
git clone *link to git repository*
cd *direction with repository*
```

**2. Install and activate a new conda environment**

We will call this conda environment *sat_env* (just as an example).
```
conda create --name sat_env --file requirements.txt
conda activate sat_env
```

## Run the example jupyter notebook

To understand how the satellite pre-processing works, you can use the jupyter notebook *satellite_pre-processing.ipynb*.
The satellite processing will be explained step-by-step.

# 2) Access High-resolution API

Will be added soon.