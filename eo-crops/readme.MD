# Earth observations for crop monitoring. 

Small python module that gathers awesome functionalities from Sentinelhub, eo-learn python package and OrfeoToolBox for basics download and data processing.

## Installation

To install the package, you need first to clone the git repository into a desired folder

```bash
git clone git@github.com:j-desloires/eo-crops.git
```

Then, open Anaconda prompt and create the environment from environment.yml

```
cd eo-crops
conda env create -f environment.yml
conda activate eo-crops
pip install .
```


If you want to run a jupyter notebook, you need to install jupyter on the environment. Then, you will be able to select the kernel.

```
conda install -c anaconda ipykernel -y
python -m ipykernel install --user --name=eo-crops
jupyter notebook
```

## Dependencies

You should have OrfeoToolBox installed (https://www.orfeo-toolbox.org/CookBook/Installation.html) if you would like to do apply pansharpening and multitemporal speckle filtering. 

Also, you should have a Sentinelhub account with credentials. The latter can be sponsored using Network of Resources program https://www.sentinel-hub.com/Network-of-Resources/.
