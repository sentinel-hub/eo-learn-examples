# Global Earth Monitor Machine Learning Framework (GEM ML)
The purpose of the GEM framework is to tackle scale: regarding the amount of data being processed, the size of models being applied and the complexity of the logic being implemented.
Two key ingredients of such pipelines are both flexibility and standardization.
Flexibility is especially needed for research as it usually tries to test different model architectures for the same problem without necessarily rewriting the whole code.
For that purpose, the code has to be in a way standardized such that this flexibility is possible.

Hence, the consortium established a framework fostering modularity or reusability, enable flexibility in a standardized way and to upscale the problems to relevant size.
We explicitly emphasize the word “relevant” in the sense that GEM tries to solve relevant use-cases requiring the ability to deal with a large amount of data in finite time.
In contrast to toy examples, that asks for an appropriate infrastructure.
That infrastructure is presented in the following.

---
## Examples
- Deforestation Detection based on Sentinel-2 (Example_DeforestationDetection)
- NDVI Prediction based on Sentinel-2 (Example_Treecurrent)
- Water Segmentation based on Sentinel-1 (Example_WaterSegmentation)

---
## Installation

In general, we tested the installation using **Python 3.8** and **pip**.


#### Create new virtual environment and activate
- `python -m venv MYENV`
- activate venv 
  - Windows: `MYENV/Scripts/activate.bat`
  - Linux: `source MYENV/bin/activate`

#### Install packages
All necessary packages are listed in the requirements file. It can be installed via
`pip install -r requirements.txt`. 

For the `eo-learn` packages, the installation through the requirements file
might fail. In that case, remove the packages `eo-learn-*` from the requirement file and
install them manually, following the [official installation instructions](https://eo-learn.readthedocs.io/en/latest/install.html).


#### Install `GEM-ML` as a package `pip install -e /PATH/TO/GEM-ML`

#### Install PyTorch separately
- follow [official installation instructions](https://pytorch.org/get-started/locally/) for your system specifications
- example for CPU only version: `pip install torch torchvision torchaudio`



---
## Authors
- Michael Engel (m.engel@tum.de)
- Colin Moldenhauer (colin.moldenhauer@tum.de)
- Niklas Eisl (niklas.eisl@tum.de)
- Joana Reuss (joana.reuss@tum.de)