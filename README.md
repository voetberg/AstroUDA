# AstroUDA
Light weight implementation of https://arxiv.org/abs/2302.02005

Install with

```
git clone https://github.com/voetberg/AstroUDA.git
pip install -e ./
```

## Data 

* https://zenodo.org/records/7473597 : SDDSS, DeCALS  & Galaxy Zoo sourced from SDSS
* https://zenodo.org/records/5514180#.Y6SM7y-B2_w : Simulated LSST Y1 vs Y10 survey data

Data is automatically downloaded to the directories given by the key "`source_dir`" in the config.

## Models 

The models included are two pre-trained Resnet18 or Resnet50 models, one as a feature extractor and a one as a classifier.  
The selected model is set with the "`feature_extractor`" config setting. 

Additional models can be implemented as a subclass of `astrouda.models.Model`. 

It is trained with a custom loss function 

$$
L = L_{CE} + \lambda(L_{AC} + L_{ES})
$$

The AC (Adaptive Clustering loss) is built off two diffeerent models, a feature embedding model and a classifier model

## Experiments 

Basic configuration files for each dataset are included in the `configs` directory. 
These scripts simply train and test a model trained with the above domain adapted loss, and do not compare against a non-adapted results. 

| Experiment |  Train Command | Inference Command |
| --- | --- | --- |
| LSST Y1 & Y10  Simulation | `python3 train.py configs/lsst.yaml` | `python3 inference.py configs/lsst.yaml` |
| Galaxy Zoo 2 SDSS & DECaLS | `python3 train.py configs/sdss-decals.yaml` | `python3 inference.py configs/sdss-decals.yaml` |
| Galaxy Zoo 2 SDSS Wide & SDSS Deep | `python3 train.py configs/sdss-views.yaml` | `python3 inference.py configs/sdss-views.yaml` |


