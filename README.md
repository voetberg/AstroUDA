# AstroUDA
Light weight implementation of https://arxiv.org/abs/2302.02005


## Data 

* https://zenodo.org/records/7473597 : SDDSS, DeCALS  & Galaxy Zoo sourced from SDSS
* https://zenodo.org/records/5514180#.Y6SM7y-B2_w : Simulated LSST Y1 vs Y10 survey data


## Models 

Base DeepAstroUDA comes with one model - ResNet50 as provided by pytorch. 
Additional models can be implemented as a subclass of `astrouda.models.Model`. 

It is trained with a custom loss function 

$$
L = L_{CE} + \lambda(L_{AC} + L_{ES})
$$

## Experiments 

Basic configuration files for each dataset are included in the `configs` directory. 
To replicate the studies done in the paper: 


| Experiment | Command |
| --- | --- |
| LSST Y1 & Y10  Simulation | `astrouda run train --config "" ` |
| Galaxy Zoo 2 SDSS & DECaLS| |
| Galaxy Zoo 2 SDSS Wide & SDSS Deep | |


