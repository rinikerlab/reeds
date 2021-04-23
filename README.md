REEDS
==============================
[//]: # (Badges)
[![CI](https://github.com/rinikerlab/reeds/actions/workflows/CI.yaml/badge.svg)](https://github.com/rinikerlab/reeds/actions/workflows/CI.yaml)
[![Documentation](https://img.shields.io/badge/Documentation-here-white.svg)](https://rinikerlab.github.io/reeds/)


Replica Exchange - Enveloping Distribution Sampling (RE-EDS) is a method to calculate the relative free energy of multiple states in a system.
It can be applied to calculate relative solvation free energies or relative binding free energies of ligands to a protein. 
One advantage of this method is, that the transition path of one state into another one is not pre-determined thanks to use of Enveloping Distribution Sampling (EDS) by Christ et al. .

The aim of the module is to make the RE-EDS pipeline accesible to everyone! :)

This Project contains:
 * For python 3.6:
    * Reeds parameter optimization and analysis Funcs ->funcLibs
    * reeds simulation pipeline -> Scripts
    * reeds theory scripts (generating the beautiful harmPot plots) ->Scripts

 * gromos Reeds Versions
 * gromos Files for REEDS
 * submodule: PyGromos is already included in the repo


The project is structured into two folders: 
    * The function_libs folder contains all the code you could use in one of your scripts.
    * The scripts folder contains code bits, you could already use with slight adaptations for your own project.

The repository aims to make the RE-EDS pipeline accessible to everyone!

For more on RE-EDS checkout:
 * [Efficient Round-Trip Time Optimization for Replica-Exchange Enveloping Distribution Sampling (RE-EDS); Dominik Sidler, Michael Cristòfol-Clough, and Sereina Riniker (2017)](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.7b00286)
 * [Replica exchange enveloping distribution sampling (RE-EDS): A robust method to estimate multiple free-energy differences from a single simulation;  Dominik Sidler, Arthur Schwaninger, and Sereina Riniker (2016)](https://aip.scitation.org/doi/abs/10.1063/1.4964781)

Further reading on EDS:
 * [An Alternative to Conventional λ-Intermediate States in Alchemical Free Energy Calculations: λ-Enveloping Distribution Sampling; Gerhard König, Nina Glaser, Benjamin Schroeder, Alžbeta Kubincová, Philippe H. Hünenberger, and Sereina Rinikerb (2020)](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c00520)
 * [Multiple free energies from a single simulation: Extending enveloping distribution sampling to nonoverlapping phase-space distributions; Clara D. Christ and Wilfred F. van Gunsteren (2008)](https://aip.scitation.org/doi/10.1063/1.2913050)

## Structure
This Project contains the code for the RE-EDS workflow of the rinikerlab. In the example folder, you can find the system input data for the pipeline, a template script folder, and several jupyter notebooks covering and explaining certain aspects of the pipeline.
The reeds folder contains some template files, a function library, and ready-to-use scripts for the RE-EDS steps.


![](.img/pipeline.png)

Additional:
    The submodule folder contains [PyGromosTools](https://github.com/rinikerlab/PyGromosTools), which is used extensively in the RE-EDS code.
    The subfolder tests are used for the automatic code testing mainly applied to the energy offset estimation and the roundtrip optimizers.

### Code
The code is written in python3 and requires a compiled version of [Gromos](http://gromos.net/).
At the current stage, an LSF queue is required to submit the simulation jobs. 
The required python packages are listed in devtools/conda-envs/full_env.yaml.

## Install
Using this repository, clone it (like the command below) into a directory on your machine and add the path to the repo to your python path.

    git clone --recurse-submodules <repo url>

Make sure you have the required python packages from devtools/conda-envs/full_env.yaml. You can install the provided env with Anacodna like:

    conda env create -f devtools/conda-envs/full_env.yaml

If you want to update the code of the PyGromos submodule, you can do this:

    git submodule init
    git submodule update

Please if your writing code for this repository, first develop it on an own branch.

     git branch mybranch    #generate your branch
     git checkout mybranch  #switch to your branch
     git merge main   #for adding new features from main to your branch


## Copyright

Copyright (c) 2020, Benjamin Ries, Salomé Rieder, Candide Champion


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.
