REEDS
==============================
[//]: # (Badges)
[![CI](https://github.com/rinikerlab/reeds/actions/workflows/CI.yaml/badge.svg)](https://github.com/rinikerlab/reeds/actions/workflows/CI.yaml)
[![Documentation](https://img.shields.io/badge/Documentation-here-white.svg)](https://rinikerlab.github.io/reeds/)



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

For using this repository, clone it (like the command below) into a directory on your machine and add the path to the repo to your python path.

    git clone --recurse-submodules <repo url>
    PYTHONPATH=${PYTHONPATH}:/path/to/reeds_Repo/reeds

If you want to update the code of the PyGromos submodule, you can do this:
    git submodule init
    git submodule update

Please if your writing code for this repository, first develop it on an own branch.

     git branch MyBranch    #generate your branch
     git checkout MyBranch  #switch to your branch
     git merge master   #for adding new features from master to your branch

Try to write test cases for your implemented features in /scritps/test. (there are already examples)
So it is easier to maintain the code and add additional features.

If you find a bug or any thing else, please raise an Issue.

required packages:
    - numpydoc
    - mdtraj
    - matplotlib
    - numpy
    - pandas
    - scipy
    - rdkit
    

If you find a bug or have an idea for a cool new feature, you are welcom to raise an Issue at the git page. :)
P.s.: I can recommend Pycharm from dstar, for exploring the repository.

### Copyright

Copyright (c) 2020, Benjamin Ries, Salomé Rieder, Candide Champion


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.
