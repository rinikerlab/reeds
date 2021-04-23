
REEDS
=====


.. image:: https://github.com/SchroederB/REEDS_Pipeline/actions/workflows/CI.yaml/badge.svg
   :target: https://github.com/SchroederB/REEDS_Pipeline/actions/workflows/CI.yaml
   :alt: CI


.. image:: https://img.shields.io/badge/Documentation-here-white.svg
   :target: https://schroederb.github.io/REEDS_Pipeline/
   :alt: Documentation


Replica Exchange - Enveloping Distribution Sampling (RE-EDS) is a method to calculate the relative free energy of multiple states in a system.
It can be applied to calculate relative solvation free energies or relative binding free energies of ligands to a protein. 
One advantage of this method is, that the transition path of one state into another one is not pre-determined thanks to use of Enveloping Distribution Sampling (EDS) by Christ et al. .

The enhanced sampling method Replica Exchange was added by Sidler et al. to speed up the sampling and ease the choice of parameters.\ :raw-html-m2r:`<br>`
Additionally multiple modules were described by Sidler to allow an automatization of the pipeline. 
In this repository we now combined these approaches to an automatic scheme for RE-EDS.


.. image:: .img/State_graph.png
   :target: .img/State_graph.png
   :alt: 


The repository aims to make the RE-EDS pipeline accessible to everyone!

For more on RE-EDS checkout:


* `Efficient Round-Trip Time Optimization for Replica-Exchange Enveloping Distribution Sampling (RE-EDS); Dominik Sidler, Michael Cristòfol-Clough, and Sereina Riniker (2017) <https://pubs.acs.org/doi/abs/10.1021/acs.jctc.7b00286>`_
* `Replica exchange enveloping distribution sampling (RE-EDS): A robust method to estimate multiple free-energy differences from a single simulation;  Dominik Sidler, Arthur Schwaninger, and Sereina Riniker (2016) <https://aip.scitation.org/doi/abs/10.1063/1.4964781>`_

This Project contains:


* 
  For python 3.6:


  * Reeds parameter optimization and analysis Funcs ->funcLibs
  * reeds simulation pipeline -> Scripts
  * reeds theory scripts (generating the beautiful harmPot plots) ->Scripts

* 
  gromos Reeds Versions

* gromos Files for REEDS
* submodule: PyGromos is already included in the repo

The project is structured into two folders: 

.. code-block::

   * The function_libs folder contains all the code you could use in one of your scripts.
   * The scripts folder contains code bits, you could already use with slight adaptations for your own project.


For using this repository, clone it (like the command below) into a directory on your machine and add the path to the repo to your python path.

.. code-block::

   git clone --recurse-submodules <repo url>
   PYTHONPATH=${PYTHONPATH}:/path/to/reeds_Repo/reeds


Make sure you have the required python packages from devtools/conda-envs/full_env.yaml. You can install the provided env with Anacodna like:

.. code-block::

   conda env create -f devtools/conda-envs/full_env.yaml


If you want to update the code of the PyGromos submodule, you can do this:
    git submodule init
    git submodule update

Please if your writing code for this repository, first develop it on an own branch.

.. code-block::

    git branch MyBranch    #generate your branch
    git checkout MyBranch  #switch to your branch
    git merge master   #for adding new features from master to your branch


Try to write test cases for your implemented features in /scritps/test. (there are already examples)
So it is easier to maintain the code and add additional features.

If you find a bug or any thing else, please raise an Issue.

    git branch mybranch    #generate your branch
    git checkout mybranch  #switch to your branch
    git merge main   #for adding new features from main to your branch

   - numpydoc
   - mdtraj
   - matplotlib
   - numpy
   - pandas
   - scipy
   - rdkit



If you find a bug or have an idea for a cool new feature, you are welcom to raise an Issue at the git page. :)
P.s.: I can recommend Pycharm from dstar, for exploring the repository.

Copyright
---------

Copyright (c) 2020, Benjamin Ries, Salome Rieder, Candide Champion

Acknowledgements
~~~~~~~~~~~~~~~~

Project based on the 
`Computational Molecular Science Python Cookiecutter <https://github.com/molssi/cookiecutter-cms>`_ version 1.3.
