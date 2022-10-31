## RE-EDS pipeline with OpenMM

### General Remarks

The RE-EDS pipeline consists of three main phases: parameter exploration, parameter optimization, and production. All necessary steps can be carried out by the generic scripts located in the subdirectories of the current directory. Note that the step and analysis scripts are currently identical for all subdirectories, the only difference are the input files in the `0_input` directory and the `global_definitions.py`, containing system-specific definitions. Once the OpenMM pipeline for RE-EDS is fully integrated into the "normal" pipeline of the `reeds` module, the duplicated scripts can be removed. Currently, many options, such as the number of time-steps of each pipeline step is hard-coded in the submission scripts and can be adapted there. In the future, the goal is of course to have default values for these variables, but make them modifiable in one place via the global definitions. Note also that for now, the scripts use the implementation without the custom integrator. If we get the custom integrator implementaiton to work with the barostat, the scripts will be updated to use the new implementation.

In the current directory, you find the instructions to run the whole RE-EDS pipeline for two systems: To calculate the relative hydration free-energy differences of a set of six benzene derivatives(see https://doi.org/10.1063/5.0107935), as well as to calculate the relative binding free-energy differences of five end-state ligands to a CHK1 complex (see https://doi.org/10.1007/s10822-021-00436-z). Note that the CHK1 system has so far only fully been studied with the GROMOS implementation of RE-EDS, and there might be some hick-ups in the OpenMM pipeline. For a straightforward introductory example, it is easier to first familiarize yourself with the set of benzene derivatives (set A).

### Preparation

In your conda environment, add the reeds module to your path with

    conda develop /path/to/reeds

You will need to install the following packages for the serial implementation

 - scipy
 - numpy
 - openmm
 - parmed
 - mpmath
 - pandas
 - seaborn
 - matplotlib
 - pycuda
 - mpi4py
 - pycuda

### Executing the Pipeline

The pipeline scripts are intened for a SLURM system. Note that if you set `n_gpus` to `0` in the global definitions (recommended, e.g., for vacuum simulations), you will get an sbatch error (`This GPU job requests 0 GPUs`). You can ignore this error, as the simulation will still be submitted correctly. The scripts should be executed in the following order:

Submitted together for the state optimization and the lower bound search:

    python submit.py a
    python submit.py b

This will create two subdirectories `a_state_optimization/simulation` and `b_lower_bound/simulation`.

After the simulations have finished, we perform the analysis. This will create two subdirectories `a_state_optimization/analysis` and `b_lower_bound/analysis`:

    srun python analysis.py a
    srun python analysis.py b

Next, we run the energy offset estimation. This will also automatically submit the analysis script:

    python submit.py c

This will create a directory `c_eoff` containing a subdirectory for the equilibration, the simulation, and the analysis of the energy offset estimation.

Next, we can run the s-value optimization. This will also automatically submit the analysis script. The number of s-optimization iterations is defined in `global_definitions.py`:

    python submit.py d

This will create a directory `d_sopt` containing a subdirectory for each s-optimization iteration. Each of these subdirectories contains an equilibration, a simulation, and an analysis directory.

Next, we can run the energy offset rebalancing. This will also automatically submit the analysis script. The number of rebalancing iterations is defined in `global_definitions.py` (can also be zero if no rebalancing is required). Note that the s-values are taken from the analysis directory of the s-optimization iteration that corresponds to the `sopt_iterations` variable of the global definitions (e.g., if `sopt_iterations = 2`, the s-values are taken from `d_sopt/s_opt_2/analysis`).

    python submit.py e

This will create a direcotry `e_eoff_rebal` containing a subdirectory for each energy offset rebalancing iteration. Each of these subdirectories contains an equilibration, a simulation, and an analysis directory.

Finally, we can run the production run. This will also automatically submit the analysis script. The number of production iterations is defined in `global_definitions.py`. By default, each iteration corresponds to a simulation time of 500ps, and simulations will be concatenated for the analysis. Note that the s-values are taken from the analysis directory of the s-optimization iteration that corresponds to the `sopt_iterations` variable of the global definitions (e.g., if `sopt_iterations = 2`, the s-values are taken from `d_sopt/s_opt_2/analysis`). If the number of energy offset rebalancing iterations is larger than zero, the energy offsets are taken from the analysis directory of the energy offset rebalancing iteration that corresponds to the `rebal_iterations` of the global definitions. Otherwise, they are taken from the analysis of the energy offset estimation directory.

    python submit.py f

This will create a directory `f_production` containing a subdirectory for each production iteration (will be concatenated for the analysis) and one for the analysis. Additionally to the usual plots, the analysis directory containts a csv file with the free-energy differences of all end-state pairs.
