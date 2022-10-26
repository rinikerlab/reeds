## RE-EDS with OpenMM

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

For the parallel implementation, you will additionally need to install:

 - mpi4py
 - pycuda

### executing the scripts
#### serial implementation

In this implementation, the replicas are executed one after the other for the number of steps between exchanges, then the exchange probabilities are calculated and the replicas are (potentially) exchanged.

With slurm, you can submit the serial scripts using

    sbatch --gpus=1 --wrap 'python set_A_water.py'
    sbatch --wrap 'python set_A_vacuum.py'

After execution you can analyze the simulation using

    sbatch --wrap 'python analysis.py

#### parallel implementation

In this implementation, the replicas are executed in parallel. Each replica is assigned its own MPI process, so the number of available MPI cores needs to be equal to the number of replicas. If enough GPUs are available, each replica is assigned its own GPU. Otherwise, the replicas are distributed evenly among the GPUs.

With slurm, you can submit the parallel scripts using

    sbatch --gpus=8 -n 16 --wrap 'mpirun -np 16 python set_A_water_parallel.py'
    sbatch -n 16 --wrap 'mpirun -np 16 python set_A_vacuum_parallel.py'
    
Of course, you can also reduce the number of requested gpus. In the above example with 16 replicas, if you request 8 GPUs, each GPU will be responsible for 2 simulations. If you only request e.g. 2 GPUs, each GPU will be responsible for 8 simulations.

After execution you can analyze the simulation using

    sbatch --wrap 'python analysis_parallel.py 
