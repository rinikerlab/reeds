"""
    The workers are script entities, that get scheduled into the job-queue by the scheduler modules.
    Therefore a worker must provide a functionality that is running isolated from any dependencies, that are not passed by parameter passing.

    This allows to schedule - job chains for RE-EDS simulations with different modules of workers and branches.
"""