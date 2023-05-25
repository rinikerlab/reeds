import sys
import openmm as mm
import openmm.app as app
from openmm import unit as u
from parmed import load_file
from copy import deepcopy
import numpy as np
import pandas as pd
import os
from collections.abc import Iterable  

from scipy.special import logsumexp
import time

from mpi4py import MPI

class EDSSimulationVariables:
  """
  define the simulation variables for an EDS simulation such as temperature, pressure etc.
  """
  def __init__(self, s_value,
                     energy_offsets,
                     distance_restraint_pairs = [],
                     temperature = 298.15 * u.kelvin, 
                     pressure = 1.01325 * u.bar,
                     cutoff = 1.2 * u.nanometers,
                     eps_reaction_field = 78.5,
                     distance_restraint_force_constant = 5000 *u.kilojoule_per_mole/(u.nanometer*u.nanometer),
                     distance_restraints_start_at_1 = True,
                     minimize = False,
                     time_step = 0.002 * u.picoseconds,
                     total_steps = 250000,
                     initial_time = 0):
                     
    self.kb = (u.BOLTZMANN_CONSTANT_kB*u.AVOGADRO_CONSTANT_NA)
    self.temperature = temperature
    self.pressure = pressure
    self.s_value = s_value
    self.energy_offsets = energy_offsets
    self.cutoff = cutoff
    self.eps_reaction_field = eps_reaction_field
    self.distance_restraint_force_constant = distance_restraint_force_constant
    self.distance_restraint_pairs = distance_restraint_pairs
    self.distance_restraints_start_at_1 = distance_restraints_start_at_1
    self.minimize = minimize,
    self.time_step = time_step
    self.total_steps = total_steps
    self.initial_time = initial_time

class REEDSSimulationVariables:
  """
  define the simulation variables for a RE-EDS simulation such as temperature, pressure etc.
  """
  def __init__(self, s_values,
                     energy_offsets,
                     distance_restraint_pairs = [],
                     temperature = 298.15 * u.kelvin, 
                     pressure = 1.01325 * u.bar,
                     cutoff = 1.2 * u.nanometers,
                     eps_reaction_field = 78.5,
                     distance_restraint_force_constant = 5000 *u.kilojoule_per_mole/(u.nanometer*u.nanometer),
                     distance_restraints_start_at_1 = True,
                     minimize = False,
                     num_steps_between_exchanges = 20,
                     time_step = 0.002 * u.picoseconds,
                     total_steps = 250000,
                     initial_time = 0):

    self.s_values = s_values
    self.num_steps_between_exchanges = num_steps_between_exchanges
    if isinstance(energy_offsets[0], Iterable):
      self.energy_offset_matrix = energy_offsets
    else:
      self.energy_offset_matrix = [energy_offsets] * len(self.s_values)
    
    for i in range(len(s_values)):
      if MPI.COMM_WORLD.Get_rank() == i:
        if isinstance(energy_offsets[0], Iterable):
          self.eds_simulation_variables = EDSSimulationVariables(s_values[i], 
                                                                 energy_offsets[i],
                                                                 distance_restraint_pairs, 
                                                                 temperature,
                                                                 pressure,
                                                                 cutoff,
                                                                 eps_reaction_field,
                                                                 distance_restraint_force_constant,
                                                                 distance_restraints_start_at_1,
                                                                 minimize,
                                                                 time_step,
                                                                 total_steps,
                                                                 initial_time)
        else:
          self.eds_simulation_variables = EDSSimulationVariables(s_values[i], 
                                                                 energy_offsets,
                                                                 distance_restraint_pairs, 
                                                                 temperature,
                                                                 pressure,
                                                                 cutoff,
                                                                 eps_reaction_field,
                                                                 distance_restraint_force_constant,
                                                                 distance_restraints_start_at_1,
                                                                 minimize,
                                                                 time_step,
                                                                 total_steps,
                                                                 initial_time)

class EDSInputFiles:                                                                                  
  """
  define the input files for an EDS simulation, i.e. parameters, coordinates and optionally state files
  """
  def __init__(self, parameter_file, coordinate_file, state_file = None):
    self.parameter_file = parameter_file
    self.coordinate_file = coordinate_file
    self.state_file = state_file

class REEDSInputFiles:
  """
  define the input files for a RE-EDS simulation, i.e. parameters, coordinates and optionally state files
  """
  def __init__(self, parameter_file, coordinate_files, state_files = None):
    for i in range(MPI.COMM_WORLD.Get_size()):
      if MPI.COMM_WORLD.Get_rank() == i:
        if isinstance(coordinate_files, list):
          if isinstance(state_files, list):
            self.eds_input_files = EDSInputFiles(parameter_file, coordinate_files[i], state_files[i])
          else:
            self.eds_input_files = EDSInputFiles(parameter_file, coordinate_files[i], state_files)
        elif isinstance(state_files, list):
          self.eds_input_files = EDSInputFiles(parameter_file, coordinate_files, state_files[i])
        else:
          self.eds_input_files = EDSInputFiles(parameter_file, coordinate_files, state_files)

class EDSSimulation(app.Simulation):
  """
  defines an enveloping distribution sampling (EDS) simulation
  """
  def __init__(self, system_name, eds_simulation_variables, eds_input_files, platform = None, properties = None):
    # initialize general variables
    self.system_name = system_name
    self.eds_simulation_variables = eds_simulation_variables
    self.eds_input_files = eds_input_files
    self.s_value = eds_simulation_variables.s_value
    self.energy_offsets = eds_simulation_variables.energy_offsets
    self.temperature = eds_simulation_variables.temperature
    self.beta = (1/(eds_simulation_variables.kb*self.temperature)).value_in_unit(u.mole/(u.joule))*1000 # mol/kJ
    self.initial_time = eds_simulation_variables.initial_time
    self.num_endstates = len(self.energy_offsets)

    self.create_system(platform, properties)
    self.create_distance_restraints()
    self.create_barostat()
    
    # put all unperturbed forces in force group zero
    for f in self.system.getForces():
      f.setForceGroup(0)

    self.create_reaction_field()
    self.initialize_positions_and_velocities()

    # add pdb reporter
    self.reporters.append(app.PDBReporter(f"{self.system_name}.pdb", 1000, enforcePeriodicBox = True))

  def create_system(self, platform = None, properties = None):
    # create system
    parmed_sys = load_file(self.eds_input_files.parameter_file, self.eds_input_files.coordinate_file)
    system = parmed_sys.createSystem(nonbondedMethod = app.CutoffPeriodic, constraints = app.AllBonds)
    sys.stdout.flush()
    mm.LangevinMiddleIntegrator(self.temperature, 1/u.picoseconds, self.eds_simulation_variables.time_step)
    self.integrator = mm.LangevinMiddleIntegrator(self.temperature, 1/u.picoseconds, self.eds_simulation_variables.time_step)

    if properties is not None:
      app.Simulation.__init__(self, parmed_sys.topology, system, self.integrator, platform, platformProperties = properties)
    else:
      app.Simulation.__init__(self, parmed_sys.topology, system, self.integrator)

    self.context.setPositions(parmed_sys.positions)

  def create_distance_restraints(self):
    # create distance restraints
    k = self.eds_simulation_variables.distance_restraint_force_constant
    disres = mm.CustomBondForce("0.5*k*r^2;")
    disres.addGlobalParameter('k', k)
    #disres.setUsesPeriodicBoundaryConditions(True)

    for pairs in self.eds_simulation_variables.distance_restraint_pairs:
      if self.eds_simulation_variables.distance_restraints_start_at_1:
        disres.addBond(pairs[0]-1,pairs[1]-1)
      else:
        disres.addBond(pairs[0],pairs[1])

    self.system.addForce(disres)

  def create_barostat(self):
    # create barostat
    if(not self.eds_simulation_variables.pressure is None):
      baro = mm.MonteCarloBarostat(self.eds_simulation_variables.pressure, self.temperature, 5)
      self.system.addForce(baro)

  def create_reaction_field(self):
    # create custom reaction field

    # remove default nonbonded force
    for i, f in enumerate(self.system.getForces()):
      if isinstance(f, mm.NonbondedForce):
        default_nonbonded_force = deepcopy(f)
        self.system.removeForce(i)
    
    default_nonbonded_force.setReactionFieldDielectric(self.eds_simulation_variables.eps_reaction_field)
    default_nonbonded_force.setCutoffDistance(self.eds_simulation_variables.cutoff)

    # assumption: end-state molecules are listed consecutively at beginning of the topology, all non end-state particles are environment
    environment_particles = []
    for mol in self.context.getMolecules()[self.num_endstates:]:
      environment_particles.extend(mol)

    active_particles_ = [self.context.getMolecules()[i] for i in range(self.num_endstates)]
    active_particles_.append(environment_particles)            
    
    # custom forces

    for i, active_particles in enumerate(active_particles_):
        if(active_particles != environment_particles):
          a, b, c, d = (self.custom_reaction_field(i, default_nonbonded_force, active_particles, environment_particles))
          a.setName(f"lj_rf_endstate_{i+1}")
          b.setName(f"lj_rf_endstate_{i+1}_one_four")
          c.setName(f"rf_endstate_{i+1}_excluded")
          d.setName(f"rf_endstate_{i+1}_self_interaction")
          a.setForceGroup(i+1)
          b.setForceGroup(i+1)
          c.setForceGroup(i+1)
          d.setForceGroup(i+1)
        else:
          a, b, c, d = (self.custom_reaction_field(i, default_nonbonded_force, active_particles, environment_particles))
          a.setName("lj_rf_environment_environment")
          b.setName("lj_rf_environment_environment_one_four")
          c.setName("rf_environment_environment_excluded")
          d.setName("rf_environment_environment_self_interaction")
          a.setForceGroup(0)
          b.setForceGroup(0)
          c.setForceGroup(0)
          d.setForceGroup(0)
        
        self.system.addForce(a)

        if (b.getNumBonds()):
            self.system.addForce(b)
            
        if (c.getNumBonds()): 
            self.system.addForce(c)
            
        if (d.getNumBonds()):
            self.system.addForce(d)
    
    self.context.reinitialize() 

  def initialize_positions_and_velocities(self):
    if(self.eds_input_files.state_file is None):
      parmed_sys = load_file(self.eds_input_files.parameter_file, self.eds_input_files.coordinate_file)
      self.context.setPositions(parmed_sys.positions)
      self.context.setVelocitiesToTemperature(self.temperature)
    else:
      self.loadState(self.eds_input_files.state_file)

  def custom_reaction_field(self, end_state, original_nonbonded_force, active_particles, environment_particles):
    """
    defines a reaction field with a shifting function (see A. Kubincova et al, Phys. Chem. Chem. Phys. 2020, 22)

    Parameters
    ----------
    end_state: int
      index of current end-state
    original_nonbonded_force: mm.NonbondedForce
      original nonbonded force of the system
    active_particles: List
      list of particle indices of current particles (either all particles of current end-state or all environment particles)
    environment_particles: List
      list of environment particles

    Returns
    -------
    force_lj_crf: mm.CustomNonbondedForce
      lennard jones and reaction field force
    force_lj_crf_one_four: mm.CustomBondForce
      lennard jones and reaction field force for third neighbor particles
    force_crf_excluded: mm.CustomBondForce
      reaction field force for excluded neighbor particles
    force_self_term: mm.CustomBondForce
      reaction field force for self term
    """
    # define parameters
    cutoff = original_nonbonded_force.getCutoffDistance()
    scal = 'scaling_' + str(end_state)

    eps_rf = original_nonbonded_force.getReactionFieldDielectric()
    krf = ((eps_rf - 1) / (1 + 2 * eps_rf)) * (1 / cutoff**3)
    ONE_4PI_EPS0 = 138.935456 #* u.kilojoules_per_mole*u.nanometer/(u.elementary_charge_base_unit*u.elementary_charge_base_unit)
    
    mrf = 4
    nrf = 6
    arfm = (3 * cutoff**(-(mrf+1))/(mrf*(nrf - mrf)))* ((2*eps_rf+nrf-1)/(1+2*eps_rf))
    arfn = (3 * cutoff**(-(nrf+1))/(nrf*(mrf - nrf)))* ((2*eps_rf+mrf-1)/(1+2*eps_rf))
    
    crf = ((3 * eps_rf) / (1 + 2 * eps_rf)) * (1 / cutoff) + arfm * cutoff**mrf + arfn * cutoff ** nrf
    original_nonbonded_force.setUseDispersionCorrection(False)
    original_nonbonded_force.setIncludeDirectSpace(False)

    #
    # nonbonded pairlist interactions (lennard jones and coulomb reaction field)
    #
    lj_crf  = scal + "*(4*epsilon*(sigma_over_r12 - sigma_over_r6) + ONE_4PI_EPS0*chargeprod*(1/r + krf*r2 + arfm*r4 + arfn*r6 - crf));"
    lj_crf += "krf = {:f};".format(krf.value_in_unit(u.nanometer**-3))
    lj_crf += "crf = {:f};".format(crf.value_in_unit(u.nanometer**-1))
    lj_crf += "r6 = r2*r4;"
    lj_crf += "r4 = r2*r2;"
    lj_crf += "r2 = r*r;"
    lj_crf += "arfm = {:f};".format(arfm.value_in_unit(u.nanometer**-5))
    lj_crf += "arfn = {:f};".format(arfn.value_in_unit(u.nanometer**-7))
    lj_crf += "sigma_over_r12 = sigma_over_r6 * sigma_over_r6;"
    lj_crf += "sigma_over_r6 = sigma_over_r3 * sigma_over_r3;"
    lj_crf += "sigma_over_r3 = sigma_over_r * sigma_over_r * sigma_over_r;"
    lj_crf += "sigma_over_r = sigma/r;"
    lj_crf += "epsilon = sqrt(epsilon1*epsilon2);"
    lj_crf += "sigma = 0.5*(sigma1+sigma2);"
    lj_crf += "chargeprod = charge1*charge2;"
    lj_crf += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)

    force_lj_crf = mm.CustomNonbondedForce(lj_crf)
    force_lj_crf.addPerParticleParameter('charge')
    force_lj_crf.addPerParticleParameter('sigma')
    force_lj_crf.addPerParticleParameter('epsilon')
    force_lj_crf.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    force_lj_crf.setCutoffDistance(cutoff)
    force_lj_crf.setUseLongRangeCorrection(False)
    force_lj_crf.addGlobalParameter(scal, 1)

    # copy per particle parameters from original nonbonded force
    for index in range(original_nonbonded_force.getNumParticles()):
      charge, sigma, epsilon = original_nonbonded_force.getParticleParameters(index)
      force_lj_crf.addParticle([charge, sigma, epsilon])
        
    # copy exceptions from original nonbonded force
    for index in range(original_nonbonded_force.getNumExceptions()):
      j, k, chargeprod, sigma, epsilon = original_nonbonded_force.getExceptionParameters(index)
      force_lj_crf.addExclusion(j, k)

    # set interaction groups -> end-state with itself and end-state with environment (or only environment with environment)
    force_lj_crf.addInteractionGroup(active_particles, environment_particles)
    if(active_particles != environment_particles):
      force_lj_crf.addInteractionGroup(active_particles, active_particles)

    #
    # third neighbor interactions (lennard jones and coulomb reaction field)
    #
    lj_crf_one_four = scal + "*(4*epsilon*(sigma_over_r12 - sigma_over_r6) + ONE_4PI_EPS0*chargeprod*(1/r) +ONE_4PI_EPS0*chargeprod_*(krf*r2 + arfm*r4 + arfn*r6 - crf));"
    lj_crf_one_four += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)
    lj_crf_one_four += "krf = {:f};".format(krf.value_in_unit(u.nanometer**-3))
    lj_crf_one_four += "crf = {:f};".format(crf.value_in_unit(u.nanometer**-1))
    lj_crf_one_four += "r6 = r2*r4;"
    lj_crf_one_four += "r4 = r2*r2;"
    lj_crf_one_four += "r2 = r*r;"
    lj_crf_one_four += "sigma_over_r12 = sigma_over_r6 * sigma_over_r6;"
    lj_crf_one_four += "sigma_over_r6 = sigma_over_r3 * sigma_over_r3;"
    lj_crf_one_four += "sigma_over_r3 = sigma_over_r * sigma_over_r * sigma_over_r;"
    lj_crf_one_four += "sigma_over_r = sigma/r;"
    lj_crf_one_four += "arfm = {:f};".format(arfm.value_in_unit(u.nanometer**-5))
    lj_crf_one_four += "arfn = {:f};".format(arfn.value_in_unit(u.nanometer**-7))
    force_lj_crf_one_four = mm.CustomBondForce(lj_crf_one_four)
    force_lj_crf_one_four.addPerBondParameter('chargeprod')
    force_lj_crf_one_four.addPerBondParameter('sigma')
    force_lj_crf_one_four.addPerBondParameter('epsilon')
    force_lj_crf_one_four.addPerBondParameter('chargeprod_')
    force_lj_crf_one_four.addGlobalParameter(scal, 1)
    #force_lj_crf_one_four.setUsesPeriodicBoundaryConditions(True)

    # copy third neighbors from original nonbonded force
    for index in range(original_nonbonded_force.getNumExceptions()):
      j, k, chargeprod, sigma, epsilon = original_nonbonded_force.getExceptionParameters(index)
  
      if j in active_particles and k in active_particles and (chargeprod._value != 0 or epsilon._value != 0 ):
        ch1, _, _ = original_nonbonded_force.getParticleParameters(j)
        ch2, _, _ = original_nonbonded_force.getParticleParameters(k)
        force_lj_crf_one_four.addBond(j, k, [chargeprod, sigma, epsilon, ch1*ch2])

    #
    # excluded neighbors reaction field
    #
    crf_excluded = scal + "*(ONE_4PI_EPS0*chargeprod_*(krf*r2 + arfm*r4 + arfn*r6 -crf));"
    crf_excluded += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)
    crf_excluded += "krf = {:f};".format(krf.value_in_unit(u.nanometer**-3))
    crf_excluded += "crf = {:f};".format(crf.value_in_unit(u.nanometer**-1))
    crf_excluded += "r6 = r2*r4;"
    crf_excluded += "r4 = r2*r2;"
    crf_excluded += "r2 = r*r;"
    crf_excluded += "arfm = {:f};".format(arfm.value_in_unit(u.nanometer**-5))
    crf_excluded += "arfn = {:f};".format(arfn.value_in_unit(u.nanometer**-7))
    force_crf_excluded = mm.CustomBondForce(crf_excluded)
    force_crf_excluded.addPerBondParameter('chargeprod_')
    force_crf_excluded.addGlobalParameter(scal, 1)
    #force_crf_excluded.setUsesPeriodicBoundaryConditions(True)

    # copy excluded neighbors from reaction field
    for index in range(original_nonbonded_force.getNumExceptions()):
      j, k, chargeprod, sigma, epsilon = original_nonbonded_force.getExceptionParameters(index)
      if j in active_particles and k in active_particles and (chargeprod._value == 0):
        ch1, _, _ = original_nonbonded_force.getParticleParameters(j)
        ch2, _, _ = original_nonbonded_force.getParticleParameters(k)
        force_crf_excluded.addBond(j, k, [ch1*ch2])

    #
    # self term
    #
    crf_self_term = scal + "*(0.5 * ONE_4PI_EPS0* chargeprod_ * (-crf));"
    crf_self_term += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)
    crf_self_term += "crf = {:f};".format(crf.value_in_unit(u.nanometer**-1))
    force_crf_self_term = mm.CustomBondForce(crf_self_term)
    force_crf_self_term.addPerBondParameter('chargeprod_')
    force_crf_self_term.addGlobalParameter(scal, 1)
    #force_crf_self_term.setUsesPeriodicBoundaryConditions(True)

    # add self interaction of all current end-state/environment particles
    for i in active_particles:
      ch1, _, _ = original_nonbonded_force.getParticleParameters(i)
      force_crf_self_term.addBond(i, i, [ch1*ch1])

    return force_lj_crf, force_lj_crf_one_four, force_crf_excluded, force_crf_self_term

  def step(self, steps):
    for step in range(steps):
      for j in range(self.num_endstates):
          self.context.setParameter('scaling_' + str(j), 1.0)

      scal = self.get_scaling()        

      for j in range(self.num_endstates):
        self.context.setParameter('scaling_' + str(j), scal[j])

      super().step(1)

    for j in range(self.num_endstates):
      self.context.setParameter('scaling_' + str(j), 1.0)

  def logsumexp_(self, s, Vi):
    """
    logsumexp as implemented in GROMOS -> unused
    """
    partA = -self.beta * s * (Vi[0] - self.energy_offsets[0])       
    partB = -self.beta * s * (Vi[1] - self.energy_offsets[1])
    sum_prefactors = max(partA, partB) + np.log(1+np.exp(min(partA, partB) - max(partA, partB)))
    
    prefactors = np.array([0.] * self.num_endstates)
    prefactors[0] = partA
    prefactors[1] = partB
    
    for i in range(2, self.num_endstates):
      part = -self.beta * s * (Vi[i] - self.energy_offsets[i])
      sum_prefactors = max(sum_prefactors, part) + np.log(1 + np.exp(min(sum_prefactors, part) - max(sum_prefactors, part)))
      prefactors[i] = part

    return sum_prefactors

  def get_scaling(self):
    """
    calculates scaling factors for the end-state energies/forces
    """
    self.Vi = self.get_Vi()
    terms = np.exp(-self.beta * self.s_value * (self.Vi - self.energy_offsets))
    scaling_factors = terms / np.sum(terms)
    
    return scaling_factors

  def get_VR(self):
    return - 1/(self.beta * self.s_value) * self.logsumexp_(self.s_value, self.Vi)

  def get_Vi(self):
    self.Vi = [self.context.getState(getEnergy=True, groups=1<<i+1).getPotentialEnergy().value_in_unit(u.kilojoules_per_mole) for i in range(self.num_endstates)]
    return self.Vi

class REEDS:
  """
  performs a RE-EDS simulation
  """  

  def __init__(self, system_name, reeds_simulation_variables, reeds_input_files):

    self.comm = MPI.COMM_WORLD
    self.rank = self.comm.Get_rank()

    # initialize general variables
    self.s_values = reeds_simulation_variables.s_values
    self.num_replicas = len(self.s_values)
    self.replica_positions = [i for i in range(self.num_replicas)]
    self.reeds_simulation_variables = reeds_simulation_variables
    self.energy_offset_matrix = self.reeds_simulation_variables.energy_offset_matrix

    if(self.comm.Get_size() != self.num_replicas):
      raise Exception(f"not as many MPI cores ({self.comm.Get_size()}) as replicas ({self.num_replicas})")  
   
    #create simulation
    n_gpu = self.get_num_gpus()
    print("ngpu", n_gpu)
    if(self.comm.Get_size() > 1):
      eds_system_name = f"{system_name}_{self.rank}"
    else:
      eds_system_name = system_name
    if(n_gpu):
      platform = mm.Platform.getPlatformByName("CUDA")
      os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
      properties = {'DeviceIndex': str(self.rank % n_gpu)}
      self.EDS_simulation = EDSSimulation(eds_system_name, reeds_simulation_variables.eds_simulation_variables, reeds_input_files.eds_input_files, platform, properties)
    else:
      self.EDS_simulation = EDSSimulation(eds_system_name, reeds_simulation_variables.eds_simulation_variables, reeds_input_files.eds_input_files)

    # print some infos
    print("rank", self.rank, "num_endstates", self.EDS_simulation.num_endstates, "num_replicas ", self.num_replicas)
    print(self.rank, self.EDS_simulation.context.getPlatform(),
                     self.EDS_simulation.context.getPlatform().getName(),
                     self.EDS_simulation.context.getPlatform().getNumPlatforms(),
                     self.EDS_simulation.context.getPlatform().getSpeed())   
    for name in self.EDS_simulation.context.getPlatform().getPropertyNames():
      print(self.rank, name, self.EDS_simulation.context.getPlatform().getPropertyValue(self.EDS_simulation.context, name))

    self.system_name = system_name
    self.initialize_output()

    sys.stdout.flush()
  
  def get_num_gpus(self):
    """Returns the number of GPUs available"""
    from pycuda import driver
    try:
      driver.init()
      num_gpus = driver.Device.count()
      return num_gpus 
    except:
      return 0  

  def initialize_output(self):
    # initialize output files, i.e. energy trajectories and repdat files
    if(self.num_replicas > 1):
      self.ene_traj_filenames = [f"ene_traj_{self.system_name}_{i}" for i in range(1, self.num_replicas + 1)]
    else:
      self.ene_traj_filenames = [f"ene_traj_{self.system_name}"]
    self.ene_traj_files = [open(name, "w") for name in self.ene_traj_filenames]
    if self.rank == 0:
      for file in self.ene_traj_files:
          file.write('{0: <15}'.format("t"))
          for i in range(1, self.EDS_simulation.num_endstates +1):
              file.write('{0: <15}'.format(f"V_{i}"))
          file.write('{0: <15}'.format("V_R") + "\n")

      self.repdat = open(f"repdat_{self.system_name}", "w")
      self.repdat.write('{0: <15}'.format("time") + '{0: <15}'.format("partner_i") + '{0: <15}'.format("partner_j")+ '{0: <15}'.format("s_i") + '{0: <15}'.format("s_j")+'{0: <15}'.format("position_i")+'{0: <15}'.format("position_j")+'{0: <15}'.format("probability")+'{0: <15}'.format("exchanged")+"\n")
      
      # repdat in GROMOS format to use with existing analysis script -> remove in the long run
      self.repdat_gromos = open(f"repdat_gromos_{self.system_name}", "w")
      self.repdat_gromos.write("#======================\n#REPLICAEXSYSTEM\n#======================\n#Number of temperatures:\t1\n#Dimension of temperature values:\t1")
      self.repdat_gromos.write("#Number of lambda values:\t" + str(self.num_replicas) + "\n")
      self.repdat_gromos.write("#T\t" + str(self.EDS_simulation.temperature) + "\n")
      self.repdat_gromos.write("#lambda\t")
      for s in self.s_values:
        self.repdat_gromos.write(" " + str(s))
      self.repdat_gromos.write("\n#s (RE-EDS)\t")
      for s in self.s_values:
        self.repdat_gromos.write(" " + str(s))
      self.repdat_gromos.write("\n")
      for i in range(self.EDS_simulation.num_endstates):
        self.repdat_gromos.write("#eir(s), numstate = " + str(i+1) + " (RE - EDS) ")
        for j in range(self.num_replicas):
          self.repdat_gromos.write(" " + str(self.energy_offset_matrix[j][i]))
        self.repdat_gromos.write("\n")
      self.repdat_gromos.write("#\n\n")    
      self.repdat_gromos.write("pos\tID\tcoord_ID\tpartner\tpartner_start\tpartner_coord_ID\trun\tEpoti\tEpotj\tp\ts\t")
      for i in range(self.EDS_simulation.num_endstates):
        self.repdat_gromos.write("Vr" + str(i+1) + "\t")
      self.repdat_gromos.write("\n")

  def calculate_free_energy_differences(self, s_index):
    """
    calculate the free energy differences of all end-state pairs for the s value with index s_index
    """
    if self.rank == 0:
      ene_traj = pd.read_csv(self.ene_traj_filenames[s_index], header = [0], delim_whitespace = True)
      df = [- 1/self.EDS_simulation.beta * np.log(np.mean(np.exp(-self.EDS_simulation.beta * (ene_traj["V_" + str(j+1)] - ene_traj["V_R"])))/np.mean(np.exp(-self.EDS_simulation.beta * (ene_traj["V_" + str(i+1)] - ene_traj["V_R"])))) for i in range(self.EDS_simulation.num_endstates) for j in range(i+1, self.EDS_simulation.num_endstates)]
        
    return df

  def run(self):
    """
    perform a RE-EDS simulation
    """
    if self.reeds_simulation_variables.eds_simulation_variables.minimize:
      for i in range(self.EDS_simulation.num_endstates):
        self.EDS_simulation.context.setParameter('scaling_' + str(i), 1/self.EDS_simulation.num_endstates)
      self.EDS_simulation.minimizeEnergy(maxIterations = 100000)
      
    self.sim_time = self.EDS_simulation.initial_time
    step_size = self.EDS_simulation.integrator.getStepSize()._value
    self.begin = 1
    self.run = 1
    start_time = time.time()

    for total_steps in range(0,self.reeds_simulation_variables.eds_simulation_variables.total_steps, self.reeds_simulation_variables.num_steps_between_exchanges):
      # print time every 1000th step
      if(self.rank == 0 and not (total_steps % 1000)):
        print("time ", "{:.4f}".format(self.sim_time))
        sys.stdout.flush()
        
      # propagate replica at position idx      
      self.EDS_simulation.step(self.reeds_simulation_variables.num_steps_between_exchanges)
      
      # print output to energy trajectories
      self.sim_time += step_size * self.reeds_simulation_variables.num_steps_between_exchanges
      self.Vi = self.EDS_simulation.get_Vi()
      self.V_R = self.EDS_simulation.get_VR()
      self.write_ene_traj()

      # perform replica exchanges
      self.perform_replica_exchanges()

      # store state files every 1000 steps and flush
      # TODO: add output frequency as REEDSSimulationVariables member
      if(not (total_steps % 10000)):
        self.save_state()
   
    print("simulation time: ", time.time() - start_time)
    self.save_state()

  def write_ene_traj(self):
    if self.rank == 0:
      self.Vi_all = [[0] * self.EDS_simulation.num_endstates] * self.num_replicas
      for idx, pos in enumerate(self.replica_positions):
        #sim = self.simulations[pos]
        if pos == 0:
          VR_ = self.V_R
          Vi_ = self.Vi
        else:
          VR_ = self.comm.recv(source = pos)
          Vi_ = self.comm.recv(source = pos)

        self.Vi_all[idx] = Vi_
        self.ene_traj_files[idx].write('{0: <14}'.format("{:.4f}".format(self.sim_time)) + " ")

        for i in range(self.EDS_simulation.num_endstates):
          self.ene_traj_files[idx].write('{0: <14}'.format("{:.10f}".format(self.Vi_all[idx][i])) + " ")

        if(any(np.isnan(Vi_))):
          print(f"Error: there is a nan in the energies of replica {idx} (s = {self.EDS_simulation.s_value}): {Vi_}")
          sys.stdout.flush()
          self.comm.Abort()
      
        self.ene_traj_files[idx].write('{0: <15}'.format("{:.10f}".format(VR_)))
        self.ene_traj_files[idx].write("\n")
    else:
      self.comm.send(self.V_R, dest = 0)
      self.comm.send(self.Vi, dest = 0)

  def perform_replica_exchanges(self):
    # replica 0 calculates exchange probabilities and sends new s values to other replicas
    if self.rank == 0:
      if(self.begin or self.num_replicas == 2):
        self.begin = 0
      else:
        self.begin = 1
        # if first replica doesn't have a partner -> print info to repdat file
        self.repdat_gromos.write("1\t1\t" + str(self.replica_positions[0]+1) + "\t" +  "1\t1\t" + str(self.replica_positions[0]+1) + "\t" + str(self.run) + "\t0\t0\t0\t0")
        for j in range(self.EDS_simulation.num_endstates):
          self.repdat_gromos.write("\t" + str(self.Vi_all[0][j]))
        self.repdat_gromos.write("\n")
        if(self.replica_positions[0] != 0):
          self.comm.send(self.s_values[self.replica_positions[0]], dest = self.replica_positions[0])
          self.comm.send(self.energy_offset_matrix[self.replica_positions[0]], dest = self.replica_positions[0])
            
      i = 0      
      # alternate replica partners (i.e. even = s values at 0-1, 2-3, 4-5, ... and odd = s values at 1-2, 3-4, 5-6, ...)    
      for i in range(self.begin, self.num_replicas-1,2):
        # calculate exchange probability
        p1 = self.replica_positions[i]
        p2 = self.replica_positions[i+1]
        partners = [p1, p2]
        rnd = np.random.uniform(0,1)
        #prob, V_orig, V_exch = self.exchange_probability(partners)

        V_orig_p1 = -1/(self.EDS_simulation.beta * self.s_values[p1]) * self.EDS_simulation.logsumexp_(self.s_values[p1], self.Vi_all[i])
        V_exch_p1 = -1/(self.EDS_simulation.beta * self.s_values[p2]) * self.EDS_simulation.logsumexp_(self.s_values[p2], self.Vi_all[i])

        V_orig_p2 = -1/(self.EDS_simulation.beta * self.s_values[p2]) * self.EDS_simulation.logsumexp_(self.s_values[p2], self.Vi_all[i+1])
        V_exch_p2 = -1/(self.EDS_simulation.beta * self.s_values[p1]) * self.EDS_simulation.logsumexp_(self.s_values[p1], self.Vi_all[i+1])

        delta = V_exch_p1 + V_exch_p2 - (V_orig_p1 + V_orig_p2)
        if delta < 0:
          prob = 1
        else:
          prob = np.exp(- self.EDS_simulation.beta * delta)

        # print info to repdat file
        self.repdat.write('{0: <14}'.format("{:.4f}".format(self.sim_time)) + " ")
        self.repdat.write('{0: <14}'.format(str(i)) + " ")
        self.repdat.write('{0: <14}'.format(str(i+1)) + " ")
        
        self.repdat.write('{0: <14}'.format("{:.4f}".format(self.s_values[p1])) + " ")
        self.repdat.write('{0: <14}'.format("{:.4f}".format(self.s_values[p2])) + " ")
        self.repdat.write('{0: <14}'.format(str(p1)) + " ")
        self.repdat.write('{0: <14}'.format(str(p2)) + " ")
        self.repdat.write('{0: <14}'.format("{:.4f}".format(prob)) + " ")
        
        if(prob > rnd):
          # perform exchange
          self.s_values[p1], self.s_values[p2] = self.s_values[p2], self.s_values[p1]
          self.energy_offset_matrix[p1], self.energy_offset_matrix[p2] = self.energy_offset_matrix[p2], self.energy_offset_matrix[p1]
          self.replica_positions[i] = p2
          self.replica_positions[i+1] = p1

          # print info to repdat file
          self.repdat.write('{0: <14}'.format("1"))
          self.repdat_gromos.write(str(i+1) + "\t" + str(i+1) + "\t" + str(self.replica_positions[i]+1) + "\t" + str(i+2) + "\t" + str(i+2) + "\t" + str(self.replica_positions[i+1]+1)+ "\t" + str(self.run) + "\t" + str(V_orig_p1) + "\t" + str(V_orig_p2) + "\t" + str(prob) + "\t1")
          for j in range(self.EDS_simulation.num_endstates):
            self.repdat_gromos.write("\t" + str(self.Vi_all[i][j]))
          self.repdat_gromos.write("\n")
          self.repdat_gromos.write(str(i+2) + "\t" + str(i+2) + "\t" + str(self.replica_positions[i+1]+1) + "\t" + str(i+1) + "\t" + str(i+1) + "\t" + str(self.replica_positions[i]+1)+ "\t" + str(self.run) + "\t" + str(V_orig_p2) + "\t" + str(V_orig_p1) + "\t" + str(prob) + "\t1")
          for j in range(self.EDS_simulation.num_endstates):
            self.repdat_gromos.write("\t" + str(self.Vi_all[i+1][j]))
          self.repdat_gromos.write("\n")

        else:
          # print info to repdat file
          self.repdat.write('{0: <14}'.format("0"))
          self.repdat_gromos.write(str(i+1) + "\t" + str(i+1) + "\t" + str(self.replica_positions[i]) + "\t" + str(i+2) + "\t" + str(i+2) + "\t" + str(self.replica_positions[i+1])+ "\t" + str(self.run) + "\t" + str(V_orig_p1) + "\t" + str(V_orig_p2) + "\t" + str(prob) + "\t0")
          for j in range(self.EDS_simulation.num_endstates):
            self.repdat_gromos.write("\t" + str(self.Vi_all[i][j]))
          self.repdat_gromos.write("\n")
          self.repdat_gromos.write(str(i+2) + "\t" + str(i+2) + "\t" + str(self.replica_positions[i+1]) + "\t" + str(i+1) + "\t" + str(i+1) + "\t" + str(self.replica_positions[i])+ "\t" + str(self.run) + "\t" + str(V_orig_p2) + "\t" + str(V_orig_p1) + "\t" + str(prob) + "\t0")
          for j in range(self.EDS_simulation.num_endstates):
            self.repdat_gromos.write("\t" + str(self.Vi_all[i+1][j]))
          self.repdat_gromos.write("\n")

        if(p1 != 0):
          self.comm.send(self.s_values[p1], dest = p1)
          self.comm.send(self.energy_offset_matrix[p1], dest = p1)
        else:
          self.EDS_simulation.s_value = self.s_values[p1]
          self.EDS_simulation.energy_offsets = self.energy_offset_matrix[p1]

        if(p2 != 0):
          self.comm.send(self.s_values[p2], dest = p2)
          self.comm.send(self.energy_offset_matrix[p2], dest = p2)
        else:
          self.EDS_simulation.s_value = self.s_values[p2]
          self.EDS_simulation.energy_offsets = self.energy_offset_matrix[p2]

        self.repdat.write("\n")

      # if last replica doesn't have a partner -> print info to repdat file
      if(i+2 < self.num_replicas):
        self.repdat_gromos.write(str(i+3) + "\t" + str(i+3) + "\t" + str(self.replica_positions[i+2]+1) + "\t" + str(i+3) + "\t" + str(i+3) + "\t" + str(self.replica_positions[i+2]+1)+ "\t" + str(self.run) + "\t0\t0\t" + str(0) + "\t0")
        for j in range(self.EDS_simulation.num_endstates):
          self.repdat_gromos.write("\t" + str(self.Vi_all[i+2][j]))
        self.repdat_gromos.write("\n")
        if self.replica_positions[i+2] != 0:
          self.comm.send(self.s_values[self.replica_positions[i+2]], dest = self.replica_positions[i+2])
          self.comm.send(self.energy_offset_matrix[self.replica_positions[i+2]], dest = self.replica_positions[i+2])
          
      self.run += 1

    # replicas != 0 receive their current s values
    else:
      self.EDS_simulation.s_value = self.comm.recv(source = 0)
      self.EDS_simulation.energy_offsets = self.comm.recv(source = 0)

  def save_state(self):
    if self.rank == 0:
      if self.comm.Get_size() == 1:
        self.EDS_simulation.saveState(self.system_name)
      else:
        for idx, pos in enumerate(self.replica_positions):
          if pos == 0:
            self.EDS_simulation.saveState(self.system_name + "_state_s_" + str(idx))
          else:
            self.comm.send(idx, dest = pos)
      for idx in range(self.num_replicas):
        self.ene_traj_files[idx].flush()
      self.repdat.flush()
      self.repdat_gromos.flush()
      sys.stdout.flush()
      
    else:
      idx = self.comm.recv(source = 0)
      self.EDS_simulation.saveState(self.system_name + "_state_s_" + str(idx))
