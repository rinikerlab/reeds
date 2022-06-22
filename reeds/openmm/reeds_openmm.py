import sys
import openmm as mm
import openmm.app as app
from openmm import unit as u
from parmed import load_file
from copy import deepcopy
import numpy as np
import pandas as pd

#from mpmath import mp
#mp.dps = 5

from scipy.special import logsumexp

class ReedsSimulationVariables:
  """
  define the simulation variables for a RE-EDS simulation such as temperature, pressure etc.
  """
  def __init__(self, s_values,
                     energy_offsets,
                     distance_restraint_pairs,
                     temperature = 298.15 * u.kelvin, 
                     pressure = 1.01325 * u.bar,
                     cutoff = 1.2 * u.nanometers,
                     eps_reaction_field = 78.5,
                     distance_restraint_force_constant = 5000 *u.kilojoule_per_mole/(u.nanometer*u.nanometer),
                     distance_restraints_start_at_1 = True,
                     num_steps_between_exchanges = 20, 
                     total_steps = 250000,
                     initial_time = 0):
                     
    self.kb = (u.BOLTZMANN_CONSTANT_kB*u.AVOGADRO_CONSTANT_NA)
    self.temperature = temperature
    self.pressure = pressure
    self.s_values = s_values
    self.energy_offsets = energy_offsets
    self.cutoff = cutoff
    self.eps_reaction_field = eps_reaction_field
    self.distance_restraint_force_constant = distance_restraint_force_constant
    self.distance_restraint_pairs = distance_restraint_pairs
    self.distance_restraints_start_at_1 = distance_restraints_start_at_1
    self.num_steps_between_exchanges = num_steps_between_exchanges
    self.total_steps = total_steps
    self.initial_time = initial_time
    
class ReedsInputFiles:
  """
  define the input files for a RE-EDS simulation, i.e. parameters, coordinates and optionally state files
  """
  def __init__(self, parameter_file, coordinate_files, state_files = None):
    self.parameter_file = parameter_file
    self.coordinate_files = coordinate_files
    self.state_files = state_files 

class Reeds:
  """
  performs a RE-EDS simulation
  """

  def __init__(self, system_name, reeds_simulation_variables, reeds_input_files):
    # initialize general variables    
    self.system_name = system_name
    self.reeds_simulation_variables = reeds_simulation_variables
    self.reeds_input_files = reeds_input_files
  
    self.s_values = reeds_simulation_variables.s_values
    self.energy_offsets = reeds_simulation_variables.energy_offsets
    self.temperature = reeds_simulation_variables.temperature
    self.beta = (1/(reeds_simulation_variables.kb*self.temperature)).value_in_unit(u.mole/(u.joule))*1000 # mol/kJ
    self.initial_time = reeds_simulation_variables.initial_time
  
    self.num_replicas = len(self.s_values)
    self.num_endstates = len(self.energy_offsets)
    print("num_replicas ", self.num_replicas)    
    print("num_endstates ", self.num_endstates)
    self.replica_positions = [i for i in range(self.num_replicas)]

    # create system
    if isinstance(reeds_input_files.coordinate_files, list):
      parmedSys = (load_file(reeds_input_files.parameter_file, reeds_input_files.coordinate_files[0]))
      system = (parmedSys.createSystem(nonbondedMethod=app.CutoffPeriodic, constraints = app.AllBonds))
    else:
      parmedSys = (load_file(reeds_input_files.parameter_file, reeds_input_files.coordinate_files))
      system = (parmedSys.createSystem(nonbondedMethod=app.CutoffPeriodic, constraints = app.AllBonds))
        
    #create simulation
    self.integrators = (mm.LangevinMiddleIntegrator(self.temperature, 1/u.picoseconds, 0.002*u.picoseconds))
    self.simulations = (app.Simulation(parmedSys.topology, system, self.integrators))
    self.simulations.context.setPositions(parmedSys.positions)

    self.states = []

    for i in range(self.num_replicas):
      self.states.append(self.simulations.context.getState(getPositions=True, getVelocities=True))

    # create distance restraints
    k = reeds_simulation_variables.distance_restraint_force_constant
    disres = mm.CustomBondForce("0.5*k*r^2;")
    disres.addGlobalParameter('k', k)
    disres.setUsesPeriodicBoundaryConditions(True)

    for pairs in reeds_simulation_variables.distance_restraint_pairs:
      if reeds_simulation_variables.distance_restraints_start_at_1:
        disres.addBond(pairs[0]-1,pairs[1]-1)
      else:
        disres.addBond(pairs[0],pairs[1])

    self.simulations.system.addForce(disres)
    
    # create barostat
    if(not reeds_simulation_variables.pressure is None):
      baro = mm.MonteCarloBarostat(reeds_simulation_variables.pressure, self.temperature, min(np.ceil(self.reeds_simulation_variables.num_steps_between_exchanges/4), 25))
      self.simulations.system.addForce(baro)
  
    # put all forces in force group zero
    for f in self.simulations.system.getForces():
      f.setForceGroup(0)

    # create custom reaction field

    # remove default nonbonded force
    for i, f in enumerate(self.simulations.system.getForces()):
      if isinstance(f, mm.NonbondedForce):
        default_nonbonded_force = deepcopy(f)
        self.simulations.system.removeForce(i)
    default_nonbonded_force.setReactionFieldDielectric(reeds_simulation_variables.eps_reaction_field)
    default_nonbonded_force.setCutoffDistance(reeds_simulation_variables.cutoff)

    # assumption: end-state molecules are listed consecutively at beginning of the topology, all non end-state particles are unperturbed
    unperturbed_particles = []
    for mol in self.simulations.context.getMolecules()[self.num_endstates:]:
        unperturbed_particles.extend(mol)

    active_particles_ = [self.simulations.context.getMolecules()[i] for i in range(self.num_endstates)]
    active_particles_.append(unperturbed_particles)

    for i, active_particles in enumerate(active_particles_):

        a, b, c, d = (self.custom_reaction_field(i, default_nonbonded_force, active_particles, unperturbed_particles))

        if(active_particles != unperturbed_particles):
          a.setName("lj_rf_endstate_" + str(i+1))
          b.setName("lj_rf_endstate_" + str(i+1) + "_one_four")
          c.setName("rf_endstate_" + str(i+1) + "_excluded")
          d.setName("rf_endstate_" + str(i+1) + "_self_interaction")
        else:
          a.setName("lj_rf_unperturbed_unperturbed")
          b.setName("lj_rf_unperturbed_unperturbed_one_four")
          c.setName("rf_endstate_unperturbed_unperturbed_excluded")
          d.setName("rf_endstate_unperturbed_unperturbed_self_interaction")

        a.setForceGroup(i+1)
        self.simulations.system.addForce(a)

        if (b.getNumBonds()):
            b.setForceGroup(i+1)
            self.simulations.system.addForce(b)
            
        if (c.getNumBonds()):
            c.setForceGroup(i+1)
            self.simulations.system.addForce(c)
            
        if (d.getNumBonds()):
            d.setForceGroup(i+1)
            self.simulations.system.addForce(d)
    
    self.simulations.context.reinitialize()  

    # initialize positions and velocities
    if(reeds_input_files.state_files is None):
        if isinstance(reeds_input_files.coordinate_files, list):
          parmedSys = (load_file(reeds_input_files.parameter_file, reeds_input_files.coordinate_files[0]))
          self.simulations.context.setPositions(parmedSys.positions)
          self.simulations.context.setVelocitiesToTemperature(self.temperature)
          for i in range(self.num_replicas):
              parmedSys = (load_file(reeds_input_files.parameter_file, reeds_input_files.coordinate_files[i]))
              self.simulations.context.setPositions(parmedSys.positions)
              self.states[i] = self.simulations.context.getState(getPositions=True, getVelocities=True)
        
        else:
            self.simulations.context.setPositions(parmedSys.positions)
            self.simulations.context.setVelocitiesToTemperature(self.temperature)
            for i in range(self.num_replicas):
                self.states[i] = self.simulations.context.getState(getPositions=True, getVelocities=True)
    else:
        for i in range(self.num_replicas):
            self.simulations.loadState(reeds_input_files.state_files[i])
            self.states[i] = self.simulations.context.getState(getPositions=True, getVelocities=True)

    #self.simulations.reporters.append(app.PDBReporter("out_" + self.system_name + ".pdb", 10000, enforcePeriodicBox = True))
    #self.simulations.reporters.append(app.StateDataReporter(sys.stdout, 1000, step=True,
    #        potentialEnergy=True, temperature=True, kineticEnergy = True))

    # initialize output files, i.e. energy trajectories and repdat files
    self.ene_traj_filenames = ["ene_traj_" + self.system_name + "_" + str(i) for i in range(1, self.num_replicas +1)]
    self.ene_traj_files = [open(name, "w") for name in self.ene_traj_filenames]
    for file in self.ene_traj_files:
        file.write('{0: <15}'.format("t"))
        for i in range(1, self.num_endstates +1):
            file.write('{0: <15}'.format("V_" + str(i)))
        file.write('{0: <15}'.format("V_R") + "\n")

    self.repdat = open("repdat_" + self.system_name, "w")
    self.repdat.write('{0: <15}'.format("time") + '{0: <15}'.format("partner_i") + '{0: <15}'.format("partner_j")+ '{0: <15}'.format("s_i") + '{0: <15}'.format("s_j")+'{0: <15}'.format("position_i")+'{0: <15}'.format("position_j")+'{0: <15}'.format("probability")+'{0: <15}'.format("exchanged")+"\n")
    
    # repdat in GROMOS format to use with existing analysis script -> remove in the long run
    self.repdat_gromos = open("repdat_gromos_" + self.system_name, "w")
    self.repdat_gromos.write("#======================\n#REPLICAEXSYSTEM\n#======================\n#Number of temperatures:\t1\n#Dimension of temperature values:\t1")
    self.repdat_gromos.write("#Number of lambda values:\t" + str(self.num_replicas) + "\n")
    self.repdat_gromos.write("#T\t" + str(self.temperature) + "\n")
    self.repdat_gromos.write("#lambda\t")
    for s in self.s_values:
      self.repdat_gromos.write(" " + str(s))
    self.repdat_gromos.write("\n#s (RE-EDS)\t")
    for s in self.s_values:
      self.repdat_gromos.write(" " + str(s))
    self.repdat_gromos.write("\n")
    for i in range(self.num_endstates):
      self.repdat_gromos.write("#eir(s), numstate = " + str(i+1) + " (RE - EDS) ")
      for j in range(self.num_replicas):
        self.repdat_gromos.write(" " + str(self.energy_offsets[i]))
      self.repdat_gromos.write("\n")
    self.repdat_gromos.write("#\n\n")    
    self.repdat_gromos.write("pos\tID\tcoord_ID\tpartner\tpartner_start\tpartner_coord_ID\trun\tEpoti\tEpotj\tp\ts\t")
    for i in range(self.num_endstates):
      self.repdat_gromos.write("Vr" + str(i+1) + "\t")
    self.repdat_gromos.write("\n")

  def custom_reaction_field(self, end_state, original_nonbonded_force, active_particles, unperturbed_particles):
    """
    defines a reaction field with a shifting function (see A. Kubincova et al, Phys. Chem. Chem. Phys. 2020, 22)

    Parameters
    ----------
    end_state: int
      index of current end-state
    original_nonbonded_force: mm.NonbondedForce
      original nonbonded force of the system
    active_particles: List
      list of particle indices of current particles (either the particles of the current end-state or all unperturbed particles)
    unperturbed_particles: List
      list of unperturbed particles (e.g. solvent, protein, cofactor, etc.)

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
    scal = 'scaling_' + str(end_state)
    cutoff = original_nonbonded_force.getCutoffDistance()

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
    force_lj_crf.addGlobalParameter(scal, 1)
    force_lj_crf.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    force_lj_crf.setCutoffDistance(cutoff)
    force_lj_crf.setUseLongRangeCorrection(False)

    # copy per particle parameters from original nonbonded force
    for index in range(original_nonbonded_force.getNumParticles()):
      charge, sigma, epsilon = original_nonbonded_force.getParticleParameters(index)
      force_lj_crf.addParticle([charge, sigma, epsilon])
        
    # copy exceptions from original nonbonded force
    for index in range(original_nonbonded_force.getNumExceptions()):
      j, k, chargeprod, sigma, epsilon = original_nonbonded_force.getExceptionParameters(index)
      force_lj_crf.addExclusion(j, k)

    # set interaction groups -> end-state with itself and end-state with unperturbed particles (or only unperturbed with unperturbed)
    force_lj_crf.addInteractionGroup(active_particles, unperturbed_particles)
    if(active_particles != unperturbed_particles):
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
    force_lj_crf_one_four.setUsesPeriodicBoundaryConditions(True)

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
    force_crf_excluded.setUsesPeriodicBoundaryConditions(True)

    # copy excluded neighbors from original nonbonded force
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
    force_crf_self_term.setUsesPeriodicBoundaryConditions(True)

    # add self interaction of all current end-state/unperturbed particles
    for i in active_particles:
      ch1, _, _ = original_nonbonded_force.getParticleParameters(i)
      force_crf_self_term.addBond(i, i, [ch1*ch1])

    return force_lj_crf, force_lj_crf_one_four, force_crf_excluded, force_crf_self_term

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

    return sum_prefactors, prefactors

  def reference_state(self, pos):
    """
    calculates V_R of replica at position pos
    """
    Vi = np.array([self.simulations.context.getState(getEnergy=True, groups=1<<i+1).getPotentialEnergy().value_in_unit(u.kilojoules_per_mole) for i in range(self.num_endstates)])
    s = self.s_values[pos]
    V_R = - 1/(self.beta * s) * logsumexp(-self.beta * s * (Vi - self.energy_offsets))
          
    return V_R 

  def get_scaling(self, s):
    """
    calculates scaling factors for the end-state energies/forces
    """
    Vi = [self.simulations.context.getState(getEnergy=True, groups=1<<i+1).getPotentialEnergy().value_in_unit(u.kilojoules_per_mole) for i in range(self.num_endstates)]
    terms = np.exp(-self.beta * s * (Vi - self.energy_offsets))
    scaling_factors = terms / np.sum(terms)
    
    return scaling_factors

  def exchange_probability(self, partners):
    """
    calculates exchange probability between replicas at positions partners[0] and partners[1]
    """
    V_orig = np.array([0.,0.])
    V_exch = np.array([0.,0.])
    for i, p in enumerate(partners):
      self.simulations.context.setState(self.states[p])
      
      for j in range(self.num_endstates):
        self.simulations.context.setParameter('scaling_' + str(j), 1)
      
      Vi = [self.simulations.context.getState(getEnergy=True, groups=1<<j+1).getPotentialEnergy().value_in_unit(u.kilojoules_per_mole) for j in range(self.num_endstates)]
      
      # calculate reference state with original s value
      s = self.s_values[p]
      V_orig[i] = - 1/(self.beta * s) * logsumexp(-self.beta * s * (Vi - self.energy_offsets))
      
      # calculate reference state with partner's s value
      s = self.s_values[partners[-(i+1)]]
      V_exch[i] = - 1/(self.beta * s) * logsumexp(-self.beta * s * (Vi - self.energy_offsets))

    delta = np.sum(V_exch) - np.sum(V_orig)
    if delta < 0:
      return 1, V_orig, V_exch
    else:
      return np.exp(- self.beta * delta), V_orig, V_exch

  def calculate_free_energy_differences(self, s_index):
    """
    calculate the free energy differences of all end-state pairs for the s value with index s_index
    """
    ene_traj = pd.read_csv(self.ene_traj_filenames[s_index], header = [0], delim_whitespace = True)
    df = [- 1/self.beta * np.log(np.mean(np.exp(-self.beta * (ene_traj["V_" + str(j+1)] - ene_traj["V_R"])))/np.mean(np.exp(-self.beta * (ene_traj["V_" + str(i+1)] - ene_traj["V_R"])))) for i in range(self.num_endstates) for j in range(i+1, self.num_endstates)]
        
    return df

  def run(self):
    """
    perform a RE-EDS simulation
    """
    even = True
    time = self.initial_time
    step_size = self.integrators.getStepSize()._value
    run = 1

    for total_steps in range(0,self.reeds_simulation_variables.total_steps, self.reeds_simulation_variables.num_steps_between_exchanges):
      # print time every 1000th step
      if(not (total_steps % 1000)):
        print("time ", "{:.4f}".format(time))
        sys.stdout.flush()

      for idx, state in enumerate(self.states):
        # propagate replica at position idx
        self.simulations.context.setState(state)
        for step in range(self.reeds_simulation_variables.num_steps_between_exchanges):
          # calculate scaling factors for forces
          for i in range(self.num_endstates):
            self.simulations.context.setParameter('scaling_' + str(i), 1)

          scal = self.get_scaling(self.s_values[idx])        

          for j in range(self.num_endstates):
            self.simulations.context.setParameter('scaling_' + str(j), scal[j])

          # perform a simulation step
          self.simulations.step(1)

        # store state of current replica
        self.states[idx] = self.simulations.context.getState(getPositions=True, getVelocities=True)

      # print output to energy trajectories
      time += step_size * self.reeds_simulation_variables.num_steps_between_exchanges
      Vi = [[0.] * self.num_endstates] * self.num_replicas
    
      for idx, pos in enumerate(self.replica_positions):
        self.simulations.context.setState(self.states[pos])
        self.ene_traj_files[idx].write('{0: <14}'.format("{:.4f}".format(time)) + " ")

        for i in range(self.num_endstates):
          self.simulations.context.setParameter('scaling_' + str(i), 1)
          Vi[idx][i] = self.simulations.context.getState(getEnergy=True, groups=1<<i+1).getPotentialEnergy().value_in_unit(u.kilojoules_per_mole)
          self.ene_traj_files[idx].write('{0: <14}'.format("{:.10f}".format(Vi[idx][i])) + " ")
        
        self.ene_traj_files[idx].write('{0: <15}'.format("{:.10f}".format(self.reference_state(pos))))
        self.ene_traj_files[idx].write("\n")

      # perform replica exchanges      
      if(even):
        even = False
        begin = 0
      else:
        even = True
        begin = 1
        # if first replica doesn't have a partner -> print info to repdat file
        self.repdat_gromos.write("1\t1\t" + str(self.replica_positions[0]+1) + "\t" +  "1\t1\t" + str(self.replica_positions[0]+1) + "\t" + str(run) + "\t0\t0\t0\t0")
        for j in range(self.num_endstates):
          self.repdat_gromos.write("\t" + str(Vi[0][j]))
        self.repdat_gromos.write("\n")
          
      # alternate replica partners (i.e. even = s values at 0-1, 2-3, 4-5, ... and odd = s values at 1-2, 3-4, 5-6, ...)    
      for i in range(begin,self.num_replicas-1,2):
        # calculate exchange probability
        partners = [self.replica_positions[i],self.replica_positions[i+1]]
        rnd = np.random.uniform(0,1)
        prob, V_orig, V_exch = self.exchange_probability(partners)

        # print info to repdat file
        self.repdat.write('{0: <14}'.format("{:.4f}".format(time)) + " ")
        self.repdat.write('{0: <14}'.format(str(i)) + " ")
        self.repdat.write('{0: <14}'.format(str(i+1)) + " ")
        p1 = partners[0]
        p2 = partners[1]
        self.repdat.write('{0: <14}'.format("{:.4f}".format(self.s_values[p1])) + " ")
        self.repdat.write('{0: <14}'.format("{:.4f}".format(self.s_values[p2])) + " ")
        self.repdat.write('{0: <14}'.format(str(p1)) + " ")
        self.repdat.write('{0: <14}'.format(str(p2)) + " ")
        self.repdat.write('{0: <14}'.format("{:.4f}".format(prob)) + " ")
        
        if(prob > rnd):
          # perform exchange
          self.s_values[p1], self.s_values[p2] = self.s_values[p2], self.s_values[p1]
          self.replica_positions[i] = p2
          self.replica_positions[i+1] = p1

          # print info to repdat file
          self.repdat.write('{0: <14}'.format("1"))
          self.repdat_gromos.write(str(i+1) + "\t" + str(i+1) + "\t" + str(self.replica_positions[i]+1) + "\t" + str(i+2) + "\t" + str(i+2) + "\t" + str(self.replica_positions[i+1]+1)+ "\t" + str(run) + "\t" + str(V_orig[0]) + "\t" + str(V_orig[1]) + "\t" + str(prob) + "\t1")
          for j in range(self.num_endstates):
            self.repdat_gromos.write("\t" + str(Vi[i][j]))
          self.repdat_gromos.write("\n")
          self.repdat_gromos.write(str(i+2) + "\t" + str(i+2) + "\t" + str(self.replica_positions[i+1]+1) + "\t" + str(i+1) + "\t" + str(i+1) + "\t" + str(self.replica_positions[i]+1)+ "\t" + str(run) + "\t" + str(V_orig[1]) + "\t" + str(V_orig[0]) + "\t" + str(prob) + "\t1")
          for j in range(self.num_endstates):
            self.repdat_gromos.write("\t" + str(Vi[i+1][j]))
          self.repdat_gromos.write("\n")

        else:
          # print info to repdat file
          self.repdat.write('{0: <14}'.format("0"))
          self.repdat_gromos.write(str(i+1) + "\t" + str(i+1) + "\t" + str(self.replica_positions[i]) + "\t" + str(i+2) + "\t" + str(i+2) + "\t" + str(self.replica_positions[i+1])+ "\t" + str(run) + "\t" + str(V_orig[0]) + "\t" + str(V_orig[1]) + "\t" + str(prob) + "\t0")
          for j in range(self.num_endstates):
            self.repdat_gromos.write("\t" + str(Vi[i][j]))
          self.repdat_gromos.write("\n")
          self.repdat_gromos.write(str(i+2) + "\t" + str(i+2) + "\t" + str(self.replica_positions[i+1]) + "\t" + str(i+1) + "\t" + str(i+1) + "\t" + str(self.replica_positions[i])+ "\t" + str(run) + "\t" + str(V_orig[1]) + "\t" + str(V_orig[0]) + "\t" + str(prob) + "\t0")
          for j in range(self.num_endstates):
            self.repdat_gromos.write("\t" + str(Vi[i+1][j]))
          self.repdat_gromos.write("\n")

        self.repdat.write("\n")

      # if last replica doesn't have a partner -> print info to repdat file
      if(i+2 < self.num_replicas):
        self.repdat_gromos.write(str(i+3) + "\t" + str(i+3) + "\t" + str(self.replica_positions[i+2]+1) + "\t" + str(i+3) + "\t" + str(i+3) + "\t" + str(self.replica_positions[i+2]+1)+ "\t" + str(run) + "\t0\t0\t" + str(0) + "\t0")
        for j in range(self.num_endstates):
          self.repdat_gromos.write("\t" + str(Vi[i+2][j]))
        self.repdat_gromos.write("\n")
          
      run += 1

      # store state files every 1000 steps and flush
      if(not (total_steps % 1000)):
        for idx, pos in enumerate(self.replica_positions):
          self.simulations.context.setState(self.states[pos])
          self.simulations.saveState(self.system_name + "_state_s_" + str(idx))
        for idx in range(self.num_replicas):
          self.ene_traj_files[idx].flush()
        self.repdat.flush()
        self.repdat_gromos.flush()
        sys.stdout.flush()
