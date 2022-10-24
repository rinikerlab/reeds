import sys
import numpy as np

"""
create perturbation topology for simple dual-topology case
assume that the end-states are listed first in the topology
"""

top = sys.argv[1]
num_endstates = int(sys.argv[2])

with open(top) as file:
  line = file.readline()
  while line:  

    if "ATOM_NAME" in line:
      line = file.readline()
      line = file.readline()
      num_atoms = 0
      while "FLAG" not in line:
        num_atoms += len(line.split())
        line = file.readline()
    
      print(num_atoms)

    elif "RESIDUE_POINTER" in line:
      line = file.readline()
      line = file.readline()
      residue_start = []
      while "FLAG" not in line:
        residue_start.extend(map(int, line.split()))
        line = file.readline()
        
    elif "CHARGE" in line:
      line = file.readline()
      line = file.readline()
      charges = []
      while "FLAG" not in line:
        charges.extend(map(float, line.split()))
        line = file.readline()
        
    elif "LENNARD_JONES_ACOEF" in line:
      line = file.readline()
      line = file.readline()
      lj_acoeff = []
      while "FLAG" not in line:
        lj_acoeff.extend(map(float, line.split()))
        line = file.readline()    
        
    elif "LENNARD_JONES_BCOEF" in line:
      line = file.readline()
      line = file.readline()
      lj_bcoeff = []
      while "FLAG" not in line:
        lj_bcoeff.extend(map(float, line.split()))
        line = file.readline()    
        
    elif "ATOM_TYPE_INDEX" in line:
      line = file.readline()
      line = file.readline()
      atom_type_index = []
      while "FLAG" not in line:
        atom_type_index.extend(map(int, line.split()))
        line = file.readline()   
   
    elif "NONBONDED_PARM_INDEX" in line:
      line = file.readline()
      line = file.readline()
      nonbonded_parm_index = []
      while "FLAG" not in line:
        nonbonded_parm_index.extend(map(int, line.split()))
        line = file.readline()     
        
    else:
      line = file.readline()
        
        

residue_start = [i-1 for i in residue_start]
atom_type_index = [i-1 for i in atom_type_index]
nonbonded_parm_index = [i-1 for i in nonbonded_parm_index]
residue_end = residue_start[1:]
residue_end.append(num_atoms)

ntypes = int(np.sqrt(len(nonbonded_parm_index)))
        
with open("perturbations.py", "w") as file:
  file.write("topology_type = \"amber\"\n\n")
  file.write("perturbed_particles = {}\n")
  for i in range(num_endstates):
    for j in range(residue_start[i], residue_end[i]):
      file.write(f"perturbed_particles[{j}] = [")
      print(j)
      for k in range(0,i):
        file.write("[0.0, 0.0, 0.0], ")
        
      index = nonbonded_parm_index[ntypes * (atom_type_index[j]) + atom_type_index[j]]
      file.write(f"[{charges[j]}, {lj_acoeff[index]}, {lj_bcoeff[index]}]")      
      
      if(i < num_endstates -1):
        file.write(", ")
        for k in range(i+1, num_endstates-1):
          file.write("[0.0, 0.0, 0.0], ")
          
        file.write("[0.0, 0.0, 0.0]")
      
      file.write("]\n")
    file.write("\n")
