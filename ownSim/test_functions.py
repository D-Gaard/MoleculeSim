import molecules as mol
import forces as fc
import numpy as np
import copy

# function to yield energies and step acceptance for different distances between 2 molecules.03d
def test_step_acceptance(mol1,mol2, universe):
  e_current = mol.get_energy(molecule, universe)

  inter = mol.inter_dist(mol1,mol2)
  dists = np.linspace(0,inter, 20) # evenly spaced distances between
  delta_moves = [(mol2.pos - mol1.pos - mol2.radius - mol1.radius)/dist for dist in dists] #get possible steps
  
  energies = []
  accepts = []
  for move in delta_moves:  
      mol_copy = copy.deepcopy(mol1)
      mol_copy.move(move)

      #calculate energy for potential new location
      e_new = mol.get_energy(mol_copy,universe)

      # if move is accepted, make the real molecule perform the step
      accepted = fc.accept_move(e_current,e_new, mol.BETA)

      energies.append(e_new)
      accepts.append(accepted)
  
  return e_current, energies, accepts