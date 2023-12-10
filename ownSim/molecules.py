import numpy as np
import scipy.constants as consts
import forces as fc
import random
import copy

VISCOSITY = 1.2    # might not be accurate
TEMPERATURE = 310.15 # degrees kelvin
BOLTZMANN = 1/(consts.Boltzmann * TEMPERATURE) #denoted beta
MAX_TRANSLATE = 5 #maximum step size that can be taken along each axis, given in nanometers
BETA = 1

#NOTE
#Considerations: self.mass, group, group movement, diffusion coefficient or similar?

#molecule class, includes x,y,z coordinates and radius
class Molecule:
  def __init__(self, pos, radius):
    self.pos = pos #np.array of size 3
    self.radius = radius
  
  def move(self, delta_pos):
    self.pos += delta_pos

  def update_pos(self, pos):
    self.pos = pos
  


#NOTE
#include diffusion coefficient/mass for groups? how to deal with multiple groups/molecules

#Contains a collection of atoms, applies the same update to the entire group
class GroupMolecule:
  def __init__(self):
    self.molecules = []
  
  def add_molecule(self, mol):
    self.molecules.append(mol)



def dummy_move(mol, threshold):
  x = np.random.uniform(0, 1)

def get_energy(mol_fixed,universe): #compute the energy of all molecules with respect to
  u_nom1 = [m for m in universe.molecules if all(m.pos != mol_fixed.pos)]
  energy = sum([fc.total_force_molecule(mol_fixed,m2) for m2 in u_nom1])
  return energy


# apply basic movement to molecule/group
def step(universe,molecule,window = []):
  step_taken = False
  # get current energy
  e_current = get_energy(molecule, universe)
  while (not step_taken):
    # Get step vector and add it to molecule copy
    delta_pos = np.random.uniform(-MAX_TRANSLATE,MAX_TRANSLATE,3)
    mol_copy = copy.deepcopy(molecule)
    mol_copy.move(delta_pos)

    #calculate energy for potential new location
    e_new = get_energy(mol_copy,universe)

    # if move is accepted, make the real molecule perform the step
    accepted = fc.accept_move(e_current,e_new, BETA)
    if accepted:
      molecule.move(delta_pos)
      step_taken = True



class SimpleUniverse:
  def __init__(self,molecules,seed):
    self.seed = seed
    self.molecules = molecules #define the molecular groupings of the universe

    random.seed(seed)

  #def update_group(self,): #moves a single group based on forces considered

  #select random molecule
  def select_molecule(self):
    return random.choice(self.molecules)
  
  #get state of universe
  def get_state(self):
    return np.array([mol.pos for mol in self.molecules])
    
  def time_step(self): #apply a single timestep update of all groups
    for elm in self.molecules:
      movement = 0#step(elm ,window)
      elm.move(movement)

#the distance between two molecules
def dist(m1,m2):
  return np.linalg.norm([m1.pos,m2.pos])