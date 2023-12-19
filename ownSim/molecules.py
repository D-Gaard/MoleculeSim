import numpy as np
import scipy.constants as consts
import forces as fc
import random
import copy

VISCOSITY = 1.2    # might not be accurate
TEMPERATURE = 310.15 # degrees kelvin
BOLTZMANN = 1/(consts.Boltzmann * TEMPERATURE) #denoted beta
MAX_TRANSLATE = 5 #maximum step size that can be taken along each axis, given in nanometers
BETA = BOLTZMANN

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

# def get_energy(mol_fixed,universe): #compute the energy of all molecules with respect to
#   u_nom1 = [m for m in universe.molecules if all(m.pos != mol_fixed.pos)]
#   energy = sum([fc.total_force_molecule(mol_fixed,m2) for m2 in u_nom1])
#   return energy
  
def get_energy(mol_fixed, universe, mol_moved = None): #compute the energy of all molecules with respect to
  idx = universe.molecules.index(mol_fixed)
  nbs = [m for m in universe.molecules]
  del nbs[idx]
  if mol_moved != None: # if we moved the molecule
    energy = sum([fc.total_force_molecule(mol_moved,m2) for m2 in nbs])
  else: #just compute energy for the fixed/previous molecule
    energy = sum([fc.total_force_molecule(mol_fixed,m2) for m2 in nbs])
  return energy


def within_box(molecule, delta_pos, box_size):
  new_pos = molecule.pos + delta_pos + molecule.radius
  
  if np.any(new_pos > np.array([0,0,0])) or np.any(new_pos < box_size): #check if any coordinate is out of bounds
    return False
  return True

# apply basic movement to molecule/group
def step(universe,molecule,window = []):
  step_taken = False
  # get current energy
  e_current = get_energy(molecule, universe)
  while (not step_taken):
    # Get step vector and add it to molecule copy
    delta_pos = np.random.uniform(-MAX_TRANSLATE,MAX_TRANSLATE,3)

    if within_box(molecule, delta_pos, universe.box_size): #check that move is possible in universe

      mol_copy = copy.deepcopy(molecule)
      mol_copy.move(delta_pos)

      #calculate energy for potential new location
      e_new = get_energy(molecule, universe, mol_copy)

      # if move is accepted, make the real molecule perform the step
      accepted = fc.accept_move(e_current,e_new, BETA)
      if accepted:
        molecule.move(delta_pos)
        step_taken = True

  


def create_initial_molecules(box_size,num_molecules,radii,own_molecules):
  if own_molecules != None: #own deffined molecule list
    return own_molecules
  else: #generate uniformly random molecules with radii
    return [Molecule(np.array([random.uniform(0,box_size[0]), 
                      random.uniform(0,box_size[1]), 
                      random.uniform(0,box_size[2]) ]),radii[i]) for i in range(num_molecules)]


class SimpleUniverse:
  def __init__(self,box_size, num_molecules,radii,seed,own_molecules=None):
    #Reproduction setup
    self.seed = seed
    random.seed(seed)
    
    #molecule/world setup
    self.box_size = box_size
    self.num_molecules = num_molecules
    #define the molecular groupings of the universe
    self.molecules = create_initial_molecules(box_size,num_molecules, radii,own_molecules) 



  

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
  return np.linalg.norm(m2.pos - m1.pos)

def inter_dist(m1,m2):
  return np.linalg.norm(m2.pos - m1.pos) -  m1.radius - m2.radius

def avg_dist(frames):
    dist = []
    for frame in frames:
        cum = 0
        for i in range(len(frame)):
            mol1 = frame[i]
            for j in range(i+1,len(frame)):
                mol2 = frame[j]
                current_dist = np.linalg.norm([mol1,mol2])
                cum += current_dist
        avg = cum / ((len(frame)-1) * (len(frame)-2))
        dist.append(avg)
    return dist