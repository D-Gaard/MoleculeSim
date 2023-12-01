import numpy as np
import scipy.constants as consts

VISCOSITY = 1.2    # might not be accurate
TEMPERATURE = 310.15 # degrees kelvin
BOLTZMANN = 1/(consts.Boltzmann * TEMPERATURE) #denoted beta

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

def get_energy(mol_fixed,mol_universe): #compute the energy of all molecules with respect to 
  return 0


# apply basic movement to molecule/group
def step(universe,molecule,window):
  step_taken = False
  #while(not step_taken):
    #...
  return 0


class SimpleUniverse:
  def __init__(self,molecules):
    self.molecules = molecules #define the molecular groupings of the universe

  #def update_group(self,): #moves a single group based on forces considered

  def time_step(self): #apply a single timestep update of all groups
    for elm in self.molecules:
      movement = 0#step(elm ,window)
      elm.move(movement)

#the distance between two molecules
def dist(m1,m2):
  return np.linalg.norm([m1.pos,m2.pos])