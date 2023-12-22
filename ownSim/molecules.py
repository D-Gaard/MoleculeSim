import numpy as np
import scipy.constants as consts
import forces as fc
import random
import copy

VISCOSITY = 1.2    # might not be accurate
TEMPERATURE = 310.15 # degrees kelvin
BOLTZMANN = 1/(consts.Boltzmann * TEMPERATURE) #denoted beta
MAX_TRANSLATE = 5 #maximum step size that can be taken along each axis, given in nanometers
BETA = 1 # formulas alread calculate energy as U*BOLTZMANN, so it is built in

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
  


# #NOTE
# #include diffusion coefficient/mass for groups? how to deal with multiple groups/molecules

# #Contains a collection of atoms, applies the same update to the entire group
# class GroupMolecule:
#   def __init__(self):
#     self.molecules = []
  
#   def add_molecule(self, mol):
#     self.molecules.append(mol)


#energy of all molecules with respect to moved/fixed. 
def get_energy(mol_fixed, universe, mol_moved = None):
  idx = universe.molecules.index(mol_fixed)
  nbs = [m for m in universe.molecules]
  del nbs[idx]
  if mol_moved != None: # if we moved the molecule
    energy = sum([universe.force_fun(mol_moved,m2) for m2 in nbs])
  else: #just compute energy for the fixed/previous molecule
    energy = sum([universe.force_fun(mol_fixed,m2) for m2 in nbs])

  if np.isnan(energy):
    print("encountered nan energy, converting to 0 (consider fixing this) \n")
    energy = 0

  return energy


def within_box(molecule, delta_pos, box_size):
  new_pos = molecule.pos + delta_pos + molecule.radius
  
  if np.any(new_pos < np.array([0,0,0])) or np.any(new_pos > box_size): #check if any coordinate is out of bounds
    return False
  return True

# apply basic movement to molecule runs until a move for the given molecule is found
# not ideal to use
def step(universe,molecule,window = []):
  step_taken = False
  # get current energy
  e_current = get_energy(molecule, universe)
  print("energy", e_current,type(e_current))
  while (not step_taken):
    # Get step vector and add it to molecule copy
    delta_pos = np.random.uniform(-MAX_TRANSLATE,MAX_TRANSLATE,3)
    print("am i in box?", within_box(molecule, delta_pos, universe.box_size))
    if within_box(molecule, delta_pos, universe.box_size): #check that move is possible in universe
      print("im in")
      mol_copy = copy.deepcopy(molecule)
      mol_copy.move(delta_pos)

      #calculate energy for potential new location
      e_new = get_energy(molecule, universe, mol_copy)

      # if move is accepted, make the real molecule perform the step
      accepted = fc.accept_move(e_current,e_new, BETA)
      if accepted:
        molecule.move(delta_pos)
        step_taken = True




#attemt 1 step, return if successful or not, and update the univrese
#assumes energies are calculated as B*U, so the beta used is 1
def simple_step(universe,molecule):
  step_taken = False
  
  e_current = get_energy(molecule, universe)

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

  return step_taken


#uniformly attempt to spawn spheres, if overlap, then a new position is chosen (so a mc scheeme)
#fails after max_attempts fails for a single molecule
def spawn_uniformly_random(num_spheres, box_dimensions, radii, max_attempts = 100):

  def check_overlap(new_sphere, existing_spheres):
    for sphere in existing_spheres:
        distance = np.linalg.norm(new_sphere[:3] - sphere[:3])
        if distance < new_sphere[3] + sphere[3]:
            return True
    return False

  spheres = []
  box_min = np.array([0, 0, 0])
  box_max = np.array(box_dimensions)
  attempts = 0

  for i in range(num_spheres):
    radius = radii[i]
    position = np.random.uniform(box_min + radius, box_max - radius)
    new_sphere = np.append(position, radius)

    attempts = 0
    while check_overlap(new_sphere, spheres):
        position = np.random.uniform(box_min + radius, box_max - radius)
        new_sphere = np.append(position, radius)
        attempts += 1

        if attempts > max_attempts:
          print("Unable to spawn points withing the box")
          return None

    spheres.append(new_sphere)

  return [Molecule(np.array([x,y,z]),r) for (x,y,z,r) in spheres]


def create_initial_molecules(box_size,num_molecules,radii,own_molecules):
  if own_molecules != None: #own deffined molecule list
    return own_molecules
  else: #generate uniformly random molecules with radii
    return spawn_uniformly_random(num_molecules,box_size,radii)
    #return [Molecule(np.array([random.uniform(0,box_size[0]), 
    #                  random.uniform(0,box_size[1]), 
    #                  random.uniform(0,box_size[2]) ]),radii[i]) for i in range(num_molecules)]

#spawn molecules within a grid
#def grid_spawner(box_size,radius, spacing):
#  x,y,z = np.arange(radius, box_size[0]- radius, step = 2*radius + spacing)


#force_fun = force function used between molecules (vdw, steric and electrostatic as default)
class SimpleUniverse:
  def __init__(self,box_size, num_molecules,radii,seed,own_molecules=None, force_fun = fc.total_force_molecule):
    #Reproduction setup
    self.seed = seed
    random.seed(seed)
    
    #molecule/world setup
    self.box_size = box_size
    self.num_molecules = num_molecules
    #define the molecular groupings of the universe
    self.molecules = create_initial_molecules(box_size,num_molecules, radii,own_molecules) 
    #define force use
    self.force_fun = force_fun


    #stats:
    self.move_ctr = 0 #amount of steps taken
    #coutners for acceptence/rejection, and list of actions
    self.acceptance_list = []
    self.acceptance_ctr = 0
    self.rejction_ctr = 0 


  #select random molecule
  def select_molecule(self):
    return random.choice(self.molecules)
  
  #get state of universe
  # 0 = position (default)
  # 1 = with radius
  def get_state(self,with_radius = False):
    if (with_radius):
      return np.array([mol.pos for mol in self.molecules]), np.array([mol.radius for mol in self.molecules])
    else:
      return np.array([mol.pos for mol in self.molecules])
  
  #get stats
  # 0 (default) == everything
  def get_stats(self, state=0):
    if (state==0):
      return  self.acceptance_ctr, self.rejction_ctr, self.acceptance_list, self.move_ctr
  
  #update move stats, should be caled after/at the end of move
  def update_move_stats(self, move_bool):
    if move_bool:
      self.acceptance_ctr += 1
    else:
      self.rejction_ctr += 1
    
    self.acceptance_list.append(move_bool)

  #take a step (maybe add support for passing a step function)
  def make_step(self):
    if (len(self.molecules)<= 0): #check if step possible
      print("No molecules in universe, cant perform a step")
      
    else: #perform a step
      mol = self.select_molecule()

      step_bool = simple_step(self,mol)
      self.update_move_stats(step_bool)



    



#the distance between two molecules
def dist(m1,m2):
  return np.linalg.norm(m2.pos - m1.pos)

def inter_dist(m1,m2):
  return np.linalg.norm(m2.pos - m1.pos) -  m1.radius - m2.radius

# def avg_dist(frames):
#     dist = []
#     for frame in frames:
#         cum = 0
#         for i in range(len(frame)):
#             mol1 = frame[i]
#             for j in range(i+1,len(frame)):
#                 mol2 = frame[j]
#                 current_dist = np.linalg.norm(mol2-mol1)
#                 cum += current_dist
#         avg = cum / ((len(frame)-1) * (len(frame)-2))
#         dist.append(avg)
#     return dist
