import numpy as np
import scipy.constants as consts
import forces as fc
import random
import copy
import uuid

TEMPERATURE = 310.15 # degrees kelvin
BOLTZMANN = 1/(consts.Boltzmann * TEMPERATURE) #boltzman factor
MAX_TRANSLATE = 5  #5 #maximum step size that can be taken along each axis, given in nanometers
BETA = 1 # formulas alread calculate energy as U*BOLTZMANN, so it is built in otherwise BETA=BOLTZMANN


#simple molecule (sphere) class, includes x,y,z coordinates and radius
class Molecule:
  def __init__(self, pos, radius):
    self.pos = pos #np.array of size 3
    self.radius = radius
  
  def move(self, delta_pos):
    self.pos += delta_pos

  def update_pos(self, pos):
    self.pos = pos
  
  
#compute energy of all molecules with respect to moved/fixed.
#attempted multithreading using joblib (was not faster)
def get_energy(mol_fixed, universe, mol_moved = None):
  idx = universe.molecules.index(mol_fixed)
  nbs = [m for m in universe.molecules]
  del nbs[idx] #exclude molecule from universe so we dont calculate its energy wrt. it self

  if mol_moved != None: # if we moved the molecule
    energy = sum([universe.force_fun(mol_moved,m2) for m2 in nbs])
  else: #just compute energy for the fixed/previous molecule
    energy = sum([universe.force_fun(mol_fixed,m2) for m2 in nbs])
  
  if np.isnan(energy): #incase of problematic force function
    print("encountered nan energy, converting to 100000000 (consider fixing this) \n")
    energy = 100000000 

  return energy

#check if a molecule (sphere) is within the bounding box
def within_box(molecule, delta_pos, box_size):
  new_pos = molecule.pos + delta_pos + molecule.radius
  
  if np.any(new_pos < np.array([0,0,0])) or np.any(new_pos > box_size): #check if any coordinate is out of bounds
    return False
  return True


#attemt 1 step, return if successful or not, and update the univrese
#assumes energies are calculated as B*U, so the beta used is 1
def simple_step(universe,molecule):
  step_taken = False #flag for indicating step succes
  
  e_current = get_energy(molecule, universe) #cur energy

  delta_pos = np.random.uniform(-MAX_TRANSLATE,MAX_TRANSLATE,3) #random offset

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
def spawn_uniformly_random(num_spheres, box_dimensions, radii, max_attempts = 1000):

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


#wrapper for choosing between pased molecules or generating them randomly
def create_initial_molecules(box_size,num_molecules,radii,own_molecules):
  if own_molecules != None: #own deffined molecule list
    return own_molecules
  else: #generate uniformly random molecules with radii
    return spawn_uniformly_random(num_molecules,box_size,radii)




#Universe class for creating and running simulations
#box_size = universe cobe from (0,0,0) to box size
#num_molecules = the desired amount of molecules
#radii = list of radii for each of the desired molecules
#seed = for controling reproducability
#own_molecules = if you want to pass own list of point arrays for molecules
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
    
    self.move_ctr += 1
    self.acceptance_list.append(move_bool)

  #take a step (maybe add support for passing a step function)
  def make_step(self):
    if (len(self.molecules)<= 0): #check if step possible
      print("No molecules in universe, cant perform a step")
      
    else: #perform a step
      mol = self.select_molecule()
      step_bool = simple_step(self,mol)
      self.update_move_stats(step_bool)
  
  #take a step with thresholding
  def make_step_thresh(self):
    if (len(self.molecules)<= 0): #check if step possible
      print("No molecules in universe, cant perform a step")
      
    else: #perform a step
      mol = self.select_molecule()
      step_bool = threshold_step(self,mol)
      self.update_move_stats(step_bool)


#the distance between two molecules
def dist(m1,m2):
  return np.linalg.norm(m2.pos - m1.pos)

#the surface distance between two molecules
def inter_dist(m1,m2):
  return np.linalg.norm(m2.pos - m1.pos) -  m1.radius - m2.radius


#save a simulation - point frames, radii, acceptance list
#file name: name, num frames saved, boxsize x and y, seed, stepsize, frames skiped, uniqe chars
def save_molecule_steps(points,radii,accs,box_size,seed,skipsize, stepsize,name="simV3R"):
  #create unique name
  name = name + "_" + str(len(points)) + "_" + str(box_size) + "_" + str(seed) + "_"  + str(stepsize) + "_" + str(skipsize) + "_" + str(uuid.uuid4().hex)[:5] + ".npy"

  with open(name, 'wb') as f:
    np.save(f,np.array(points))
    np.save(f,np.array(radii))
    np.save(f,np.array(accs))

  return name

#Load a simulation - point frames, radii, acceptance list 
def load_molecule_steps(name):
  with open(name, 'rb') as f:
    poss = np.load(f)
    radii = np.load(f)
    accs = np.load(f)
  return poss, radii, accs




#----------------------- Attempted to do grouping of molecules -----------------------

#attemt 1 step, return if successful or not, and update the univrese
#if move results in overlapping spheres then move molecule back until this is no longer the case
#assumes energies are calculated as B*U, so the beta used is 1
def threshold_step(universe,molecule):
  step_taken = False
  
  e_current = get_energy(molecule, universe)

  delta_pos = np.random.uniform(-MAX_TRANSLATE,MAX_TRANSLATE,3)

  if within_box(molecule, delta_pos, universe.box_size): #check that move is possible in universe
    mol_temp= copy.deepcopy(molecule)
    mol_temp.move(delta_pos)
    #pos = molecule.pos + delta_pos # compare this to all
    
    idx = universe.molecules.index(molecule)
    nbs = [m for m in universe.molecules]
    del nbs[idx]
    gap = np.array([inter_dist(mol_temp,m) for m in nbs])
    if any(x < 0 for x in gap): # check if we post move are trying to move inside another molecule
      idx = gap.index(max(gap)) # get molecule idx with biggest overlap
      counter_dist = mol_temp.radius + nbs[idx].radius - dist(nbs[idx],mol_temp) #to make them overlap less
      unit_vec = (mol_temp.pos - nbs[idx].pos) / np.linalg.norm(mol_temp.pos - nbs[idx].pos)
      delta_pos += counter_dist * unit_vec

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

#group of molecules
class Group:
  def __init__(self,molecules, id = 0):
    self.molecules = molecules
    self.id = id
    self.total_mass = np.sum([((4/3)*np.pi*m.radius**3) for m in molecules])

  def update_mass(self,molecule):
    self.total_mass += (4/3)*np.pi*molecule.radius**3

  def get_weight(self,molecule):
    return ((4/3)*np.pi*molecule.radius**3) / self.total_mass

#creates initial groups using a list of ids for each molecule
def init_groups(molecules, id_list):
  grps = [Group(m,i) for m,i in zip(molecules,id_list)]
  return grps


class GroupUniverse:
  def __init__(self,box_size, num_molecules,radii,seed,own_molecules=None, force_fun = fc.total_force_molecule):
    #Reproduction setup
    self.seed = seed
    random.seed(seed)
    
    #molecule/world setup
    self.box_size = box_size
    self.num_molecules = num_molecules
    #define the molecular groupings of the universe
    self.molecules = create_initial_molecules(box_size,num_molecules, radii,own_molecules) 
    self.groups = init_groups(self.molecules, np.arange(num_molecules)) #attach a grp ID to every molecule at the beginning
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
  
  #select random Group
  def select_molecule(self):
    return random.choice(self.groups)

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
    
    self.move_ctr += 1
    self.acceptance_list.append(move_bool)

  #take a step (maybe add support for passing a step function)
  def make_step(self):
    if (len(self.molecules)<= 0): #check if step possible
      print("No molecules in universe, cant perform a step")
      
    else: #perform a step
      mol = self.select_molecule()

      step_bool = simple_step(self,mol)
      self.update_move_stats(step_bool)
    