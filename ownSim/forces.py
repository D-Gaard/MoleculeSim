import numpy as np
import random
import scipy.constants as consts
#from molecules import dist 
#import molecules as mc
#from joblib import wrap_non_picklable_objects, delayed


#BETA = 0.38 # std of diameter distribution of casein micelles

#Van der Wall constants:
HAMAKER = 1.0 # Estimate of Hamaker constant for casein micelles, 

#Electri repulsion constants:
EPS0 = consts.epsilon_0#electric permittivity of vacuum
EPS =  72#dielectric constant of the solvent, estimated to be in range [70,75] so far based on papers
PHII = -8 #electrostaticpotential at the surface of particle i, generalized to be 8 based on papers
#KAPPA = #inverse Debye length
TEMPERATURE = 309.15 #kelvin = 36C
I = 0.08 #Ionic strength, defined as (M,mol L)

#steric repulstion constats:
CULLING_PERCENTAGE = 0.75 #75% are cleaved off
SIGMA = 0.006 * (1-CULLING_PERCENTAGE) #grafting density multiplied by cleaving (75% are cleaved off)
H = 7 #width of the polyelectrolyte brush in nm
MAX_FORCE_VAL = 99999999999.



def kappa():
  nom = EPS0 * EPS * consts.R * TEMPERATURE
  denom = consts.physical_constants["Faraday constant"][0]**2 * I
  return np.sqrt(nom/denom) * 10 ** (9)

def elec_repv2(m1,m2):
  h = dist2(m1.pos,m2.pos)-m1.radius-m2.radius #mc.dist(m1,m2)
  left = 2*np.pi*m1.radius*EPS0*EPS*(PHII**2) #only using m1 radius here perhaps need change
  right1 = kappa()**(-1)
  right2 = np.exp(-(right1 * h))
  right3 = np.log(1+right2)
  print("left: ", left, "kappa", kappa(), "kappa inv",kappa()**(-1), ", kappa exp: ", right1, "h:", h, "np exp: ", right2, ", numpy log: ", right3,"res", left * right3 )
  return left * right3 

#Van der Waal (w.r. to m1)
#@delayed
#@wrap_non_picklable_objects
def vdw(m1,m2):
  _2a1a2 = 2*m1.radius*m2.radius
  r2 = dist2(m1.pos,m2.pos)**2
  square_p = (m1.radius + m2.radius)**2
  square_n = (m1.radius - m2.radius)**2

  fst_term = _2a1a2/(r2 - square_p)
  snd_term = _2a1a2/(r2 - square_n)
  thrd_term = np.log((r2 - square_p)/(r2 - square_n))
  return (-1/6) * HAMAKER*(fst_term + snd_term + thrd_term)

#electrostatic repulsion (w.r. to m1)
def elec_rep(m1,m2):
  U_el = EPS0 * EPS * (PHII**2) * np.log(1 + np.exp( -(dist2(m1.pos,m2.pos)-m1.radius-m2.radius)))
  #print(np.sqrt(2/(EPS0*EPS*np.log(1 + np.exp(dist(m1,m2)-m1.radius-m2.radius)))))
  return U_el * 10 ** (8)


#electrostatic repulsion (w.r. to m1)
def elec_rep3(m1,m2):
  #BU_el = EPS0 * EPS * (PHII**2) * np.log(1 + np.exp(dist(m1,m2)-m1.radius-m2.radius))
  #print(np.sqrt(2/(EPS0*EPS*np.log(1 + np.exp(dist(m1,m2)-m1.radius-m2.radius)))))
  KAPPA = 1
  U_el = (SIGMA*m1.radius)/(EPS*EPS0*(1+KAPPA*m2.radius))
  return U_el


#steric (w.r. to m1)
def steric(m1,m2):
  h = (dist2(m1.pos,m2.pos)-m1.radius-m2.radius)

  if (h > 2*H):
    return 0
  
  a_eff = ((1/m1.radius) + (1/m2.radius)) ** (-1) # maybe a_eff = m1.radius
  fst_nominator = 16 * np.pi * a_eff * H * H * (SIGMA ** (3/2))
  fst_denominator = 35 # * SIGMA

  y = (h / (2*H))

  snd = 28 * (y**(-1/4) - 1)
  trd = (20/11) * (1 - y**(11/4))
  fth = 12 * (y - 1)

  U_st = (fst_nominator / fst_denominator) * (snd + trd + fth)

  return U_st #*10**(-4.6)


# def total_force_molecule(m1,m2):
#   return vdw(m1,m2) + elec_rep(m1,m2) + steric(m1,m2) 

def total_force_molecule(m1,m2,threshold = 1):
  if (inter_dist2(m1.pos,m2.pos, m1.radius, m2.radius) > threshold): #check if surface distance is possible
    return vdw(m1,m2)  + steric(m1,m2) + elec_rep(m1,m2)
  else: #return impossible energy -> rejection
    return MAX_FORCE_VAL



#calculate if move is accepted based on forces
def accept_move(ePrev,eNew,Beta):
  proba = 0
  if eNew-ePrev <= 0:
    proba = 1
  else:
    proba = min(1, np.exp(-Beta*(eNew-ePrev)))
  return random.random() < proba



def dist2(p1,p2):
  return np.linalg.norm(p2 - p1)

def inter_dist2(p1,p2,r1,r2):
  return np.linalg.norm(p2-p1) -  r1 - r2


