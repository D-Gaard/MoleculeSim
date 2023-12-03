import numpy as np
import random
import scipy.constants as consts
from molecules import dist


#BETA = 0.38 # std of diameter distribution of casein micelles

#Van der Wall constants:
HAMAKER = 1.0 # Estimate of Hamaker constant for casein micelles, 

#Electri repulsion constants:
EPS0 = consts.epsilon_0#electric permittivity of vacuum
EPS =  72#dielectric constant of the solvent, estimated to be in range [70,75] so far based on papers
PHII = 8 #electrostaticpotential at the surface of particle i, generalized to be 8 based on papers
#KAPPA = #inverse Debye length
TEMPERATURE = 309.15 #kelvin = 36C
I = 0.08 #Ionic strength, defined as (M,mol L)

#steric repulstion constats:
SIGMA = 0.006 #grafting density
H = 7 #width of the polyelectrolyte brush in nm

def kappa():
  nom = consts.epsilon_0 * EPS * consts.R * TEMPERATURE
  denom = consts.physical_constants["Faraday constant"][0]**2 * I
  return np.sqrt(nom/denom)

def elec_repv2(m1,m2):
  h = dist(m1,m2)-m1.radius-m2.radius
  left = -2*np.pi*m1.radius*EPS0*EPS*(PHII**2)
  right1 = kappa()**(-1)
  right2 = np.exp(kappa()**(-1) * h)
  right3 = np.log(1+np.exp(kappa()**(-1) * h))
  print("left: ", left, ", kappa inv: ", right1, "h:", h, "np exp: ", right2, ", numpy log: ", right3)
  return -2*np.pi*m1.radius*EPS0*EPS*(PHII**2) * np.log(1+np.exp(kappa()**(-1) * h))

#Van der Waal (w.r. to m1)
def vdw(m1,m2):
  _2a1a2 = 2*m1.radius*m2.radius
  r2 = dist(m1,m2)**2
  square_p = (m1.radius + m2.radius)**2
  square_n = (m1.radius - m2.radius)**2

  fst_term = _2a1a2/(r2 - square_p)
  snd_term = _2a1a2/(r2 - square_n)
  thrd_term = np.log((r2 - square_p)/(r2 - square_n))
  return (-1/6) * HAMAKER*(fst_term + snd_term + thrd_term)

#electrostatic repulsion (w.r. to m1)
def elec_rep(m1,m2):
  BU_el = EPS0 * EPS * (PHII**2) * np.log(1 + np.exp(dist(m1,m2)-m1.radius-m2.radius))
  #print(np.sqrt(2/(EPS0*EPS*np.log(1 + np.exp(dist(m1,m2)-m1.radius-m2.radius)))))
  return BU_el


#steric (w.r. to m1)
def steric(m1,m2):
  a_eff = ((1/m1.radius) + (1/m2.radius)) ** (-1)
  fst_nominator = 16 * np.pi * a_eff * H
  fst_denominator = 35 * SIGMA
  y= ((dist(m1,m2)-m1.radius-m2.radius) / (2*H))


  snd = 28 * (y**(-1/4) - 1)
  trd = (20/11) * (1 - y**(11/4))
  fth = 12 * (y - 1)

  BU_st = (fst_nominator / fst_denominator) * (snd + trd + fth)
  return BU_st #*10**(-4.6)



#test force for movment of system
def dummy_force(m1,m2):
  return 3 if dist(m1,m2) else 0


#calculate if move is accepted based on forces
def accept_move(ePrev,eNew,Beta):
  proba = min(1, np.exp(-Beta*(ePrev-eNew)))
  return random.random() < proba