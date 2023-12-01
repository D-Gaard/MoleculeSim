import numpy as np
import random
import scipy.constants as consts
from molecules import dist


#BETA = 0.38 # std of diameter distribution of casein micelles

#Van der Wall constants:
HAMAKER = 1.0 # Estimate of Hamaker constant for casein micelles, 

#Electri repulsion constants:
EPS0 = consts.epsilon_0#electric permittivity of vacuum
#EPS =  #dielectric constant of the solvent 
#PHII = #electrostaticpotential at the surface of particle i
#KAPPA = #inverse Debye length

#steric repulstion constats:
SIGMA = 0.006 #grafting density
H = 0 #width of the polyelectrolyte brush

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
  BU_el = EPS0 * e * phi_1 * phi_2 * np.log(1 + np.exp(-a1-a2))
  U_el = BU_el
  return 0


#steric (w.r. to m1)
def steric(m1,m2):
  a_eff = ((1/m1.radius) + (1/m2.radius)) ** (-1)
  fst_nominator = 16 * np.pi * a_eff * H
  fst_denominator = 35 * SIGMA
  #y= dist(m1,m2)


  snd = 28 * (pow(y,-1/4) - 1)
  trd = 20/11 * (1 - pow(y,11/4))
  fth = 12 * (y - 1)

  BU_st = fst_nominator / fst_denominator + snd + trd + fth
  U_st = BU_st
  return 0



#test force for movment of system
def dummy_force(m1,m2):
  return 3 if dist(m1,m2) else 0


#calculate if move is accepted based on forces
def accept_move(ePrev,eNew,Beta):
  proba = min(1, np.exp(-Beta*(ePrev-eNew)))
  return random.random() < proba