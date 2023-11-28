import numpy as np
import random

BETA = 0.38 # std of diameter distribution of casein micelles
HAMAKER = 1.0 # Estimate of Hamaker constant for casein micelles, 

#Van der Waal (w.r. to m1)
def vdw(m1,m2):
  _2a1a2 = 2*m1.radius*m2.radius
  r2 = (m1.pos - m2.pos)**2
  square_p = (m1.radius + m2.radius)**2
  square_n = (m1.radius - m2.radius)**2

  fst_term = _2a1a2/(r2 - square_p)
  snd_term = _2a1a2/(r2 - square_n)
  thrd_term = np.log((r2 - square_p)/(r2 - square_n))
  return HAMAKER*(fst_term + snd_term + thrd_term)

#steric (w.r. to m1)
def steric(m1,m2):
  return 0

#electrostatic repulsion (w.r. to m1)
def elec_rep(m1,m2):
  return 0

#test force for movment of system
def dummy_force(m1,m2):
  return 3 if np.linalg.norm(m1.pos, m2.pos) else 0


#calculate if move is accepted based on forces
def accept_move(ePrev,eNew,Beta):
  proba = min(1, np.exp(-Beta*(ePrev-eNew)))
  return random.random() < proba