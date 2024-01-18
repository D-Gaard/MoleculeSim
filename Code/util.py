import numpy as np
from scipy import integrate, optimize


d0 = 200 # mean diameter
beta = 0.38 #std of distribution 

# PDF over diameter of casein micelle
def pdf(d):
  return (1/(np.sqrt(2*np.pi)*beta*d)) * np.exp(-(np.log(d/d0)**2)/(2*beta**2))


def sample_radius():
  tg_val = np.random.rand() # get random unifrom [0,1] value
  eq = lambda x: integrate.quad(pdf, -0.0000000000001, x)[0] - tg_val
  solution = optimize.fsolve(eq, 200) # 200 = mean

  return solution[0] / 2 #solution is diameter, we want radius

def get_n_radii(N = 1, seed = 1414):
  np.random.seed(seed)
  return np.array([sample_radius() for _ in range(N)])
