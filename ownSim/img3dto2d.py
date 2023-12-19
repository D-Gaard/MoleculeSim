import numpy as np
import matplotlib.pyplot as plt
#import molecules as mc

def plot_sphere(img, x, y, radius):
    for i in range(max(0, int(x - radius)), min(img.shape[1], int(x + radius + 1))):
        for j in range(max(0, int(y - radius)), min(img.shape[0], int(y + radius + 1))):
            if (i - x)**2 + (j - y)**2 <= radius**2:
                img[i, j] = 1

def gaussian(x, y, x0, y0, sigma):
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def plot_sphere_gauss(img, x, y, radius,sigma):
    for i in range(max(0, int(x - radius)), min(img.shape[1], int(x + radius + 1))):
        for j in range(max(0, int(y - radius)), min(img.shape[0], int(y + radius + 1))):
            if (i - x)**2 + (j - y)**2 <= radius**2:
                img[i, j] += gaussian(i, j, x, y, sigma)


def get_points2d(universe):
  mols = universe.molecules
  pos = [m.pos for m in mols]
  pos2d = np.array(pos)[:,:2]
  radii = [m.radius for m in mols]

  return list(zip(pos2d,radii))

#NOTE: converts x/y and radius to int to index
#p2dr = list of ((x,y),r)
def get_3d_to_2d_img(p2dr,img_size):
  img = np.zeros(img_size)

  for _,((x,y),r) in enumerate(p2dr):
    plot_sphere(img,x,y,r)

  return img

#NOTE: converts x/y and radius to int to index
#p2dr = list of ((x,y),r)
def get_3d_to_2d_img_gauss(p2dr,img_size,sigma_scale):
  img = np.zeros(img_size)

  for _,((x,y),r) in enumerate(p2dr):
    plot_sphere_gauss(img,x,y,r,r/sigma_scale)
  
  return img

#quick way to plot the ortographic projection in the z direction
def plot_universe_in_2d(universe,sigma_scale=3):
  pd2r = get_points2d(universe)
  img = get_3d_to_2d_img(pd2r,universe.box_size[:2])
  img_gauss = get_3d_to_2d_img_gauss(pd2r,universe.box_size[:2],sigma_scale)

  plt.figure(figsize=(8,8))
  plt.subplot(1,2,1)
  plt.title("Ortographic projection in z (circle)")
  plt.imshow(img,cmap="gray")
  plt.subplot(1,2,2)
  plt.title("Ortographic projection in z (Gauss)")
  plt.imshow(img_gauss,cmap="gray")
  plt.show()
