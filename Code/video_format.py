import cv2
import os
import numpy as np

#folder consts (set these according to own paths)
DATA_FOLDER = r"C:\Users\DGaard\Desktop\Gits\MoleculeSim\Data\\"
VIDEO_NAME = "cheese_gel.avi"
VIDEO_PATH = os.path.join(os.path.dirname(DATA_FOLDER), VIDEO_NAME)
FRAME_FOLDER = r"C:\Users\DGaard\Desktop\Gits\MoleculeSim\Data\frames\\"
FRAME_PATH = os.path.dirname(FRAME_FOLDER)


#produces the number of frames from the video
def get_num_frames(video_path=VIDEO_PATH):
  vidcap = cv2.VideoCapture(video_path)
  _,_ = vidcap.read() 

  if not vidcap.isOpened():
    print("ERROR: video could not open")
    return -1

  total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
  vidcap.release()

  return total_frames


# yields given frame from video
def get_frame(frame_number, video_path=VIDEO_PATH):
  vidcap = cv2.VideoCapture(video_path)
  _,_ = vidcap.read()

  if not vidcap.isOpened():
    print("ERROR: video could not open")
    return -1

  total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

  if frame_number > total_frames or frame_number < 0:
    print("ERROR: frame number outside scope of video")
    return -1
  
  # set frame used, used with 0 indexing
  vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
  #read frame
  ret, frame = vidcap.read()
  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  vidcap.release()

  if not ret:
    print("ERROR: could not read frame from video")
    return -1

  return frame

#saves a specific frame from the video
def save_img(frame_number,video_path=VIDEO_PATH,save_path=FRAME_PATH):
  vidcap = cv2.VideoCapture(video_path)
  _,_ = vidcap.read()

  if not vidcap.isOpened():
    print("ERROR: video could not open")
    return -1

  total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

  if frame_number > total_frames or frame_number < 0:
    print("ERROR: frame number outside scope of video")
    return -1
  
  # set frame used, used with 0 indexing
  vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
  _,image = vidcap.read()
  image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


  #Save image as JPEG
  frame_name = f"frame{frame_number}.jpg"
  cv2.imwrite(os.path.join(save_path, frame_name), image)

  vidcap.release()

  return 0

#loads a list of wanted frames
def get_frames(frames):
  return [get_frame(f_num) for f_num in frames]

#saves a list of wanted frames, and returns list of 0's if succesfull
def save_imgs(frames):
  return [save_img(f_num) for f_num in frames]



###################################################
# CODE BELOW IS TAKEN FROM BACHELOR PAPER (https://github.com/cappertherapper/Bachelorprojekt/blob/main)

from skimage import io, morphology, filters
from skimage.color import rgb2gray, rgba2rgb
from skimage.morphology import label
from skimage.segmentation import clear_border
import numpy as np
import cv2


def biasField(I, mask):
    (rows,cols) = I.shape
    r, c = np.meshgrid(list(range(rows)), list(range(cols)))
    rMsk = r[mask].flatten()
    cMsk = c[mask].flatten()
    VanderMondeMsk = np.array([rMsk*0+1, rMsk, cMsk, rMsk**2, rMsk*cMsk, cMsk**2]).T
    ValsMsk = I[mask].flatten()
    coeff, residuals, rank, singularValues = np.linalg.lstsq(VanderMondeMsk, ValsMsk, rcond=-1)
    VanderMonde = np.array([r*0+1, r, c, r**2, r*c, c**2]).T
    J = np.dot(VanderMonde, coeff) # @ operator is a python 3.5 feature!
    J = J.reshape((rows,cols)).T
    return(J)


def resize(arr, size=None, corner=None):
    """Resizes array or ensures that dimensions are the same"""
    start, end = None, None
    min_size = min(arr.shape[0], arr.shape[1])
    if not size:
        start = 0
        end = min_size
    elif not corner:
        start = (min_size - size) // 2
        end = start + size
    else:
        start = corner
        end = corner + size
    return arr[start:end, start:end]


def preprocess(image, threshold, smooth=1, clear_borders=False):
    imFilt = filters.gaussian(image, smooth)

    imThresh = imFilt > threshold

    B = biasField(imFilt, imThresh)
    imBias = imFilt - B + B.mean()

    imBiasThresh = imBias > threshold

    if clear_borders:
        imBiasThresh = clear_border(imBiasThresh) # if components touch border, remove them

    labels = label(imBiasThresh) #get connected components of image
    return labels


def process_image(path, threshold, smooth=1, size=None):
    im = io.imread(path)
    if im.ndim == 2: 
        pass
    elif im.shape[2] == 3:
        im = rgb2gray(im) #grey scale image
    else:
        im = rgb2gray(rgba2rgb(im))
    im = resize(im)
    return preprocess(im, threshold, smooth=smooth) #apply gaussian smoothing and bias + misc to img


def get_video(path, skip_size=1):
    video = cv2.VideoCapture(path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % skip_size == 0:
            im = rgb2gray(frame)
            im = resize(im)
            frames.append(im)
        frame_count += 1
        
    video.release()

    frames_array = np.array(frames)
    return frames_array


def process_video(path, threshold=None, smooth=1, skip_size=1, size=None):
    video = cv2.VideoCapture(path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % skip_size == 0:
            im = rgb2gray(frame)
            im = resize(im, size)
            im = preprocess(im, threshold, smooth=smooth)
            frames.append(im)
        frame_count += 1
    
    video.release()

    frames_array = np.array(frames)
    return frames_array




###########################################################################

# CODE BELOW IS TAKEN FROM BACHELOR PAPER (https://github.com/cappertherapper/Bachelorprojekt/blob/main)
from scipy import ndimage
from skimage.measure import regionprops

def analyse_video(video, L=200):
    """Generates lists of F and G from a video"""
    f_list = []
    g_list = []
    # for p in tqdm(range(len(video))):
    for label_image in video:
        M = label_image.shape[0] - L 

        label_image_bounded = np.zeros(label_image.shape, dtype=int)
        cluster_num = 1
        for region in regionprops(label_image): # get all regions of image
            minr, minc, maxr, maxc = region.bbox  # bounding box of region
            if (minr > L+1) and (minc > L+1) and (maxr < M-1) and (maxc < M-1): #if region is within the area we consider
                pixel_coordinates = region.coords
                label_image_bounded[pixel_coordinates[:,0],  pixel_coordinates[:,1]] =  cluster_num
                cluster_num += 1

        curr_image = label_image_bounded

        f = np.zeros(L+1)
        g = np.zeros(L+1)
        for cluster in range(1, curr_image.max()+1):
            ref_mask = curr_image == cluster    #get mask for cluster
            rem_mask = np.logical_and(label_image,np.invert(ref_mask)) # get mask for everything but the current cluster
            
            D = ndimage.distance_transform_edt(ref_mask==0) # gives distance to current cluster
            for i in range(0,L+1): #border pixels cannot look at pixels further than this distance away
                dist_mask = D <= i
                f[i] += np.count_nonzero(np.logical_and(dist_mask, rem_mask)) #number of cluster pixels in radius of cluster, not including itself
                g[i] += np.count_nonzero(dist_mask) #total number of pixels in radius

        f = f / curr_image.max() #fraction f and g contribute compared to all clusters
        g = g / curr_image.max()

        f_list.append(f)
        g_list.append(g)

    return f_list, g_list



import numpy as np
# import imageio
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.color import label2rgb
from IPython.display import display, clear_output


def visualize(video, F_list, G_list, F_noise=np.array([]), G_noise=np.array([]), L=100, save=False,frame_name = "-1"):
    """Visualizes a list of data overlaps, can compare with one dataset"""
    fig, ax = plt.subplots(1,4, figsize=(14,6), gridspec_kw={'width_ratios': [1, 1, 1, 2]})
    plt.tight_layout()
    plt.title('Cheese gel video frames')
    colors = cm.Blues(np.linspace(0.2, 1, len(video)))

    # a = int(168 / len(video))

    for i in range(len(F_list)):
        M = video[i].shape[0] - L
        bx = (L, M, M, L, L)
        by = (L, L, M, M, L)

        ax[3].plot(bx, by, '-r')
        ax[3].imshow(label2rgb(video[i], bg_label=0))
        # ax[3].set_title('frame nr.{}'.format(a * (i+1)))
        if frame_name == "-1":
            ax[3].set_title('frame nr.{}'.format(i+1))
        else:
            ax[3].set_title(f'frame nr. {frame_name}')

        ax[0].set_title('Area overlap')
        ax[0].plot(F_list[i], color=colors[i])
        ax[0].set_xlabel('r')
        ax[0].set_ylabel('area overlap')

        ax[1].set_title('Fractional area overlap')
        ax[1].plot(F_list[i]/G_list[i], color=colors[i])
        ax[1].set_xlabel('r')
        ax[1].set_ylabel('fractional area overlap')

        ax[2].set_title('Curve overlap')
        ax[2].plot((F_list[i])[1:]-(F_list[i])[:-1], color=colors[i])
        ax[2].set_xlabel('r')
        ax[2].set_ylabel('curve overlap')

        if save:
            # filename = 'frames/subplot_{:03d}.png'.format(a * (i+1))
            filename = 'frames/subplot_{:03d}.png'.format(i+1)
            fig.savefig(filename)
            # plots.append(fig)

        display(fig)
        clear_output(wait = True)

    if F_noise.size and G_noise.size:
        ax[0].plot(F_noise, color='r')
        ax[1].plot(F_noise/G_noise, color='r')
        ax[2].plot(F_noise[1:]-F_noise[:-1], color='r')

    if save:
        # filename = 'frames/subplot_{:03d}.png'.format(a * (i+1))
        filename = 'frames/subplot_{:03d}.png'.format(i+1)
        fig.savefig(filename)
        # plots.append(fig)

        # with imageio.get_writer('animation.gif', mode='I') as writer:
        #     for plot in plots:
        #         writer.append_data(imageio.core.image_as_uint(plot.canvas.renderer.buffer_rgba()))
    plt.show()

