import numpy as np
from tqdm import tqdm

#assumes constant radius
def check_for_overlap(frames,radii):
    overlaps = []
    for frame in tqdm(frames):
        cum = 0
        for i in range(len(frame)):
            mol1 = frame[i]
            rad1 = radii[i]
            for j in range(i+1,len(frame)):
                mol2 = frame[j]
                rad2 = radii[j]
                current_dist = np.linalg.norm(mol2-mol1) - rad1 - rad2
                if current_dist < 0:
                    cum += 1
        avg = cum
        overlaps.append(avg)

    return overlaps

#assumens constant radius
def avg_dist(frames,radii):
    dist = []
    for frame in tqdm(frames):
        cum = 0
        #ctr = 0
        for i in range(len(frame)):
            mol1 = frame[i]
            rad1 = radii[i]
            for j in range(i+1,len(frame)):
                mol2 = frame[j]
                rad2 = radii[j]
                current_dist = np.linalg.norm(mol2-mol1) - rad1 - rad2
                cum += current_dist
                #ctr += 1
        #print(ctr,((len(frame)) * (len(frame)-1))/2)
        avg = cum / (((len(frame)) * (len(frame)-1))/2) #(N*(N-1)/2)
        dist.append(avg)
    return dist

#assumens constant radius
#avg distance between k molecules (and confidence interval)
def avg_k_dist(frames,radii,k,upper=0.75,lower=0.25):
    dist = []
    cum_low = []
    cum_upper = []
    for frame in tqdm(frames):
        cum = 0
        cum_stats = []
        for i in range(len(frame)):
            mol1 = frame[i]
            rad1 = radii[i]
            i_dist_list = []
            for j in range(len(frame)):
                if (j != i):
                    mol2 = frame[j]
                    rad2 = radii[j]
                    current_dist = np.linalg.norm(mol2-mol1) - rad1 - rad2
                    i_dist_list.append(current_dist)
            i_dist_list = sorted(i_dist_list)
            _sum_cum_val = sum(i_dist_list[:k])
            cum += _sum_cum_val
            cum_stats.append(_sum_cum_val)
        
        cum_stats = sorted(cum_stats)
        low_val = int(len(frame)*lower)
        up_val = int(len(frame)*upper)
        #print(cum_stats[low_val],cum_stats[up_val])
        cum_low.append(cum_stats[low_val])
        cum_upper.append(cum_stats[up_val])
        avg = cum / (len(frame))
        dist.append(avg)
    return dist, cum_low, cum_upper

#acceptence/rejectance rate
def mc_rate(acc_list):
    res_accs = []
    res_rejs = []

    acc_cum = 0
    rej_cum = 0
    lenght = len(acc_list)

    for i in acc_list:
        if i:
            acc_cum += 1
        else:
            rej_cum += 1


        res_accs.append(acc_cum / lenght)
        res_rejs.append(rej_cum / lenght)

    return res_accs, res_rejs