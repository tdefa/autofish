

import math
import numpy as np
import matplotlib.pyplot as plt
#from icp import icp
from pycpd import AffineRegistration, RigidRegistration


if __name__ == '__main__':
    # set seed for reproducible results
    np.random.seed(12345)

    # create a set of points to be the reference for ICP
    xs = np.random.random_sample((50, 1))
    ys = np.random.random_sample((50, 1))
    reference_points = np.hstack((xs, ys))

    # transform the set of reference points to create a new set of
    # points for testing the ICP implementation

    # 1. remove some points
    points_to_be_aligned = reference_points[1:47]

    # 2. apply rotation to the new point set
    theta = math.radians(12)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s],
                    [s, c]])
    points_to_be_aligned = np.dot(points_to_be_aligned, rot)

    # 3. apply translation to the new point set
    points_to_be_aligned += np.array([np.random.random_sample(), np.random.random_sample()])

    # run icp
    transformation_history, aligned_points = icp(reference_points, points_to_be_aligned, verbose=True)

    # show results
    plt.plot(reference_points[:, 0], reference_points[:, 1], 'rx', label='reference points')
    plt.plot(points_to_be_aligned[:, 0], points_to_be_aligned[:, 1], 'b1', label='points to be aligned')
    plt.plot(aligned_points[:, 0], aligned_points[:, 1], 'g+', label='aligned points')
    plt.legend()
    plt.show()


# detection on mask image

import numpy as np
import matplotlib.pyplot as plt
import tifffile

# mask the image with 2D mask for all the round and all the position

path_mask = "/media/tom/T7/2023-01-19-PAPER-20-rounds/round_impair/r1/cyto_mask2D_3dim/r1_pos0_ch0.tif"


path_to_fish_r1 = "/media/tom/T7/2023-01-19-PAPER-20-rounds/round_impair/r1/r1_pos0_ch0.tif"
path_to_fish_r15 = "/media/tom/T7/2023-01-19-PAPER-20-rounds/round_impair/r15/r15_pos0_ch0.tif"




dico_matrix_transform =  np.load("/media/tom/T7/2023-01-19-PAPER-20-rounds/test_folder/test3/dico_matrix_transform(800, 600, 600).npy",
                                 allow_pickle=True).item()

dico_bead = np.load('/media/tom/T7/2023-01-19-PAPER-20-rounds/test_folder/test3/dico_bead_detection(800, 600, 600).npy',
                    allow_pickle=True).item()


# load mask non nul coordiante for each round and each position

sp_r1  = dico_bead['r1']["pos0"]['subpixel_spots']


mask_2D = tifffile.imread(path_mask)

fish_3D_r1 = tifffile.imread(path_to_fish_r1)
fish_3D_r15 = tifffile.imread(path_to_fish_r15)
#get coordiante of the mask


coord_mask_ref = np.where(mask_2D > 0)
coord_mask_ref = np.array([coord for coord in zip(coord_mask_ref[0], coord_mask_ref[1],  coord_mask_ref[2])])

ref_round = "r1"
round_to_transfom = "r15"
pos = "pos0"
R_1_2 = dico_matrix_transform[round_to_transfom][ref_round][pos]['R']
t_1_2 = dico_matrix_transform[round_to_transfom][ref_round][pos]['t']



def apply_rigid_transform(sp, R, t):
    return ((R @ sp.T) + t).T



coord_mask_target = apply_rigid_transform(coord_mask_ref, R_1_2, t_1_2)
coord_mask_target.round()
coord_mask_target = coord_mask_target.astype(int)

## generate a mask from coordiante

mask_2D_target = np.zeros(mask_2D.shape)
for coord_index in range(len(coord_mask_target)):
    coord = coord_mask_target[coord_index]
    if coord[1] < mask_2D.shape[1] and coord[2] < mask_2D.shape[2] and coord[1] >= 0 and coord[2] >= 0:
        original_coord = coord_mask_ref[coord_index]
        mask_2D_target[0, coord[1], coord[2]] = mask_2D[0, original_coord[1], original_coord[2]]


np.save("/media/tom/T7/2023-01-19-PAPER-20-rounds/test_folder/test3/mask_2D_target", mask_2D_target)

np.save("/media/tom/T7/2023-01-19-PAPER-20-rounds/test_folder/test3/mask_2D", mask_2D)

## mask fish signal

mask_fish_r15 = (fish_3D_r15 * (mask_2D_target >0).astype(float))

np.save("/media/tom/T7/2023-01-19-PAPER-20-rounds/test_folder/test3/mask_fish_r15", mask_fish_r15)

#%%

from bigfish.detection import build_reference_spot, compute_snr_spots
from bigfish.detection.spot_detection import (_get_candidate_thresholds,
                                              spots_thresholding)
from tqdm import tqdm
import re
from pathlib import Path
from bigfish.detection import build_reference_spot, compute_snr_spots
import bigfish.stack as stack
import numpy as np
import pylab
import tifffile
from bigfish import detection, stack
#from bigfish.detection import fit_subpixel
from skimage.measure import label, regionprops
import time
import scipy
sigma = 1.35

min_distance = [3,3,3]

rna_log = stack.log_filter(fish_3D_r15, sigma)
# local maximum detection

np.save("/media/tom/T7/2023-01-19-PAPER-20-rounds/test_folder/test3/rna_log", rna_log)

mask_fish_r15_min = scipy.ndimage.minimum_filter(mask_fish_r15, size=[1, 7, 7])

mask = detection.local_maximum_detection(rna_log, min_distance=min_distance)

rna_log_masked  = rna_log * (mask_fish_r15_min >0).astype(float)

np.save("/media/tom/T7/2023-01-19-PAPER-20-rounds/test_folder/test3/rna_log_masked", rna_log_masked)


plt.imshow(np.amax(rna_log, axis=0))
plt.show()

threshold = detection.automated_threshold_setting(rna_log_masked, mask)
spots, _ = detection.spots_thresholding(rna_log_masked, mask, int(threshold))

np.save("/media/tom/T7/2023-01-19-PAPER-20-rounds/test_folder/test3/spots_r15", spots)


## plot
import numpy as np
mask_fish_r15 = np.load("/media/tom/T7/2023-01-19-PAPER-20-rounds/test_folder/test3/mask_fish_r15.npy")
spots = np.load("/media/tom/T7/2023-01-19-PAPER-20-rounds/test_folder/test3/spots_r15.npy")
import napari
viewer = napari.view_image(mask_fish_r15, colormap='magma')


points_layer = viewer.add_points(spots, size=10)
### perfrom match

dico_matched_rna[round][pos] = {}
sp0_ref = final_spots_detection[ref_round][pos]['subpixel_spots'][0]
sp1 = final_spots_detection[round][pos]['subpixel_spots'][0]
R_1_2 = dico_matrix_transform[ref_round][round][pos]['R']
t_1_2 = dico_matrix_transform[ref_round][round][pos]['t']
s_1_2 = dico_matrix_transform[ref_round][round][pos]['s']
transform_method = dico_matrix_transform[ref_round][round][pos]['transform_method']
if t_1_2 is None:
    continue

if transform_method == "affine_cpd":
    reg = AffineRegistration(**{'X': sp0_ref, 'Y': sp1})
    reg.B = R_1_2
    reg.t = t_1_2
    sp0_ref = np.array(sp0_ref)
    sp1 = np.array(sp1)
    sp1 = reg.transform_point_cloud(sp1)
    sp0_ref = sp0_ref * scale_z_x_y
    sp1 = sp1 * scale_z_x_y



## cyto_seg
image_name = "r2_pos1_ch0.tif"
r = tifffile.imread("/media/tom/T7/2023-01-19-PAPER-20-rounds/round_pair/r2/cyto_seg/" + image_name)
tifffile.imsave("/media/tom/T7/2023-01-19-PAPER-20-rounds/round_pair/r2/cyto_seg2D_3dim/" + image_name, np.amax(r, axis=0)[None,:,:])


##########


import numpy as np
dd = '/media/tom/Transcend/autofish_test/dico_spots_registered.npy'
ref_round = 'r1'
dico_spots = np.load(dd, allow_pickle=True).item()

for round in dico_spots:
    for pos in dico_spots[round]:
        print(round, pos)
        if round == ref_round:
            continue
        sp0_ref = dico_spots[ref_round][pos]
        sp1 = dico_spots[round][pos]

        list_couple_index_sp0_ref, list_couple_index_sp1, sp0_ref_pair_order, sp1_pair_order, list_distance = pairit(
            sp0_ref=sp0_ref,
            sp1=sp1,
            max_distance=1000)

        print(f'medain {np.median(list_distance)}')
        print(f'medain {np.mean(list_distance)}')