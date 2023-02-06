


import collections
import re
import time
from pathlib import Path

import bigfish
import bigfish.stack as stack
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tifffile
from bigfish import detection
from bigfish.detection import (build_reference_spot, compute_snr_spots,
                               decompose_dense, detect_spots, fit_subpixel,
                               get_dense_region, modelize_spot, precompute_erf,
                               simulate_gaussian_mixture)
from bigfish.detection.dense_decomposition import (_filter_connected_region,
                                                   _get_connected_region)
from bigfish.detection.spot_detection import (_get_candidate_thresholds,
                                              spots_thresholding)
from matplotlib.patches import RegularPolygon
# your code
from mpl_toolkits import mplot3d
from pycpd import AffineRegistration, RigidRegistration
from scipy import ndimage
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from skimage.exposure import rescale_intensity
from skimage.measure import label, regionprops
#from matplotlib import animation
#from IPython.display import HTML
from tqdm import tqdm


plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rc_context({"axes.labelsize" : 45,})


def closest_point_index(points, target):
    #print(f"index  {index_axes}")
    closest_point = min(points, key=lambda p: distance.euclidean(p, target)) #(p[0] - tz) ** 2 + (p[1] - tx) ** 2 + (p[2] - ty) ** 2)
    index = [i for i, x in enumerate(points) if np.sum(closest_point == x) == 3]
    min_dist = distance.euclidean(closest_point, target)
    min_dist_0 = np.abs(closest_point[0] -  target[0])
    min_dist_1 = np.abs(closest_point[1] -  target[1])
    min_dist_2 = np.abs(closest_point[2] -  target[2])
    if len(index) > 1:
        #print(index)
        if points[index[0]].all() ==points[index[1]].all():
            raise(Exception("coordinate not unique"))
            #print("coordinate not unique")
        else:
            raise(Exception("multiple equal neirest neigbor"))
    return index[0], closest_point, min_dist, [min_dist_0, min_dist_1,  min_dist_2]



def pairit(sp0_ref, sp1, max_distance = None):
    nb_count = 0
    list_couple_index_sp0_ref = []
    list_couple_index_sp1 = []

    list_distance = []
    list_distance_coord = []
    sp0_ref_pair_order = []
    sp1_pair_order = []
    print('in_pair_it')
    for index_sp0 in tqdm(range(len(sp0_ref))):
        index_nn_sp1_for_sp0, closest_point_sp1_0, min_distsp1_0, min_list = closest_point_index(
            points=sp1,
            target = sp0_ref[index_sp0] )  # compute the distance to the neirest neibo
        index_nn_sp0_for_sp1, closest_point_sp0_1, min_distsp0_1, min_list = closest_point_index(
                                    points = sp0_ref,
                                    target = sp1[index_nn_sp1_for_sp0])
        if index_sp0 == index_nn_sp0_for_sp1 and min_distsp0_1 < max_distance:  # check that mutual neirest neigbhbor plus infirior to a dist threshol
            nb_count += 1
            list_couple_index_sp0_ref.append(index_sp0)
            list_couple_index_sp1.append(index_nn_sp1_for_sp0)

            list_distance.append(min_distsp0_1)
            list_distance_coord.append(min_list)
            sp0_ref_pair_order.append(closest_point_sp0_1)
            sp1_pair_order.append(closest_point_sp1_0)
    print(nb_count)
    return list_couple_index_sp0_ref, list_couple_index_sp1, sp0_ref_pair_order,  sp1_pair_order, list_distance, list_distance_coord


def compute_nn_couple(sp0_ref, sp1, R_0_1, t_0_1, s_0_1,
                    transform_method =  'rigid',
                    mean_substraction = True,
                    max_distance = 10,
                      scale_z_x_y = np.array([0.300, 0.103, 0.103])):
    """

    Args:
        sp0_ref (array): spots reference
        sp1 (array): spots that are rotate translate
        R_0_1:
        t_0_1:
        s_0_1:
        transform_method:
        mean_substraction:
        max_distance:
        scale_z_x_y: array
        index_axes:

    Returns:

    """
    assert transform_method in ["affine_cpd", "rigid_cpd", 'rigid', "rigid_inverse" "all"]
    sp0_ref = np.array(sp0_ref)
    sp1 = np.array(sp1)


    if transform_method == "affine_cpd":
        reg = AffineRegistration(**{'X': sp0_ref, 'Y': sp1})
        reg.B = R_0_1
        reg.t = t_0_1
        sp1 = np.array(sp1)
        sp1  = reg.transform_point_cloud(sp1)
        print("affine_cpd")
    else:
        raise(Exception(f' {transform_method} is not implemented'))
    sp0_ref = sp0_ref * scale_z_x_y
    sp1 = sp1 * scale_z_x_y
    if not mean_substraction:
        list_couple_index_sp0_ref, list_couple_index_sp1, sp0_ref_pair_order, sp1_pair_order, list_distance, list_distance_coord = pairit(sp0_ref, sp1,
                                                                                                     return_pairs=False,
                                                                                                     max_distance = max_distance)
        return list_couple_index, list_couple_translated_spots, list_distance

    else:
        assert mean_substraction
        sp0_ref_pair_order, sp1_pair_order = pairit(sp0_ref, sp1, return_pairs=True, max_distance = max_distance)
        mean_raw = np.mean(sp1_pair_order - sp0_ref_pair_order, axis=0)
        sp1 = sp1 - mean_raw
        print(f"mean raw removal{mean_raw}")
        list_couple_index, list_couple_translated_spots, list_distance, list_distance_coord = pairit(sp0_ref, sp1, return_pairs=False)
        return list_couple_index, list_couple_translated_spots, list_distance, mean_raw, list_distance_coord




if __name__ == "__main__":

    dico_bead_detection = np.load("/media/tom/T7/2023-01-19-PAPER-20-rounds/test/dico_bead_detection.npy",
            allow_pickle=True).item()
    dico_matrix_transform = np.load("/media/tom/T7/2023-01-19-PAPER-20-rounds/test/dico_matrix_transform.npy",
            allow_pickle=True).item()
    final_spots_detection = np.load("/media/tom/T7/2023-01-19-PAPER-20-rounds/test/final_spots_detection.npy",
            allow_pickle=True).item()


    scale_z_x_y = np.array([0.300, 0.103, 0.103])
    ref_round = "r1"
    dico_spots_detection = final_spots_detection
    dico_matrix_transform = dico_matrix_transform
    mean_substraction = True
    plot_hist = True


    path_folder_image = "/media/tom/T7/2023-01-19-PAPER-20-rounds/test/histogram"
    Path(path_folder_image).mkdir(parents=True, exist_ok=True)
    dico_matched_rna = {}


    for round in final_spots_detection:
        dico_matched_rna[round] = {}
        for pos in final_spots_detection[round]:
            round = "r2"
            pos = 'pos1'
            dico_matched_rna[round][pos] = {}
            sp0_ref = final_spots_detection[ref_round][pos]['subpixel_spots']
            sp1 = final_spots_detection[round][pos]['subpixel_spots']
            R_1_2 = dico_matrix_transform[ref_round][round][pos]['R']
            t_1_2 = dico_matrix_transform[ref_round][round][pos]['t']
            s_1_2 = dico_matrix_transform[ref_round][round][pos]['s']
            transform_method = dico_matrix_transform[ref_round][round][pos]['transform_method']


            if transform_method == "affine_cpd":
                reg = AffineRegistration(**{'X': sp0_ref, 'Y': sp1})
                reg.B = R_1_2
                reg.t = t_1_2
                sp0_ref = np.array(sp0_ref)
                sp1 = reg.transform_point_cloud(sp1)
                sp0_ref = sp0_ref * scale_z_x_y
                sp1 = sp1 * scale_z_x_y
            else:
                raise (Exception(f' {transform_method} is not implemented yet'))

            list_couple_index_sp0_ref, list_couple_index_sp1, sp0_ref_pair_order,  sp1_pair_order, list_distance, list_distance_coord = pairit(
                            sp0_ref=sp0_ref,
                            sp1=sp1,
                            max_distance=1)
            print(np.median(list_distance))
            if mean_substraction:
                mean_raw = np.mean(np.array(sp1_pair_order) - np.array(sp0_ref_pair_order), axis=0)
                sp1 = sp1 - mean_raw
                list_couple_index_sp0_ref, list_couple_index_sp1, sp0_ref_pair_order, sp1_pair_order, list_distance, list_distance_coord = pairit(
                    sp0_ref=sp0_ref,
                    sp1=sp1,
                    max_distance=1)
            print(np.median(list_distance))

            if plot_hist:
                fig, ax = plt.subplots(figsize=(16, 10))    # plotting density plot for carat using distplot()
                fig.suptitle(f'{ref_round}_{round}_{pos}_median_{np.median(list_distance)}', fontsize=20, x =0.1, ha='left')

                sns.kdeplot(x=list_distance, cumulative=False)
                ax.set_ylim(ymin=0)
                ax.set_xlim(left = 0, right= 1)
                fig.savefig(path_folder_image + "/" + str(beads_signal_path)[-12:-8])
                print(np.mean(list_distance))

    list_couple_index_sp0_ref, list_couple_index_sp1, sp0_ref_pair_order, sp1_pair_order, list_distance, list_distance_coord

    dico_matrix_transform[source_folder][target_folder][pos0]['spots_0_colocalized'] = spots_0_colocalized
    dico_matrix_transform[source_folder][target_folder][pos0]['new_spots_1_colocalized'] = new_spots_1_colocalized





