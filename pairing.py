


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





def compute_pair(sp0_ref, sp1, max_distance = None):
    list_couple_index_sp0_ref = []
    list_couple_index_sp1 = []
    sp0_ref_pair_order = []
    sp1_pair_order = []
    list_distance = []

    print('in_pair_it')
    from sklearn.neighbors import NearestNeighbors
    nbrs_sp0 = NearestNeighbors(algorithm='ball_tree').fit(sp0_ref)
    neigh_dist_sp0_sp1, neigh_ind_sp0_sp1 = nbrs_sp0.radius_neighbors(X=sp1,
                                                                      radius=max_distance,
                                                                      return_distance=True,
                                                                      sort_results=True)
    nbrs_sp1 = NearestNeighbors(algorithm='ball_tree').fit(sp1)
    neigh_dist_sp1_sp0, neigh_ind_sp1_sp0 = nbrs_sp1.radius_neighbors(X=sp0_ref, radius=max_distance,
                                                                      return_distance=True, sort_results=True)

    for index_sp0 in range(len(sp0_ref)):
        if len(neigh_ind_sp1_sp0[index_sp0]) > 0:
            if neigh_ind_sp0_sp1[neigh_ind_sp1_sp0[index_sp0][0]][0] == index_sp0:
                list_couple_index_sp0_ref.append(index_sp0)
                list_couple_index_sp1.append(neigh_ind_sp1_sp0[index_sp0][0])
                sp0_ref_pair_order.append(sp0_ref[index_sp0])
                sp1_pair_order.append(sp1[neigh_ind_sp1_sp0[index_sp0][0]])
                list_distance.append(neigh_dist_sp1_sp0[index_sp0][0])


    print(np.median(list_distance))
    print(len(list_distance))
    return list_couple_index_sp0_ref, list_couple_index_sp1, sp0_ref_pair_order,  sp1_pair_order, list_distance

pairit = compute_pair

def compute_pair_folder(

        final_spots_detection,
        scale_z_x_y=np.array([270, 108, 108]),
        max_distance=1000,
        ref_round = "r1",
        mean_substraction = True,
        plot_hist = True,
        path_folder_save_plot = "/media/tom/T7/2023-01-19-PAPER-20-rounds/test3/histogram"
        ):


    Path(path_folder_save_plot).mkdir(parents=True, exist_ok=True)
    dico_matched_rna = {}

    for round in final_spots_detection:
        dico_matched_rna[round] = {}
        for pos in final_spots_detection[round]:
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
            elif transform_method == "rigid":
                from utils.transform import apply_rigid_transform
                sp0_ref = np.array(sp0_ref)

                sp1 = np.array(sp1)
                sp1 = apply_rigid_transform(sp1, R_1_2, t_1_2)
                sp0_ref = sp0_ref * scale_z_x_y
                sp1 = sp1 * scale_z_x_y
            else:
                raise Exception("transform_method not implemented")

            list_couple_index_sp0_ref, list_couple_index_sp1, sp0_ref_pair_order, sp1_pair_order, list_distance =  pairit(sp0_ref=sp0_ref,
                                                                                                                          sp1=sp1,
                                                                                                                          max_distance=max_distance)

            if mean_substraction:
                raise(Exception('mean_substraction not implemented yet'))

            sp0_ref_pair_order = np.array(sp0_ref_pair_order) / scale_z_x_y
            sp1_pair_order = np.array(sp1_pair_order) / scale_z_x_y

            dico_matched_rna[round][pos]["list_couple_index_sp0_ref"] = list_couple_index_sp0_ref
            dico_matched_rna[round][pos]["list_couple_index_sp1"] = list_couple_index_sp1
            dico_matched_rna[round][pos]["sp0_ref_pair_order"] = sp0_ref_pair_order
            dico_matched_rna[round][pos]["sp1_pair_order"] = sp1_pair_order
            dico_matched_rna[round][pos]["list_distance"] = list_distance
            dico_matched_rna[round][pos]["ref_round"] = ref_round


            if plot_hist:

                fig, ax = plt.subplots(figsize=(16, 10))    # plotting density plot for carat using distplot()
                fig.suptitle(f'{ref_round}_{round}_{pos}_median_{np.median(list_distance)}', fontsize=20, x =0.1, ha='left')
                sns.kdeplot(x=list_distance, cumulative=False)
                ax.set_ylim(ymin=0)
                ax.set_xlim(left = 0, right= max_distance)
                fig.savefig(path_folder_save_plot + "/" + f"{ref_round}_{round}_{pos}")
    return dico_matched_rna


def compute_pair_folder(
        dico_matrix_transform,
        final_spots_detection,
        scale_z_x_y=np.array([270, 108, 108]),
        max_distance=1000,
        ref_round = "r1",
        mean_substraction = True,
        plot_hist = True,
        path_folder_save_plot = "/media/tom/T7/2023-01-19-PAPER-20-rounds/test3/histogram"
        ):


    Path(path_folder_save_plot).mkdir(parents=True, exist_ok=True)
    dico_matched_rna = {}

    for round in final_spots_detection:
        dico_matched_rna[round] = {}
        for pos in final_spots_detection[round]:
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
            elif transform_method == "rigid":
                from utils.transform import apply_rigid_transform
                sp0_ref = np.array(sp0_ref)

                sp1 = np.array(sp1)
                sp1 = apply_rigid_transform(sp1, R_1_2, t_1_2)
                sp0_ref = sp0_ref * scale_z_x_y
                sp1 = sp1 * scale_z_x_y
            else:
                raise Exception("transform_method not implemented")

            list_couple_index_sp0_ref, list_couple_index_sp1, sp0_ref_pair_order, sp1_pair_order, list_distance =  pairit(sp0_ref=sp0_ref,
                                                                                                                          sp1=sp1,
                                                                                                                          max_distance=max_distance)

            if mean_substraction:
                raise(Exception('mean_substraction not implemented yet'))

            sp0_ref_pair_order = np.array(sp0_ref_pair_order) / scale_z_x_y
            sp1_pair_order = np.array(sp1_pair_order) / scale_z_x_y

            dico_matched_rna[round][pos]["list_couple_index_sp0_ref"] = list_couple_index_sp0_ref
            dico_matched_rna[round][pos]["list_couple_index_sp1"] = list_couple_index_sp1
            dico_matched_rna[round][pos]["sp0_ref_pair_order"] = sp0_ref_pair_order
            dico_matched_rna[round][pos]["sp1_pair_order"] = sp1_pair_order
            dico_matched_rna[round][pos]["list_distance"] = list_distance
            dico_matched_rna[round][pos]["ref_round"] = ref_round


            if plot_hist:

                fig, ax = plt.subplots(figsize=(16, 10))    # plotting density plot for carat using distplot()
                fig.suptitle(f'{ref_round}_{round}_{pos}_median_{np.median(list_distance)}', fontsize=20, x =0.1, ha='left')
                sns.kdeplot(x=list_distance, cumulative=False)
                ax.set_ylim(ymin=0)
                ax.set_xlim(left = 0, right= max_distance)
                fig.savefig(path_folder_save_plot + "/" + f"{ref_round}_{round}_{pos}")
    return dico_matched_rna


if __name__ == "__main__":


    dico_matrix_transform = np.load("/media/tom/T7/2023-01-19-PAPER-20-rounds/test3/dico_matrix_transform(800, 600, 600).npy",
            allow_pickle=True).item()
    final_spots_detection = np.load("/media/tom/T7/2023-01-19-PAPER-20-rounds/test_folder/test3/final_spots_detection(800, 600, 600).npy",
            allow_pickle=True).item()


    scale_z_x_y = np.array([270, 108, 108])
    ref_round = "r1"
    dico_spots_detection = final_spots_detection
    dico_matrix_transform = dico_matrix_transform
    mean_substraction = True
    plot_hist = True


    path_folder_image = "/media/tom/T7/2023-01-19-PAPER-20-rounds/test3/histogram"
    Path(path_folder_image).mkdir(parents=True, exist_ok=True)
    dico_matched_rna = {}

    for round in final_spots_detection:
        dico_matched_rna[round] = {}
        for pos in final_spots_detection[round]:
            pos = "pos0"
            round = "r15"
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



            else:
                raise (Exception(f' {transform_method} is not implemented yet'))
            max_distance = 1000

            list_couple_index_sp0_ref = []
            list_couple_index_sp1 = []
            list_distance = []
            sp0_ref_pair_order = []
            sp1_pair_order = []
            print('in_pair_it')
            from sklearn.neighbors import NearestNeighbors
            nbrs_sp0 = NearestNeighbors(algorithm='ball_tree').fit(sp0_ref)
            neigh_dist_sp0_sp1, neigh_ind_sp0_sp1 = nbrs_sp0.radius_neighbors(X=sp1,
                                                                              radius=max_distance,
                                                                              return_distance=True,
                                                                              sort_results=True)
            nbrs_sp1 = NearestNeighbors(algorithm='ball_tree').fit(sp1)
            neigh_dist_sp1_sp0, neigh_ind_sp1_sp0 = nbrs_sp1.radius_neighbors(X=sp0_ref, radius=max_distance,
                                                                              return_distance=True, sort_results=True)

            for index_sp0 in range(len(sp0_ref)):
                if len(neigh_ind_sp1_sp0[index_sp0]) > 0:
                    if neigh_ind_sp0_sp1[neigh_ind_sp1_sp0[index_sp0][0]][0] == index_sp0:
                        list_couple_index_sp0_ref.append(index_sp0)
                        list_couple_index_sp1.append(neigh_ind_sp1_sp0[index_sp0][0])
                        list_distance.append(neigh_dist_sp1_sp0[index_sp0][0])
                        sp0_ref_pair_order.append(sp0_ref[index_sp0])
                        sp1_pair_order.append(sp1[neigh_ind_sp1_sp0[index_sp0][0]])


            print(np.median(list_distance))
            print(len(list_distance))


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
                ax.set_xlim(left = 0, right= 1000)
                fig.savefig(path_folder_image + "/" + f"{ref_round}_{round}_{pos}")
                print(np.mean(list_distance))

    list_couple_index_sp0_ref, list_couple_index_sp1, sp0_ref_pair_order, sp1_pair_order, list_distance, list_distance_coord

    dico_matrix_transform[source_folder][target_folder][pos0]['spots_0_colocalized'] = spots_0_colocalized
    dico_matrix_transform[source_folder][target_folder][pos0]['new_spots_1_colocalized'] = new_spots_1_colocalized





