




#%%

from pycpd import AffineRegistration, RigidRegistration
import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error
import bigfish
from bigfish import multistack
################### pair two point cloud with CPD




def compute_transformation_matrix_folder(
    dico_detection,
    transform_method = "affine_cpd",
    voxel_size = [0.3, 0.103, 0.103],
    scale = None, ## for old method
    max_dist = 0.5,
    first_regex_check = 'r',
    method  = "old"):
    """

    Args:
        dico_detection:
        transform_method:
        voxel_size:
        max_dist:
        first_regex_check:

    Returns:

    """
    method in ['old', 'new']

    dico_matrix_transform = {}
    dico_matrix_transform_json = {}
    folder_round_list = list(dico_detection.keys())
    print(folder_round_list)
    for source_folder in folder_round_list:
        if source_folder[:len(first_regex_check)] != first_regex_check:
            continue
        dico_matrix_transform[source_folder] = {}
        dico_matrix_transform_json[source_folder] = {}
        for target_folder in folder_round_list:
            if target_folder[:len(first_regex_check)] != first_regex_check:
                continue
            dico_matrix_transform[source_folder][target_folder] = {}
            dico_folder_round0 = dico_detection[source_folder]
            dico_folder_round1 = dico_detection[target_folder]
            list_image0 = list(dico_folder_round0.keys())
            list_image1 = list(dico_folder_round1.keys())
            print(list_image0)
            print(list_image1)
            print(source_folder)
            print(target_folder)
            for pos0 in list_image0:
                for pos1 in list_image1:
                    if pos0 != pos1:
                        continue
                    print((pos0, pos1))
                    dico_matrix_transform[source_folder][target_folder][pos0] = {}
                    bead_subpixel0 = dico_folder_round0[pos0]["subpixel_spots"]
                    bead_subpixel1 = dico_folder_round1[pos0]["subpixel_spots"]

                    if len(bead_subpixel0) == 2: #I dont understadn the second return of fit_subpixel
                        bead_subpixel0 = bead_subpixel0[0]
                    if len(bead_subpixel1) == 2: #I dont understadn the second return of fit_subpixel
                        bead_subpixel1 = bead_subpixel1[0]

                    if method == 'new':
                        R, t, s, error_before_transformation, error_after_transformation,  spots_0_colocalized, spots_1_colocalized,  new_spots_1_colocalized = compute_rotation_translation(
                            sub_sp0 = bead_subpixel0,
                                                     sub_sp1 = bead_subpixel1,
                                                     transform_method=transform_method,
                                                     voxel_size=voxel_size,
                                                     max_dist=max_dist)
                    else:

                        R, t, s, error_before_transformation, error_after_transformation, \
                            spots_0_colocalized, spots_1_colocalized, new_spots_1_colocalized = compute_rotation_translation_icp(
                                                         sp0 = bead_subpixel0,
                                                         sp1 =  bead_subpixel1,
                                                         method_pairing="icp",
                                                         transform_method="affine_cpd",
                                                         radii=[5, 60, 60],
                                                         real_threshold_nn=np.array([0.5, 0.5, 0.5]),
                                                         scale=scale,
                                                         min_bead=4)
                        if error_before_transformation is not None:
                            print(f'error_before_transformation in nm {error_before_transformation * np.array([270, 108, 108])}' )
                            print(f'error_after_transformation in nm {error_after_transformation * np.array([270, 108, 108])}' )
                            print()
                        else:
                            print(f"Fail in {source_folder}_{target_folder}_{pos0}")





                    dico_matrix_transform[source_folder][target_folder][pos0]['R'] = R
                    dico_matrix_transform[source_folder][target_folder][pos0]['t'] = t
                    dico_matrix_transform[source_folder][target_folder][pos0]['s'] = s
                    dico_matrix_transform[source_folder][target_folder][pos0]['error_before_transformation'] = error_before_transformation
                    dico_matrix_transform[source_folder][target_folder][pos0]['error_after_transformation'] = error_after_transformation
                    dico_matrix_transform[source_folder][target_folder][pos0]['spots_0_colocalized'] = spots_0_colocalized
                    dico_matrix_transform[source_folder][target_folder][pos0]['spots_1_colocalized'] = spots_1_colocalized
                    dico_matrix_transform[source_folder][target_folder][pos0]['new_spots_1_colocalized'] = new_spots_1_colocalized
                    dico_matrix_transform[source_folder][target_folder][pos0]['transform_method'] = transform_method
    return dico_matrix_transform





def compute_rotation_translation(sub_sp0,
                                 sub_sp1,
                                 transform_method = "affine_cpd",
                                 voxel_size = [0.3, 0.103, 0.103],
                                 max_dist = None):



    spots_0_colocalized, spots_1_colocalized, distances, indices_1, indices_2, threshold = \
        multistack.detect_spots_colocalization(spots_1 = sub_sp0,
                                                   spots_2 = sub_sp1,
                                                   voxel_size = voxel_size,
                                                   threshold=max_dist, ## should I TAKE A MARGE ?
                                                   return_indices=True,
                                                   return_threshold=True)
    spots_0_colocalized = spots_0_colocalized[distances < max_dist]
    spots_1_colocalized = spots_1_colocalized[distances < max_dist]
    distances = distances[distances < max_dist]
    #indices_1 = indices_1[distances < max_dist]
    #indices_2 = indices_2[distances < max_dist]
    print(f'number of spots {len(spots_0_colocalized)}')

    error_before_transformation = mean_absolute_error(spots_0_colocalized,
                                                      spots_1_colocalized,
                                                      multioutput='raw_values')
    print(f'error_before_transformation {error_before_transformation}')
    if transform_method == "rigid":
        reg = RigidRegistration(**{'X': spots_0_colocalized, 'Y':spots_1_colocalized })
        reg.register()
        R, t, s = reg.get_registration_parameters()
        new_spots_1_colocalized = reg.transform_point_cloud(spots_1_colocalized)
    elif transform_method == "affine_cpd":

        reg = AffineRegistration(**{'X': spots_0_colocalized, 'Y':spots_1_colocalized })
        reg.register()
        R, t = reg.get_registration_parameters()
        s = None
        new_spots_1_colocalized = reg.transform_point_cloud(spots_1_colocalized)

    else:
        raise(Exception(f''))
    error_after_transformation = mean_absolute_error(spots_0_colocalized,
                                                      new_spots_1_colocalized,
                                                      multioutput='raw_values')
    print(f'error_after_transformation {error_after_transformation}')
    spots_0_colocalized_bis, spots_1_colocalized_bis, distances, indices_1, indices_2, threshold = \
        multistack.detect_spots_colocalization(spots_1 = spots_0_colocalized,
                                                   spots_2 = new_spots_1_colocalized,
                                                   voxel_size = voxel_size,
                                                   threshold=0.5, ## should I TAKE A MARGE ?
                                                   return_indices=True,
                                                   return_threshold=True)

    print(f'number of spots {len(spots_0_colocalized_bis)}')


    error_before_transformation = mean_absolute_error(spots_0_colocalized_bis,
                                                      spots_1_colocalized_bis,
                                                      multioutput='raw_values')
    print(f'error_before_transformation {error_before_transformation}')
    if transform_method == "rigid":
        reg = RigidRegistration(**{'X': spots_0_colocalized_bis, 'Y':spots_1_colocalized_bis })
        reg.register()
        R, t, s = reg.get_registration_parameters()
        new_spots_1_colocalized_bis = reg.transform_point_cloud(spots_1_colocalized_bis)
    elif transform_method == "affine_cpd":

        reg = AffineRegistration(**{'X': spots_0_colocalized_bis, 'Y':spots_1_colocalized_bis })
        reg.register()
        R, t = reg.get_registration_parameters()
        s = None
        new_spots_1_colocalized_bis = reg.transform_point_cloud(spots_1_colocalized_bis)

    else:
        raise(Exception(f''))
    error_after_transformation = mean_absolute_error(spots_0_colocalized_bis,
                                                      new_spots_1_colocalized_bis,
                                                      multioutput='raw_values')
    print(f'error_after_transformation {error_after_transformation * 1000} in nm')




    return R, t, s, error_before_transformation, error_after_transformation, spots_0_colocalized, spots_1_colocalized, new_spots_1_colocalized



#####################
################## compute rotation translation from last time
################################


from pyoints import (registration,  # storage,; Extent,; filters,; normals,
                     transformation)


def compute_first_icp(sp0, sp1, radii=[3, 30, 30]):
    """
    compute the candidate pair a beads using an ICP

    :param sp0:
    :param sp1:
    :param radii: parameter to tune
    :return:
    """
    coords_dict = {"sp0": np.array(sp0),
                   "sp1": np.array(sp1)}
    icp = registration.ICP(
        radii,
        max_iter=100,
        max_change_ratio=0.000001,
        k=1)
    T_dict, pairs_dict, report = icp(coords_dict)
    if len(pairs_dict["sp0"]["sp1"][0]) < 3: #not enougth match
        return None, None, None, None

    sp0_pair = sp0[pairs_dict["sp0"]["sp1"][0][:, 0]]
    sp1_pair = sp1[pairs_dict["sp0"]["sp1"][0][:, 1]]

    return sp0_pair, sp1_pair, T_dict, report



def rigid_transform_3D(A, B):
    """
    # https://github.com/nghiaho12/rigid_transform_3D
    # Input: expects 3xN matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix
    # t = 3x1 column vector
    # then R @ A.T) + t).T is equal to B
    :param A:
    :type A:
    :param B:
    :type B:
    :return:
    :rtype:
    """

    assert A.shape == B.shape
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")
    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)
    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)
    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am @ np.transpose(Bm)
    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))
    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = -R @ centroid_A + centroid_B
    return R, t


def compute_pair_icp(sp0, sp1, radii=[6,60,60],  real_threshold_nn =  [0.5, 0.5, 0.5], min_bead = 4):
    """
    :param sp0: list of spots
    :type sp0:
    :param sp1: list of spots
    :type sp1:
    :param radii: icp parameter
    :type radii:
    :param real_threshold_nn: icp parameter
    :type real_threshold_nn:
    :param min_bead: minimum bead to take into account
    :type min_bead:
    :return:
    :rtype: list, sub_sp0_pair : pair of bead to take into account to compute  the affine transformation
    :rtype: list,sub_sp1_pair,
    :rtype: list,sub_sp0_pair_remove pair of bead not takken into account
     :rtype: list, sub_sp1_pair_remove,
      :rtype: list, error_before_reduction
    :rtype:
    """

    sub_sp0_pair, sub_sp1_pair, sub_T_dict, sub_report = compute_first_icp(sp0, sp1, radii=radii)
    error_before_reduction = mean_absolute_error(sub_sp0_pair, sub_sp1_pair, multioutput='raw_values')
    ## remove vector not in the same direction than the median direction
    colinear_vect = np.sum((sub_sp0_pair - sub_sp1_pair) * np.median(sub_sp0_pair - sub_sp1_pair, axis=0), axis=1) > 0
    if np.sum(colinear_vect) != len(sub_sp0_pair) and  np.sum(colinear_vect) > 3:
        print(colinear_vect)
        sub_sp0_pair = sub_sp0_pair[colinear_vect]
        sub_sp1_pair = sub_sp1_pair[colinear_vect]
        sub_sp0_pair, sub_sp1_pair, sub_T_dict, sub_report = compute_first_icp(sub_sp0_pair, sub_sp1_pair, radii=radii)
    if sub_sp0_pair is None or len(sub_sp0_pair) <= min_bead:
        print("return None in compute_pair_icp")
        return None, None, None, None, None
    # remove pair above a threshold
    real_nn = np.abs(transformation.transform(sub_sp0_pair,
                    sub_T_dict["sp0"]) - transformation.transform(sub_sp1_pair, sub_T_dict["sp1"])) < real_threshold_nn

    fake_index_nn = (np.sum(real_nn, axis=1) < 3)

    sub_sp0_pair_remove = sub_sp0_pair[fake_index_nn]
    sub_sp1_pair_remove = sub_sp1_pair[fake_index_nn]


    real_index_nn = (np.sum(real_nn, axis=1) == 3)
    sub_sp0_pair = sub_sp0_pair[real_index_nn]
    sub_sp1_pair = sub_sp1_pair[real_index_nn]

    return sub_sp0_pair, sub_sp1_pair, sub_sp0_pair_remove, sub_sp1_pair_remove, error_before_reduction

def compute_rotation_translation_icp(sp0,
                                     sp1,
                                     method_pairing = "icp",
                                     transform_method = "affine_cpd",
                                     radii=[5, 60, 60],
                                     real_threshold_nn=np.array([0.3, 0.3, 0.3]),
                                     scale = np.array([270, 108, 108]),
                                     min_bead = 4):
    """
    :param sp0:  spots array
    :param sp1: spots array
    method (str): choose in 'icp', 'lsa,
    :param radii: list of the [x, y, z] limit for neibors distance for the ICP
    :param real_threshold_nn: Limit to considere two balls are indeed matching after transformation
    :return: R, t, error_before, error_after, sub_sp0_pair, sub_sp1_pair
    """

    if scale is not None:
        sp0 = sp0 * scale
        sp1 = sp1 * scale
        radii = np.array(radii) * scale
        real_threshold_nn = real_threshold_nn * scale
    assert method_pairing in ["icp"]
    assert transform_method in ["affine_cpd", "rigid_cpd",  'rigid', "all"]
    if method_pairing == "icp":
        sub_sp0_pair, sub_sp1_pair, sub_sp0_pair_remove, sub_sp1_pair_remove, error_before_reduction = compute_pair_icp(sp0,
                                                                                                                        sp1,
                                                                                                                        radii=radii,
                                                                                                                        real_threshold_nn =  real_threshold_nn)

    if sub_sp0_pair is None or len(sub_sp0_pair) <= (min_bead-1):
        print()
        print("None return in compute_rotation_translation")
        print("FAIL")
        print()
        return None, None, None, None, None, None, None, None
    # recompute optimal transformation with the real pair
    if transform_method == 'rigid':
        R, t = rigid_transform_3D(sub_sp1_pair.T, sub_sp0_pair.T)
        error_before_transformation = mean_absolute_error(sub_sp0_pair, sub_sp1_pair, multioutput='raw_values')
        error_after = mean_absolute_error(((R @ sub_sp1_pair.T) + t).T,
                                          sub_sp0_pair,
                                          multioutput='raw_values')
        s = None

    if transform_method == 'affine_cpd':
        reg = AffineRegistration(**{'X': sub_sp0_pair, 'Y': sub_sp1_pair})
        reg.register()
        R, t = reg.get_registration_parameters() #some time it does not work especially when there is not enought bead
        error_before_transformation = mean_absolute_error(sub_sp0_pair, sub_sp1_pair, multioutput='raw_values')
        error_after = np.mean(np.abs(sub_sp0_pair  - reg.transform_point_cloud(sub_sp1_pair)), axis = 0)
        s = None
    if transform_method == 'rigid_cpd':
        reg = RigidRegistration(**{'X': sub_sp0_pair, 'Y': sub_sp1_pair})
        reg.register()
        R, t, s = reg.get_registration_parameters()
        error_before_transformation = mean_absolute_error(sub_sp0_pair, sub_sp1_pair, multioutput='raw_values')
        error_after = np.mean(np.abs(sub_sp1_pair  - reg.transform_point_cloud(sub_sp1_pair)), axis = 0)
    print("error before: %s, error after: %s" % (str(error_before_transformation), str(error_after)))
    print("orignial potential pair: %s, final number of pair: %s" % (
    str(min(len(sp0), len(sp1))), str(len(sub_sp1_pair))))
    return R, t, s, error_before_transformation, error_after, sub_sp0_pair, sub_sp1_pair,  reg.transform_point_cloud(sub_sp1_pair)





#%%


if __name__ == "__main__":
    np.save("/media/tom/T7/2023-01-19-PAPER-20-rounds/test/dico_bead_detection",
            dico_bead_detection)
    dico_bead_detection = np.load("/media/tom/T7/2023-01-19-PAPER-20-rounds/test/dico_bead_detection.npy",
            allow_pickle=True).item()

    sub_sp0_pair = dico_bead_detection['r1']['r1_pos0_ch1.tif']['subpixel_spots']
    sub_sp1_pair = dico_bead_detection['r2']['r2_pos0_ch1.tif']['subpixel_spots']

    max_dist = 0.5

    spots_0_colocalized, spots_1_colocalized, distances, indices_1, indices_2, threshold = \
        multistack.detect_spots_colocalization(spots_1 = sub_sp0_pair,
                                                   spots_2 = sub_sp1_pair,
                                                   voxel_size = [0.3, 0.103, 0.103],
                                                   threshold=None,
                                                   return_indices=True,
                                                   return_threshold=True)

    spots_0_colocalized = spots_0_colocalized[distances < max_dist]
    spots_1_colocalized = spots_1_colocalized[distances < max_dist]
    distances = distances[distances < max_dist]
    #indices_1 = indices_1[distances < max_dist]
    #indices_2 = indices_2[distances < max_dist]
    print(f'number of spots {len(spots_0_colocalized)}')
    error_before_transformation = mean_absolute_error(spots_0_colocalized,
                                                      spots_1_colocalized,
                                                      multioutput='raw_values')
    print(f'error_before_transformation {error_before_transformation}')
    reg = RigidRegistration(**{'X': spots_0_colocalized, 'Y':spots_1_colocalized })
    reg.register()
    R, t, s = reg.get_registration_parameters()
    new_sub_sp1_pair = reg.transform_point_cloud(spots_1_colocalized)
    error_after_transformation = mean_absolute_error(spots_0_colocalized,
                                                      new_sub_sp1_pair,
                                                      multioutput='raw_values')
    print(f'error_after_transformation {error_after_transformation}')

    error_after = np.mean(np.abs(sub_sp1_pair - reg.transform_point_cloud(sub_sp1_pair)), axis=0)


    ### test main

    dico_bead_detection = np.load("/media/tom/T7/2023-01-19-PAPER-20-rounds/test/dico_bead_detection.npy",
            allow_pickle=True).item()

    dico_t = compute_transformation_matrix_folder(
        dico_detection = dico_bead_detection,
        transform_method="affine_cpd",
        voxel_size=[0.3, 0.103, 0.103],
        max_dist=0.5,
        first_regex_check='r',
    )