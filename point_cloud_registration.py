






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
    max_dist = 0.5,
    first_regex_check = 'r',
    ):


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
            for im_name0 in list_image0:
                pos0 = im_name0.split('_')[1]
                for im_name1 in list_image1:
                    pos1 = im_name1.split('_')[1]
                    if pos0 != pos1:
                        continue
                    print((pos0, pos1))
                    dico_matrix_transform[source_folder][target_folder][pos0] = {}
                    bead_subpixel0 = dico_folder_round0[im_name0]["subpixel_spots"]
                    bead_subpixel1 = dico_folder_round1[im_name1]["subpixel_spots"]

                    R, t, s, error_before_transformation, error_after_transformation,  spots_0_colocalized, new_spots_1_colocalized \
                        = compute_rotation_translation(sub_sp0 = bead_subpixel0,
                                                 sub_sp1 = bead_subpixel1,
                                                 transform_method=transform_method,
                                                 voxel_size=voxel_size,
                                                 max_dist=max_dist)
                    dico_matrix_transform[source_folder][target_folder][pos0]['R'] = R
                    dico_matrix_transform[source_folder][target_folder][pos0]['t'] = t
                    dico_matrix_transform[source_folder][target_folder][pos0]['s'] = s
                    dico_matrix_transform[source_folder][target_folder][pos0]['error_before_transformation'] = error_before_transformation
                    dico_matrix_transform[source_folder][target_folder][pos0]['error_after_transformation'] = error_after_transformation
                    dico_matrix_transform[source_folder][target_folder][pos0]['spots_0_colocalized'] = spots_0_colocalized
                    dico_matrix_transform[source_folder][target_folder][pos0]['new_spots_1_colocalized'] = new_spots_1_colocalized
                    dico_matrix_transform[source_folder][target_folder][pos0]['transform_method'] = transform_method
    return dico_matrix_transform





def compute_rotation_translation(sub_sp0,
                                 sub_sp1,
                                 transform_method = "affine_cpd",
                                 voxel_size = [0.3, 0.103, 0.103],
                                 max_dist = 0.5):



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

    error_before_transformation = mean_absolute_error(spots_0_colocalized,
                                                      spots_1_colocalized,
                                                      multioutput='raw_values')
    print(f'error_before_transformation {error_before_transformation}')
    if transform_method == "rigid":
        reg = RigidRegistration(**{'X': spots_0_colocalized, 'Y':spots_1_colocalized })
        reg.register()
        R, t, s = reg.get_registration_parameters()
        new_sub_sp1_pair = reg.transform_point_cloud(spots_1_colocalized)
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

    return R, t, s, error_before_transformation, error_after_transformation, spots_0_colocalized, new_spots_1_colocalized






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