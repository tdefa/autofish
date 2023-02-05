# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from bigfish import detection

import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from spots_detection import compute_spot_detection_of_folder, get_reference_dico
from tqdm import tqdm
from plots import  plot_beads_image_folder, plot_beads_matching_image_folder

from point_cloud_registration import compute_transformation_matrix_folder


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds_folder", help="folder with all the rounds",
                        default="/media/tom/T7/2023-01-19-PAPER-20-rounds/test")
                        #default = "/media/tom/T7/2023-01-19-PAPER-20-rounds/images")
    parser.add_argument("--beads_channel",
                        default="ch1")
    parser.add_argument("--spots_channel",
                        default="ch0")
    parser.add_argument("--voxel_size", help="[z,x,y], how is it use :"
                                             "    sigma_yx = psf_yx / voxel_size_yx, "
                                             "    radius = [np.sqrt(len(sigma)) * sigma_ for sigma_ in sigma]"
                                             "get spots volume"
                                             "    y_spot_min = max(0, int(spot_y - radius_yx)) to buld the reference spots"
                        ,
                        default=np.array([300, 108, 108])
                        )

    ### bead detection
    parser.add_argument("--spot_radius_rna", help="",
                        default=np.array([900, 300, 300])
                        )
    parser.add_argument("--reference_mode", default="median")

    ### spots detection


    parser.add_argument("--spot_radius_bead", help="",
                        default=np.array([600, 300, 300])
                        )

    parser.add_argument("--host", default='127.0.0.2')
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=52162)
    args = parser.parse_args()

############# detect the bead store in in a dico [round][image_name] = spots coordiante

    dico_bead_detection  = compute_spot_detection_of_folder(
        path_to_folder=args.rounds_folder,
        channel = args.beads_channel,
        Key_world = "os",
        min_distance = [5, 7, 7],
        sigma = [1.35, 1.35, 1.35],
        threshold = None,
        reference_number_of_spot = None,
        beta = 1,
        alpha = 0.5,
        voxel_size = tuple(args.voxel_size),
        get_nb_spots_per_cluster = False,
        subpixel_loc = True,
        spot_radius = tuple(args.spot_radius_bead),
        mode = "bead")

    np.save(args.rounds_folder + "/dico_bead_detection", dico_bead_detection)
    dico_bead_detection = np.load(args.rounds_folder + "/dico_bead_detection.npy", allow_pickle=True).item()

    plot_beads_image_folder(dico_bead_detection,
                            first_regex_check="r",
                            channel=args.beads_channel,
                            min_distance=[5, 7, 7],
                            psf=[1.35, 1.35, 1.35],
                            folder_patho=args.rounds_folder,
                            radius=9,
                            linewidth=2,
                            fill=False,
                            figsize=(20, 20),
                            fontsize_legend=8,
                            )


    #### Compute the Transformation between round

    dico_matrix_transform = compute_transformation_matrix_folder(
        dico_bead_detection,
        transform_method="affine_cpd",
        voxel_size=[0.3, 0.103, 0.103],
        max_dist=0.5,
        first_regex_check='r',
    )

    np.save(args.rounds_folder + "/dico_matrix_transform", dico_matrix_transform)
    dico_matrix_transform = np.load(args.rounds_folder + "/dico_matrix_transform.npy", allow_pickle=True).item()

    plot_beads_matching_image_folder(
        first_regex_check="r",
        channel="ch1",
        folder_patho=args.rounds_folder,
        psf="",
        min_distance="",
        round_source="r1",
        dico_matrix_transform=dico_matrix_transform,
        radius=6,
        linewidth=1,
        fill=False,
        figsize=(20, 20),
        fontsize_legend=8)

############## detect the rna with the same number of rna in each images [round][image_name] = spots coordiante

## first spots detection

    first_spots_detection  = compute_spot_detection_of_folder(path_to_folder=args.rounds_folder,
            channel = args.spots_channel,
            Key_world = "os",
            min_distance = [4, 4, 4],
            sigma = [1.35, 1.35, 1.35],
            threshold = None,
            reference_number_of_spot = None,
            beta = 1,
            alpha = 0.5,
            voxel_size = tuple(args.voxel_size),
            get_nb_spots_per_cluster = False,
            subpixel_loc = False,
            spot_radius = tuple(args.spot_radius_bead),
            mode = "bead")

    np.save(args.rounds_folder + "/first_spots_detection",
            first_spots_detection)
    first_spots_detection = np.load(args.rounds_folder + "/first_spots_detection.npy", allow_pickle="True").item()


    dico_ref_nb_spot_per_position = get_reference_dico(
        first_dico_detection = first_spots_detection,
        mode=args.reference_mode)

    ## second spot detection
    final_spots_detection  = compute_spot_detection_of_folder(
        path_to_folder=args.rounds_folder,
        channel = args.spots_channel,
        Key_world = "pos",
        min_distance = [4, 4, 4],
        sigma = [1.35, 1.35, 1.35],
        threshold = None,
        reference_number_of_spot = dico_ref_nb_spot_per_position,
        beta = 1,
        alpha = 0.5,
        voxel_size = tuple(args.voxel_size),
        get_nb_spots_per_cluster = False,
        subpixel_loc = True,
        spot_radius = tuple(args.spot_radius_bead),
        mode = "bead")
    np.save(args.rounds_folder + "/final_spots_detection", dico_bead_detection)







########### compute registation with phase corelation


## compute the number of pair RNA per round



##########  plot snr / intensité




######## plot precision de co-localisation


### compute supixel