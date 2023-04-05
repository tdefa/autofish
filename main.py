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
import tifffile

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds_folder", help="folder with all the rounds",
                        #default = "/media/tom/T7/2023-01-19-PAPER-20-rounds/round_impair")
                        default = "/media/tom/T7/2023-01-19-PAPER-20-rounds/round_pair")
                        #default = "/media/tom/T7/2023-01-19-PAPER-20-rounds/test_folder/test3")
                        #default = "/media/tom/T7/2023-01-19-PAPER-20-rounds/r1_r2")

    parser.add_argument("--beads_channel",
                        default="ch1")
    parser.add_argument("--spots_channel",
                        default="ch0")
    parser.add_argument("--voxel_size", help="[z,x,y],"
                                             " how is it use :"
                                             "sigma_yx = psf_yx / voxel_size_yx, "
                                             "radius = [np.sqrt(len(sigma)) * sigma_ for sigma_ in sigma]"
                                             "get spots volume"
                                             "y_spot_min = max(0, int(spot_y - radius_yx)) to buld the reference spots"
                        ,
                        default=np.array([270, 108, 108])
                        )
    ### bead detection
    parser.add_argument("--spot_radius_rna", help="",
                        default=np.array([400, 200, 200])
                        )
    parser.add_argument("--reference_mode", default="median")

    ## bead matching

    parser.add_argument("--transform_method", default="rigid")

    ### spots detection
    parser.add_argument("--spot_radius_bead", help="",
                        default=np.array([800, 600, 600])
                        )
    parser.add_argument("--mask_non_cytoplasmatic_spots", help="", default=True)
    ## painring
    parser.add_argument("--ref_round",
                        default="r2")
    #task_to_do
    parser.add_argument("--detect_beads", help="", type = int,
                        default=1)
    parser.add_argument("--compute_transform", help="",type = int,
                        default=0)
    parser.add_argument("--plot_beads",type = int,
                        help="", default=0)

    parser.add_argument("--detect_spots", help="",type = int,
                        default=1)
    parser.add_argument("--plot_spots", help="",type = int,
                        default=0)
    parser.add_argument("--pairing", help="",type = int,
                        default=0)
    parser.add_argument("--pairing_plots_analysis", help="",type = int,
                        default=0)
    parser.add_argument("--pairing_plots_spots", help="",type = int,
                        default=0)
    parser.add_argument("--mask_cyto_registration", help="",type = int,
                        default=0)
    parser.add_argument("--host", default='127.0.0.2')
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=52162)
    args = parser.parse_args()
    print(args)
    path_folder_save_plot = args.rounds_folder + '/histogram'


############# detect the bead store in in a dico[round][image_name] = spots coordiante

    if args.detect_beads:
        print("detect_beads")

        dico_bead_detection  = compute_spot_detection_of_folder(
            path_to_folder=args.rounds_folder,
            channel = args.beads_channel,
            Key_world = "os",
            first_key_world="r",
            min_distance = [7, 7, 7],
            sigma = [1.35, 1.35, 1.35],
            threshold = None,
            reference_number_of_spot = None,
            beta = 1,
            alpha = 0.5,
            voxel_size = tuple(args.voxel_size),
            get_nb_spots_per_cluster = False,
            subpixel_loc = True,
            spot_radius = tuple(args.spot_radius_bead),
            mode = "bead",
        artefact_removal = False)

        np.save(args.rounds_folder + "/dico_bead_detection" + str(tuple(args.spot_radius_bead)), dico_bead_detection)


    if args.compute_transform:


        print('compute_transform')
        from point_cloud_registration import compute_transformation_matrix_folder
        dico_bead_detection = np.load(args.rounds_folder + f"/dico_bead_detection{tuple(args.spot_radius_bead)}.npy",
                                      allow_pickle=True).item()


        dico_matrix_transform = compute_transformation_matrix_folder(
            dico_detection = dico_bead_detection,
            transform_method=args.transform_method,
            voxel_size=args.voxel_size,
            scale = None,
            max_dist=2,
            first_regex_check='r',
        )


        np.save(args.rounds_folder + "/dico_matrix_transform"+ str(tuple(args.spot_radius_bead)), dico_matrix_transform)





    if args.plot_beads:


        print("plot bead")
        dico_matrix_transform = np.load(
            args.rounds_folder + f"/dico_matrix_transform{tuple(args.spot_radius_bead)}.npy",
            allow_pickle=True).item()
        dico_bead_detection = np.load(args.rounds_folder + f"/dico_bead_detection{tuple(args.spot_radius_bead)}.npy",
                                      allow_pickle=True).item()

        plot_beads_image_folder(dico_bead_detection,
                                first_regex_check="r",
                                channel=args.beads_channel,
                                min_distance=[7, 7, 7],
                                psf=[1.35, 1.35, 1.35],
                                folder_patho=args.rounds_folder,
                                radius=9,
                                linewidth=2,
                                fill=False,
                                figsize=(20, 20),
                                fontsize_legend=8,
                                )




        plot_beads_matching_image_folder(
            first_regex_check="r",
            channel="ch1",
            folder_patho=args.rounds_folder,
            psf="",
            min_distance="",
            round_source=args.ref_round,
            dico_transform_spots=dico_matrix_transform,
            radius=6,
            linewidth=1,
            fill=False,
            figsize=(20, 20),
            fontsize_legend=8,
            plot_mode="bead",
            rescale=True
        )

############## detect the rna with the same number of rna in each images [round][image_name] = spots coordiante


## mask registration

    if args.mask_cyto_registration:

        from point_cloud_registration import register_seg_mask_folder
        dico_matrix_transform = np.load(
            args.rounds_folder + f"/dico_matrix_transform{tuple(args.spot_radius_bead)}.npy",
            allow_pickle=True).item()
        register_seg_mask_folder(path_rounds_folder=args.rounds_folder,
                                 mask_folder_name="cyto_mask2D_3dim",
                                 ref_round=args.ref_round,
                                 dico_matrix_transform=dico_matrix_transform,
                                 # np.load("/media/tom/T7/2023-01-19-PAPER-20-rounds/test_folder/test3/dico_matrix_transform(800, 600, 600).npy",allow_pickle=True).item(),
                                 transform_method=args.transform_method,
                                 round_first_regex="r",
                                 fish_channel=args.spots_channel,
                                 )



## first spots detection
    if args.detect_spots:

        print("detect_spots")
        first_spots_detection  = compute_spot_detection_of_folder(
                        path_to_folder=args.rounds_folder,
                        channel = args.spots_channel,
                        Key_world = "os",
                        first_key_world="r",
                        min_distance = [4, 4, 4],
                        sigma = [1.35, 1.35, 1.35],
                        threshold = None,
                        reference_number_of_spot = None,
                        beta = 1,
                        alpha = 0.5,
                        voxel_size = tuple(args.voxel_size),
                        get_nb_spots_per_cluster = False,
                        subpixel_loc = True,
                        spot_radius = tuple(args.spot_radius_rna),
                        mode = "bead",
                        mask_non_cytoplasmatic_spots=args.mask_non_cytoplasmatic_spots,
                        folder_cytoplasmatic_mask="cyto_mask2D_3dim/",
                        artefact_removal=False,
                        min_cos_theta=None,
                        order=None)

        np.save(args.rounds_folder + f"/first_spots_detection{tuple(args.spot_radius_bead)}",
                first_spots_detection)

        if False: #use averge detection to fix threshold
            first_spots_detection = np.load(args.rounds_folder + f"/first_spots_detection{tuple(args.spot_radius_bead)}.npy",
                                            allow_pickle="True").item()
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
                spot_radius = tuple(args.spot_radius_rna),
                mode = "bead")
            np.save(args.rounds_folder + f"/final_spots_detection{tuple(args.spot_radius_bead)}", final_spots_detection)
        else:
            np.save(args.rounds_folder + f"/final_spots_detection{tuple(args.spot_radius_bead)}", first_spots_detection)


    if args.plot_spots:
        print("plot_spots")

        final_spots_detection = np.load(args.rounds_folder + f"/final_spots_detection{tuple(args.spot_radius_bead)}.npy", allow_pickle=True).item()

        plot_beads_image_folder(final_spots_detection,
                                first_regex_check="r",
                                channel=args.spots_channel,
                                min_distance="",
                                psf="",
                                folder_patho=args.rounds_folder,
                                radius=9,
                                linewidth=2,
                                fill=False,
                                figsize=(20, 20),
                                fontsize_legend=8,
                                folder_name = 'rna_detection'
                                )

    if False:
        #pair with all the rounds
        print()

    ### pair it

    if args.pairing:

        final_spots_detection = np.load(args.rounds_folder + f"/final_spots_detection{tuple(args.spot_radius_bead)}.npy",
                                        allow_pickle=True).item()
        dico_matrix_transform = np.load(
            args.rounds_folder + f"/dico_matrix_transform{tuple(args.spot_radius_bead)}.npy",
            allow_pickle=True).item()

        """dico_matrix_transform = np.load(
            "/media/tom/T7/2023-01-19-PAPER-20-rounds/test_folder/test3/dico_matrix_transform(800, 600, 600).npy",
            allow_pickle=True).item()
        final_spots_detection = np.load(
            "/media/tom/T7/2023-01-19-PAPER-20-rounds/test_folder/test3/final_spots_detection(800, 600, 600).npy",
            allow_pickle=True).item()"""
        from pairing import compute_pair_folder

        dico_matched_rna = compute_pair_folder(
            dico_matrix_transform,
            final_spots_detection,
            scale_z_x_y=np.array([270, 108, 108]),
            max_distance=1000,
            ref_round=args.ref_round,
            mean_substraction=False,
            plot_hist=True,
            path_folder_save_plot=path_folder_save_plot,

        )
        np.save(args.rounds_folder + f"/dico_matched_rna_ref_round{args.ref_round}", final_spots_detection)


        all_dico_matched_rna = {}
        for r in dico_matched_rna:
            dico_matched_rna_r = compute_pair_folder(
                dico_matrix_transform,
                final_spots_detection,
                scale_z_x_y=np.array([270, 108, 108]),
                max_distance=1000,
                ref_round=r,
                mean_substraction=False,
                plot_hist=False,
                path_folder_save_plot=path_folder_save_plot
            )
            all_dico_matched_rna[r] = dico_matched_rna_r

    if args.pairing:

        #####################
        # histogram plotting
        #############"
        ref_round = args.ref_round
        pos_list = list(dico_matched_rna[list(dico_matched_rna.keys())[0]].keys())
        nb_rna_match_aggregated = []
        percentage_rna_match_aggregated = []
        median_relocalization_distance_aggregated = []
        for pos in pos_list:
            nb_rna_match = []
            percentage_rna_match = []
            median_relocalization_distance = []
            x_list = []
            round_list_sorted = sorted(list(dico_matched_rna.keys()), key=lambda kv: int(kv[1:]))[:]
            for round in round_list_sorted:
                if len(dico_matched_rna[round][pos]) == 0:
                    continue
                x_list.append(round)
                nb_rna_match.append(len(dico_matched_rna[round][pos]['list_distance']))
                ref_round = dico_matched_rna[round][pos]['ref_round']
                percentage_rna_match.append(len(dico_matched_rna[round][pos]['list_distance'])
                                    /
                                    len(dico_matched_rna[ref_round][pos]['list_distance']) * 100)
                median_relocalization_distance.append(np.median(dico_matched_rna[round][pos]['list_distance']))

            fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
            fig.suptitle(f'median_relocalization_distance, reference is {ref_round}', fontsize=30, x=0.1, ha='left')
            ax.plot(x_list[1:], median_relocalization_distance[1:])
            ax.set_ylabel("median relocalization distance  in [nm]", fontsize=18)
            ax.set_xlabel("round", fontsize=18)
            fig.savefig(path_folder_save_plot + "/" + f'median_relocalization_distance_reference_{ref_round}_{pos}')

            plt.show()

            fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
            fig.suptitle(f'percentage_rna_match, reference is {ref_round}_{pos}', fontsize=30, x=0.1, ha='left')
            ax.plot(x_list, percentage_rna_match)
            ax.set_ylabel(f"percentage of rna colocalize from {ref_round}_{pos}", fontsize=18)
            ax.set_xlabel("round", fontsize=18)

            ax.set_ylim(ymin=0)
            fig.savefig(path_folder_save_plot + "/" + f'percentage_rna_match_reference_is_{ref_round}_{pos}')
            plt.show()

            fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
            fig.suptitle(f'nb_rna_match, reference is {ref_round}_{pos}', fontsize=30, x=0.1, ha='left')
            ax.plot(x_list, nb_rna_match)
            ax.set_ylabel(f"number of rna match  colocalize from {ref_round}_{pos}", fontsize=18)
            ax.set_xlabel("round", fontsize=18)
            ax.set_ylim(ymin=0)
            fig.savefig(path_folder_save_plot + "/" + f'nb_rna_match_{ref_round}_{pos}')
            plt.show()

            nb_rna_match_aggregated.append(nb_rna_match)
            percentage_rna_match_aggregated.append(percentage_rna_match)
            median_relocalization_distance_aggregated.append(median_relocalization_distance)

        nb_rna_match_aggregated = np.sum(nb_rna_match_aggregated, axis = 0)
        percentage_rna_match_aggregated = np.mean(np.array(percentage_rna_match_aggregated), axis = 0)
        median_relocalization_distance_aggregated = np.mean(median_relocalization_distance_aggregated, axis = 0)


        fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
        fig.suptitle(f'median_relocalization_distance, reference is {ref_round}', fontsize=30, x=0.1, ha='left')
        ax.plot(x_list[1:], median_relocalization_distance_aggregated[1:])
        ax.set_ylabel("median relocalization distance  in [nm]", fontsize=18)
        ax.set_xlabel("round", fontsize=18)
        fig.savefig(path_folder_save_plot + "/" + f'aggregated_median_relocalization_distance_reference_{ref_round}')

        plt.show()

        fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
        fig.suptitle(f' aggregated, percentage_rna_match, reference is {ref_round}', fontsize=30, x=0.1, ha='left')
        ax.plot(x_list, percentage_rna_match_aggregated)
        ax.set_ylabel(f"aggregated, percentage of rna colocalize from {ref_round} ", fontsize=18)
        ax.set_xlabel("round", fontsize=18)
        ax.set_ylim(ymin=0)
        fig.savefig(path_folder_save_plot + "/" + f'aggregated_percentage_rna_match_reference_is_{ref_round}')
        plt.show()

        fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
        fig.suptitle(f' aggregated, nb_rna_match, reference is {ref_round}', fontsize=30, x=0.1, ha='left')
        ax.plot(x_list, nb_rna_match_aggregated)
        ax.set_ylabel(f"number of rna match  colocalize from {ref_round}", fontsize=18)
        ax.set_xlabel("round", fontsize=18)
        ax.set_ylim(ymin=0)
        fig.savefig(path_folder_save_plot + "/" + f'aggregated_nb_rna_match_{ref_round}')
        plt.show()


        ########
        # signal quality ploting
        #######
        from utils.signal_quality import compute_signal_quality

        ref_round = args.ref_round
        dico_signal_quality = compute_signal_quality(
                               final_spots_detection = final_spots_detection,
                               dico_matched_rna = dico_matched_rna,
                               path_folder_img=args.rounds_folder,
                               channel=args.spots_channel,
                               ref_round= args.ref_round,
                               voxel_size=list(args.voxel_size),
                               spot_radius=(400, 300, 300)
                                )

        ## plot medain snr per round
        round_list_sorted = sorted(list(dico_signal_quality.keys()), key=lambda kv: int(kv[1:]))[:]
        aggregated_snr = []
        aggregated_intensity = []
        aggregated_median_background = []
        for pos in dico_signal_quality[ref_round].keys():
            median_snr = []
            median_intensity = []
            median_background = []
            for r in round_list_sorted:
                median_snr.append(dico_signal_quality[r][pos]['snr_median'])
            median_intensity = []
            for r in round_list_sorted:
                median_intensity.append(np.median(dico_signal_quality[r][pos]['intensity_list']))
            for r in round_list_sorted:
                median_background.append(np.median(dico_signal_quality[r][pos]['median_background']))

            fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
            fig.suptitle(f'median_snr_ reference is {ref_round}, pos is {pos}', fontsize=30, x=0.1, ha='left')
            ax.plot(round_list_sorted, median_snr)
            ax.set_ylabel("median_snr", fontsize=18)
            ax.set_xlabel("round", fontsize=18)
            ax.set_ylim(ymin=0)
            fig.savefig(path_folder_save_plot + "/" + f'median_snr_{ref_round}_{pos}')
            plt.show()
            aggregated_snr.append(median_snr)


            fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
            fig.suptitle(f'median_intensity, reference is {ref_round}, pos is {pos}', fontsize=30, x=0.1, ha='left')
            ax.plot(round_list_sorted, median_intensity)
            ax.set_ylabel("median_intensity value", fontsize=18)
            ax.set_xlabel("round", fontsize=18)
            ax.set_ylim(ymin=0)
            fig.savefig(path_folder_save_plot + "/" + f'median_intensity{ref_round}_{pos}')
            plt.show()
            aggregated_intensity.append(median_intensity)

            aggregated_median_background.append(median_background)



        aggregated_snr = np.mean(np.array(aggregated_snr), axis = 0)
        aggregated_intensity = np.mean(np.array(aggregated_intensity), axis = 0)
        aggregated_median_background = np.mean(np.array(aggregated_median_background), axis = 0)



        fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
        fig.suptitle(f'aggregated_snr, reference is {ref_round} aggreagted on all position', fontsize=30, x=0.1, ha='left')
        ax.plot(round_list_sorted, aggregated_snr)
        ax.set_ylabel("median_snr value", fontsize=18)
        ax.set_xlabel("round", fontsize=18)
        ax.set_ylim(ymin=0)
        fig.savefig(path_folder_save_plot + "/" + f'aggregated_snr')
        plt.show()


        fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
        fig.suptitle(f'aggregated_intensity, reference is {ref_round} aggreagted on all position', fontsize=30, x=0.1, ha='left')
        ax.plot(round_list_sorted, aggregated_intensity)
        ax.set_ylabel("median_intensity value", fontsize=18)
        ax.set_xlabel("round", fontsize=18)
        ax.set_ylim(ymin=0)
        fig.savefig(path_folder_save_plot + "/" + f'aggregated_intensity')
        plt.show()




        fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
        fig.suptitle(f'median background', fontsize=30, x=0.1, ha='left')
        ax.plot(round_list_sorted, aggregated_median_background)
        ax.set_ylabel("median_bacground_intensity value", fontsize=18)
        ax.set_xlabel("round", fontsize=18)
        ax.set_ylim(ymin=0)
        fig.savefig(path_folder_save_plot + "/" + f'aggregated_median_background')
        plt.show()


       #### threshold plots


        pos_list = list(final_spots_detection[round_list_sorted[0]].keys())

        dico_pos_threshold = {}
        for pos in pos_list:
            dico_pos_threshold[pos] = []
            for round_name in round_list_sorted:
                dico_pos_threshold[pos].append(final_spots_detection[round_name][pos]['threshold'])

            fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
            fig.suptitle(f'threshold for position {pos} on log filter image', fontsize=30, x=0.1, ha='left')
            ax.plot(round_list_sorted, dico_pos_threshold[pos])
            ax.set_ylabel(f"threshold", fontsize=18)
            ax.set_xlabel("round", fontsize=18)
            ax.set_ylim(ymin=0)
            fig.savefig(path_folder_save_plot + "/" + f'threshold{pos}')
            plt.show()


        #RNA PER CELL
        ## GET A LIST [NB RNA CELL_I, ....]

        dico_pos_RNA = {}
        for pos in pos_list:
            dico_pos_RNA[pos] = {}
            for round_name in round_list_sorted:
                path_mask = list(Path(f'{args.rounds_folder }/{round_name}/cyto_mask2D_3dim/').glob(f'r*{pos}*.tif*'))
                print(path_mask)
                assert len(path_mask) == 1
                mask_cyto = tifffile.imread(path_mask)
                list_cyto = np.unique(mask_cyto)[1:]
                dico_pos_RNA[pos][round_name] = []
                list_nb_rna_cyto = []
                for cyto in list_cyto:
                    nb_rna_cyto_ind = 0
                    mask_cyto_ind = (mask_cyto == cyto).astype(int)
                    for s in final_spots_detection[round_name][pos]['raw_spots']:
                        nb_rna_cyto_ind+= mask_cyto_ind[0, s[1], s[2]]
                    list_nb_rna_cyto.append(nb_rna_cyto_ind)
                dico_pos_RNA[pos][round_name] = list_nb_rna_cyto

            import seaborn as sns
            import pandas as pd

            fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()

            ax = sns.boxplot(data=pd.DataFrame(dico_pos_RNA[pos]),
                        orient = 'v', ax = ax)
            ax.set_ylabel(f"nb rna per cell", fontsize=24)
            ax.set_title(f"nb rna per cell for position {pos}", fontsize=35)
            ax.get_figure().savefig(path_folder_save_plot + "/" +f'rna_per_cell_boxplot_{pos}.png')
            plt.show()


            ###### plot pair rna

    if args.pairing_plots_spots:
        plot_beads_matching_image_folder(
            first_regex_check="r",
            channel="ch0",
            folder_patho="/media/tom/T7/2023-01-19-PAPER-20-rounds/round_pair/",
            psf="",
            min_distance="",
            round_source="r2",
            dico_transform_spots=all_dico_matched_rna,
            final_spots_detection=final_spots_detection,
            radius=6,
            linewidth=1,
            fill=False,
            figsize=(20, 20),
            fontsize_legend=8,
            plot_mode="rna",
            rescale=True,
            name_file = "spots")


    ########### compute registation with phase corelation


    ## compute the number of pair RNA per round



    ##########  plot snr / intensité




    ######## plot precision de co-localisation


    ### compute supixel