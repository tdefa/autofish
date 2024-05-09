


#%%

import numpy as np

from pairing.pairing import compute_pair
from tqdm import tqdm

from pathlib import Path

import matplotlib.pyplot as plt


def match_all_rna_wit_ref_spots(dict_spots_registered_df_with_cell,
                                ref_round,
                                list_round):
    """
    Match all RNA with spots in ref roundfor all position
    Args:
        dict_spots_registered_df_with_cell:
        ref_round:
        list_round:

    Returns:
         dict_distance_matching is a dictionary with the distance between the matched spots
         dict_distance_matching[position][round] = list_distance
         list_distance is a list with the distance between the matched spots use to compute the mean relocalisation distance
         dict_spots_df is a dictionary with the matched spots
            dict_spots_df[position]  = df_spots with the spots coordiandate of all round at the position and the matched distance



    """

    dict_distance_matching = {}
    dict_spots_df = {}

    list_position = dict_spots_registered_df_with_cell.keys()
    ### match all RNA WITH REF SPOTS
    for pos in list_position:
        df_spots = dict_spots_registered_df_with_cell[pos]
        sp0_ref = df_spots[df_spots['round'] == ref_round]
        df_matching =  pd.DataFrame(columns =  df_spots.columns)
        sp0_ref = list(zip(sp0_ref['z'], sp0_ref['y'], sp0_ref['x']))
        dict_distance_matching[pos] = {}
        for r in list_round:
            sp1_df = df_spots[df_spots['round'] == r]
            sp1_index = np.array(sp1_df.index.to_list())
            sp1 = list(zip(sp1_df['z'], sp1_df['y'], sp1_df['x']))
            list_couple_index_sp0_ref, list_couple_index_sp1, sp0_ref_pair_order,  sp1_pair_order, list_distance = compute_pair(sp0_ref,
                                                                                                                                sp1, max_distance=15)
            dict_distance_matching[pos][r] = list_distance

            df_matched_spots = sp1_df[np.isin(sp1_df.index, sp1_index[list_couple_index_sp1])]
            df_matched_spots["distance"] = list_distance

            df_matching = pd.concat([df_matching, df_matched_spots])
            #breakkkkk
        dict_spots_df[pos] = df_matching
    return dict_distance_matching, dict_spots_df


def compute_aggregated_distance(dict_distance_matching,
                                list_position,
                                list_round,
                                ref_round):
    """

    Args:
        dict_distance_matching:
        list_position:
        list_round:
        ref_round:

    Returns:
        result accros round ,  aggregated accross all position,
        nb_rna_match_aggregated : the number of matched RNA,
         percentage_rna_match_aggregated : the percentage of matched RNA,
         median_relocalization_distance_aggregated : the median relocalization distance
         dict_median_relocalization_distance : the median relocalization distance for each round at the position used as key

    """
    nb_rna_match_aggregated = []
    percentage_rna_match_aggregated = []
    median_relocalization_distance_aggregated = []
    dict_median_relocalization_distance = {}
    for pos in list_position:
        nb_rna_match = []
        percentage_rna_match = []
        median_relocalization_distance = []
        x_list = []
        for round in list_round:
            if len(dict_distance_matching[pos][round]) == 0:
                continue
            x_list.append(round)
            nb_rna_match.append(len(dict_distance_matching[pos][round]))
            # ref_round = dico_matched_rna[pos][round]['ref_round']
            percentage_rna_match.append(len(dict_distance_matching[pos][round])
                                        /
                                        len(dict_distance_matching[pos][ref_round]) * 100)
            median_relocalization_distance.append(np.median(dict_distance_matching[pos][round]))
        nb_rna_match_aggregated.append(nb_rna_match)
        percentage_rna_match_aggregated.append(percentage_rna_match)
        median_relocalization_distance_aggregated.append(median_relocalization_distance)
        dict_median_relocalization_distance[pos] = median_relocalization_distance

    nb_rna_match_aggregated = np.sum(nb_rna_match_aggregated, axis=0)
    percentage_rna_match_aggregated = np.mean(np.array(percentage_rna_match_aggregated), axis=0)
    median_relocalization_distance_aggregated = np.mean(median_relocalization_distance_aggregated, axis=0)

    return nb_rna_match_aggregated, percentage_rna_match_aggregated, median_relocalization_distance_aggregated, dict_median_relocalization_distance


def compute_nb_rna_per_cell(dict_spots_df,
                            list_position,
                            list_round):
    """

    Args:
        dict_spots_df:
        list_position:
        list_round:

    Returns:

        dict_all_pos_RNA : the number of RNA per cell per round for all positions aggregated

    """
    dico_all_pos_RNA = {r : [] for r in list_round}
    for pos in tqdm(list_position):
        df_matching = dict_spots_df[pos]
        unique_cell = df_matching.cell_assignment.unique()
        if -1 in unique_cell:
            unique_cell = unique_cell[:-1]
        if 0 in unique_cell:
            unique_cell = unique_cell[1:]
        dico_pos_RNA = {}
        for round_name in list_round:
            list_nb_rna_cyto = []
            for cell in unique_cell:
                nb_rna_cyto_ind = df_matching[(df_matching.cell_assignment == cell) & (df_matching['round'] == round_name)]
                list_nb_rna_cyto.append(len(nb_rna_cyto_ind))
            dico_pos_RNA[round_name] = list_nb_rna_cyto
            dico_all_pos_RNA[round_name] += list_nb_rna_cyto
    return dico_all_pos_RNA, dico_pos_RNA



def compute_rna_intensity(dict_spots_registered_df_with_cell,
                            round_to_take,
                          dict_translation,
                          ref_round,
                          folder_of_rounds,
                          ):
    dico_pos_RNA_intensity = {}
    for r in round_to_take:
        dico_pos_RNA_intensity[r] = []
    list_position = list(dict_spots_registered_df_with_cell.keys())
    for pos in tqdm(list_position):
        df_spots = dict_spots_registered_df_with_cell[pos]
        for round_name in tqdm(round_to_take):
            if round_name == ref_round:
                x_translation = 0
                y_translation = 0
            else:
                x_translation = dict_translation[pos][ref_round][round_name]['x_translation']
                y_translation = dict_translation[pos][ref_round][round_name]['y_translation']
            sp_df = df_spots[df_spots['round'] == ref_round]
            sp0_ref = list(zip(sp_df['z'], sp_df['y'], sp_df['x']))
            translated_spots = sp0_ref + np.array([0, y_translation, x_translation])
            fish_signal = tifffile.imread(Path(folder_of_rounds) / f"{round_name}/{round_name}_{pos}_ch0.tif")
            z_range = range(0, fish_signal.shape[0])
            y_range = range(0, fish_signal.shape[1])
            x_range = range(0, fish_signal.shape[2])
            intensity_list = []
            for spot in translated_spots:
                z, y, x = round(spot[0]), round(spot[1]), round(spot[2])
                if z in z_range and y in y_range and x in x_range:
                    intensity_list.append(fish_signal[z, y, x])
            dico_pos_RNA_intensity[round_name] += intensity_list
    return dico_pos_RNA_intensity




def compute_snr(dict_signal_quality,
                round_list_sorted,
                ):
    aggregated_snr = []
    dict_median_snr = {}
    dict_snr = {}
    aggregated_intensity = []
    dict_median_intensity = {}
    dict_intensity = {}
    aggregated_median_background = []
    dict_median_background = {}
    dict_background = {}


    for r in round_list_sorted:
            dict_median_snr[r] = []
            dict_median_intensity[r] = []
            dict_median_background[r] = []
            dict_snr[r] = []
            dict_intensity[r] = []
            dict_background[r] = []

    pos_list = dict_signal_quality[list(dict_signal_quality.keys())[0]].keys()

    for pos in pos_list:
        median_snr = []
        median_background = []
        median_intensity = []
        for r in round_list_sorted:
            median_snr.append(np.median(dict_signal_quality[r][pos]['snr']))
            dict_median_snr[r].append(np.median(dict_signal_quality[r][pos]['snr']))
            dict_snr[r] = dict_signal_quality[r][pos]['snr']
        for r in round_list_sorted:
            median_intensity.append(np.median(dict_signal_quality[r][pos]['intensity']))
            dict_median_intensity[r].append(np.median(dict_signal_quality[r][pos]['intensity']))
            dict_intensity[r] = dict_signal_quality[r][pos]['intensity']
        for r in round_list_sorted:
            median_background.append(np.median(dict_signal_quality[r][pos]['background']))
            dict_median_background[r].append(np.median(dict_signal_quality[r][pos]['background']))
            dict_background[r] = dict_signal_quality[r][pos]['background']
        aggregated_intensity.append(median_intensity)
        aggregated_median_background.append(median_background)
        aggregated_snr.append(median_snr)
    aggregated_snr = np.mean(np.array(aggregated_snr), axis=0)
    aggregated_intensity = np.mean(np.array(aggregated_intensity), axis=0)
    aggregated_median_background = np.mean(np.array(aggregated_median_background), axis=0)

    return aggregated_snr, aggregated_intensity, aggregated_median_background, dict_median_snr









#%%

if __name__ == "__main__":
#%%

    plt.rcParams['xtick.labelsize'] = 35
    plt.rcParams['ytick.labelsize'] = 35
    plt.rc_context({"axes.labelsize": 35, })



    #dict_spots_registered_df_with_cell = np.load("/media/tom/Transcend/autofish/2023-06-28_AutoFISH_22rounds/dict_spots_registered_df_r1_with_cell.npy",
     #                                            allow_pickle=True).item()
    dict_spots_registered_df_with_cell = np.load("/media/tom/Transcend/autofish/2023-06-28_AutoFISH_22rounds/dict_spots_registered_df_assigned_050424_mattes.npy",
                                                allow_pickle=True).item()

    # KIF1C 2023-06-28
    list_round_kif1c = ["r3", "r6", "r8", "r10", "r12", "r14", "r16", "r18", "r20", 'r22']
    ref_round_kif1c = "r3"
    #XPO1 2023-06-28
    list_round_xpo1 = ["r1",  "r5", "r7", "r9", 'r11', "r13", "r15", "r17", "r19", "r21"]
    ref_round_xpo1 = "r1"
    list_position = dict_spots_registered_df_with_cell.keys()

    dict_distance_matching_kif1c, dict_spots_df_kif1c = match_all_rna_wit_ref_spots(dict_spots_registered_df_with_cell,
                                ref_round = ref_round_kif1c,
                                list_round = list_round_kif1c)

    dict_distance_matching_xpo1, dict_spots_df_xpo1 = match_all_rna_wit_ref_spots(dict_spots_registered_df_with_cell,
                                ref_round = ref_round_xpo1,
                                list_round = list_round_xpo1)

    nb_rna_match_aggregated_kif1c, percentage_rna_match_aggregated_kif1c, median_relocalization_distance_aggregated_kif1c, \
        dict_median_relocalization_distance_kif1c = compute_aggregated_distance(
        dict_distance_matching = dict_distance_matching_kif1c,
                                list_position = list_position,
                                list_round = list_round_kif1c,
                                ref_round = ref_round_kif1c)

    nb_rna_match_aggregated_xpo1, percentage_rna_match_aggregated_xpo1, median_relocalization_distance_aggregated_xpo1, \
        dict_median_relocalization_distance_xpo1 = compute_aggregated_distance(
        dict_distance_matching = dict_distance_matching_xpo1,
                                list_position = list_position,
                                list_round = list_round_xpo1,
                                ref_round = ref_round_xpo1)



    path_folder_save = "/media/tom/Transcend/autofish/png_for_figure"
    list_round = list(range(3,21))
    list_round_int_xpo1 = list(range(3, 21, 2))
    list_round_int_kif1c = list(range(4, 21, 2))

    COLOR_XPO1 = '#fc8d62'
    COLOR_KIF1C = '#8da0cb'


    plt.rcParams['xtick.labelsize'] = 55
    plt.rcParams['ytick.labelsize'] = 55
    plt.rc_context({"axes.labelsize": 55, })


    ############################
    ### median relocalization distance
    ############################


    fig, ax = plt.subplots(figsize=(16, 12))  # plotting density plot for carat using distplot()
    #fig.suptitle(f'median_relocalization_distance, reference is {ref_round_xpo1}', fontsize=30, x=0.1, ha='left')
    # set line color in green with point at each round
    ax.set_xticklabels(list_round)
    ax.plot(list_round_int_xpo1, median_relocalization_distance_aggregated_xpo1[1:] * 108, color=COLOR_XPO1,
            marker='o', ## set spot and line size
            markersize=30,
            linewidth=6)


    ax.plot(list_round_int_kif1c, median_relocalization_distance_aggregated_kif1c[1:]* 108,  color=COLOR_KIF1C,
            marker='o', ## set spot and line size
            markersize=30,
            linewidth=6)

    #ax.set_ylabel("median relocalization distance  in nm", fontsize=18)
    #ax.set_xlabel("round", fontsize=18)

    ## set y axis ticks
    ax.set_yticks(np.arange(50, 400, 50))
    list_round_str = [f'round {i}' for i in range(3, 20)]

    ax.set_xticks(list_round)
    ax.set_xticklabels(list_round, rotation=45)
    ## rotate x axis label
    filename = Path(path_folder_save) / f'median_relocalization_distance_reference_in_nm'
    image_format = 'svg'  # e.g .png, .svg, etc.
    file_name = f'{filename}. {image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    image_format = 'png'  # e.g .png, .svg, etc.
    file_name = f'{filename}. {image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')

    plt.show()



    ############################
    ### percentage of matched RNA
    ############################


    fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
    #fig.suptitle(f'median_relocalization_distance, reference is {ref_round_xpo1}', fontsize=30, x=0.1, ha='left')
    # set line color in green with point at each round


    ax.plot(list_round_int_kif1c, percentage_rna_match_aggregated_kif1c[:],  color=COLOR_XPO1,
            marker='o', ## set spot and line size
            markersize=30,
            linewidth=6)

    ax.plot(list_round_int_xpo1, percentage_rna_match_aggregated_xpo1[:], color=COLOR_KIF1C,
            marker='o', ## set spot and line size
            markersize=30,
            linewidth=6)


    #ax.set_ylabel("median relocalization distance  in nm", fontsize=18)
    #ax.set_xlabel("round", fontsize=18)

    ## set y axis ticks
    #ax.set_yticks(np.arange(50, 400, 50))
    ax.set_yticks(np.arange(75, 105, 5))

    list_round_str = [f'round {i}' for i in range(3, 20)]

    ax.set_xticks(list_round)
    ax.set_xticklabels(list_round, rotation=45)
    ## rotate x axis label
    filename = Path(path_folder_save)  / f'percentage_of_matched_rna'
    image_format = 'svg'  # e.g .png, .svg, etc.
    file_name = f'{filename}. {image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    image_format = 'png'  # e.g .png, .svg, etc.
    file_name = f'{filename}. {image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    plt.show()

    ############################
    ### number of matched RNA
    ############################

    dict_all_pos_RNA_xpo1, dict_pos_RNA_xpo1 = compute_nb_rna_per_cell(
        dict_spots_df = dict_spots_df_xpo1,
        list_position = list_position,
        list_round = list_round_xpo1,
    )

    dict_all_pos_RNA_kif1c, dict_pos_RNA_kif1c = compute_nb_rna_per_cell(dict_spots_df = dict_spots_df_kif1c,
                            list_position = list_position,
                            list_round = list_round_kif1c)





    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import tifffile



    ### XPO1
    fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
    df_nb_cell = pd.DataFrame(dict_all_pos_RNA_xpo1)

    ## rename all columns without the first latter
    df_nb_cell.columns =["1"] + [str(i) for i in list_round_int_xpo1]


    ax = sns.boxplot(data=df_nb_cell,
                     color = COLOR_XPO1)
    ax.set_ylim(ymin=0, ymax=450)


    ## rotate x axis label
    filename = Path(path_folder_save)  / f'rna_per_cell_boxplot_all_position_XPO1'
    image_format = 'svg'  # e.g .png, .svg, etc.
    file_name = f'{filename}.{image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    image_format = 'png'  # e.g .png, .svg, etc.
    file_name = f'{filename}.{image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    plt.show()


    ### KIF1C
    fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
    df_nb_cell = pd.DataFrame(dict_all_pos_RNA_kif1c)

    ## rename all columns without the first latter
    df_nb_cell.columns = [str(i) for i in list_round_int_kif1c] + ["20"]



    ax = sns.boxplot(data=df_nb_cell,
                     color = COLOR_XPO1)
    ax = sns.boxplot(data=pd.DataFrame(df_nb_cell),
                     color = COLOR_KIF1C)
    ax.set_ylim(ymin=0, ymax=250)
    ## rotate x axis label
    filename = Path(path_folder_save)  / f'rna_per_cell_boxplot_all_position_KIF1C'
    image_format = 'svg'  # e.g .png, .svg, etc.
    file_name = f'{filename}.{image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    image_format = 'png'  # e.g .png, .svg, etc.
    file_name = f'{filename}.{image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    plt.show()


    #########################
    # signal quality metrics
    ########################"""

    ######## intensity to check would it be possible to use the intensity before registration
    # for now I have to un-registrate the spots coordiante to get the intensity
    ## IT IS ERROR PRONE
    # TO FINISH


    dico_spots_not_registered = np.load('/media/tom/Transcend/autofish/2023-06-28_AutoFISH_22rounds/' +
                                        '18jully_dico_spots_local_detection0_r_mask_artefact0_1.3_remove_non_sym0.npy',
                                        allow_pickle=True).item()





    #dict_translation = np.load('/media/tom/Transcend/autofish/2023-06-28_AutoFISH_22rounds/dict_translation_ref_r1.npy',
     #                            allow_pickle=True).item()

    dict_translation = np.load("/media/tom/Transcend/autofish/2023-06-28_AutoFISH_22rounds/dict_translation_ref_r1_050424.npy",
                                 allow_pickle=True).item()
    ref_round = 'r1'

    folder_of_rounds = "/media/tom/Transcend/autofish/2023-06-28_AutoFISH_22rounds/"
    path_folder_save_plot = "/media/tom/Transcend/autofish/2023-06-28_AutoFISH_22rounds/plots_intensity"
    Path(path_folder_save_plot).mkdir(parents=True, exist_ok=True)



    dict_pos_RNA_intensity_kif1c = compute_rna_intensity(dict_spots_registered_df_with_cell,
                          round_to_take = list_round_kif1c,
                          dict_translation = dict_translation,
                          ref_round = "r1",
                          folder_of_rounds = folder_of_rounds,
                          )

    dict_pos_RNA_intensity_xpo1 = compute_rna_intensity(
                            dict_spots_registered_df_with_cell,
                            round_to_take = list_round_xpo1,
                            dict_translation = dict_translation,
                            ref_round = "r1",
                            folder_of_rounds = folder_of_rounds,
                            )


    fig, ax = plt.subplots(figsize=(20, 20))  # plotting density plot for carat using distplot()

    df = pd.concat([pd.DataFrame(dict_pos_RNA_intensity_kif1c[k]) for k in list_round_kif1c], axis=1)
    bplot1  = ax.boxplot(df.dropna().values,  ## ADD COLOR
                            patch_artist=True,
                            positions = list(range(2, (len(list_round_kif1c) +1)*2, 2 )),
                            boxprops=dict(facecolor=COLOR_KIF1C),
                            medianprops=dict(color="black"),
                            widths=0.6,
    showfliers = False  # hide outliers

    )

    ## OVERLAP xp01


    df = pd.concat([pd.DataFrame(dict_pos_RNA_intensity_xpo1[k]) for k in list_round_xpo1], axis=1)
    bplot2  = ax.boxplot(df.dropna().values,  ## ADD COLOR
                            patch_artist=True,
                            positions = list(range(1, len(list_round_xpo1) * 2, 2)),
                            boxprops=dict(facecolor=COLOR_XPO1),
                            medianprops=dict(color="black"),
                            widths=0.6,
    showfliers = False  # hide outliers

    )

    ax.set_ylim(ymin=0, ymax=3000)
    #ax.set_xticklabels(list_round, fontsize=24)

    #ax.set_ylabel(f"intensity ", fontsize=24)
    #ax.set_title(f" round poll for all positions", fontsize=35)

    filename = Path(path_folder_save)  / f'intensity_all_round'
    image_format = 'svg'  # e.g .png, .svg, etc.
    file_name = f'{filename}.{image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    image_format = 'png'  # e.g .png, .svg, etc.
    file_name = f'{filename}.{image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    plt.show()



    ########
    # SNR
    ########

    dict_signal_quality = np.load('/media/tom/Transcend/autofish/2023-06-28_AutoFISH_22rounds/dico_signal_quality.npy',
            allow_pickle=True).item()



    aggregated_snr_xpo1, aggregated_intensity_xpo1, aggregated_median_background_xpo1, \
        dict_median_snr_xpo1 = compute_snr(
        dict_signal_quality,
                round_list_sorted = list_round_xpo1,
                )

    aggregated_snr_kif1c, aggregated_intensity_kif1c, aggregated_median_background_kif1c, \
        dict_median_snr_kif1c = compute_snr(
        dict_signal_quality,
                round_list_sorted = list_round_kif1c,
                )

############ SNR
    fig, ax = plt.subplots(figsize=(20, 20))  # plotting density plot for carat using distplot()
    df = pd.concat([pd.DataFrame(dict_median_snr_kif1c[k]) for k in list_round_kif1c], axis=1)
    bplot1  = ax.boxplot(df.dropna().values,  ## ADD COLOR
                            patch_artist=True,
                            positions = list(range(2, (len(list_round_kif1c) +1)*2, 2 )),
                            boxprops=dict(facecolor=COLOR_KIF1C),
                            medianprops=dict(color="black"),
                            widths=0.6,
    showfliers = False  # hide outliers

    )
    ## OVERLAP xp01
    df = pd.concat([pd.DataFrame(dict_median_snr_xpo1[k]) for k in list_round_xpo1], axis=1)
    bplot2  = ax.boxplot(df.dropna().values,  ## ADD COLOR
                            patch_artist=True,
                            positions = list(range(1, len(list_round_xpo1) * 2, 2)),
                            boxprops=dict(facecolor=COLOR_XPO1),
                            medianprops=dict(color="black"),
                            widths=0.6,
    showfliers = False  # hide outliers
    )
    ax.set_ylim(ymin=2, ymax=12)
    ax.set_xticks(list(range(20)))
    ax.set_xticklabels(labels=list(range(20)),
                       rotation=45)
    filename = Path(path_folder_save)  / f'snr'
    image_format = 'svg'  # e.g .png, .svg, etc.
    file_name = f'{filename}.{image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    image_format = 'png'  # e.g .png, .svg, etc.
    file_name = f'{filename}.{image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    plt.show()

    #ax.set_xticklabels(list_round, fontsize=24)

    #ax.set_ylabel(f"intensity ", fontsize=24)
    #ax.set_title(f" round poll for all positions", fontsize=35)

##### SNR AGGREGATED


    fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
    ax.set_xticklabels(list_round)
    ax.plot([1] + list_round_int_xpo1, aggregated_snr_xpo1[:], color=COLOR_XPO1,
            marker='o',  ## set spot and line size
            markersize=30,
            linewidth=6)

    ax.plot(list_round_int_kif1c + [20], aggregated_snr_kif1c[:], color=COLOR_KIF1C,
            marker='o',  ## set spot and line size
            markersize=30,
            linewidth=6)

    ax.set_ylim(ymin=0, ymax=10)

    ax.set_xticks(list(range(20)))
    ax.set_xticklabels(list(range(20)), rotation=45)
    ## rotate x axis label
    ## WRITE Y tick in scientific writing


    filename = Path(path_folder_save) / f'SNR_aggregated'
    image_format = 'svg'  # e.g .png, .svg, etc.
    file_name = f'{filename}.{image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    image_format = 'png'  # e.g .png, .svg, etc.
    file_name = f'{filename}.{image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    plt.show()




##### SNR AGGREGATED RATIO


    fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
    ax.set_xticklabels(list_round)
    ax.plot([1] + list_round_int_xpo1, aggregated_intensity_xpo1 /aggregated_median_background_xpo1, color=COLOR_XPO1,
            marker='o',  ## set spot and line size
            markersize=30,
            linewidth=6)

    ax.plot(list_round_int_kif1c + [20], aggregated_intensity_kif1c /aggregated_median_background_kif1c, color=COLOR_KIF1C,
            marker='o',  ## set spot and line size
            markersize=30,
            linewidth=6)

    ax.set_ylim(ymin=0, ymax=3)

    ax.set_xticks(list(range(20)))
    ax.set_xticklabels(list(range(20)), rotation=45)
    ## rotate x axis label
    ## WRITE Y tick in scientific writing


    filename = Path(path_folder_save) / f'SNR_aggregated_ratio'
    image_format = 'svg'  # e.g .png, .svg, etc.
    file_name = f'{filename}.{image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    image_format = 'png'  # e.g .png, .svg, etc.
    file_name = f'{filename}.{image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    plt.show()


##### INTENSITY AGGREGATED to f

    fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
    ax.set_xticklabels(list_round)
    ax.plot([1] + list_round_int_xpo1, aggregated_intensity_xpo1[:] , color=COLOR_XPO1,
            marker='o', ## set spot and line size
            markersize=30,
            linewidth=6)


    ax.plot(list_round_int_kif1c + [20], aggregated_intensity_kif1c[:],  color=COLOR_KIF1C,
            marker='o', ## set spot and line size
            markersize=30,
            linewidth=6)

    ax.set_ylim(ymin=0, ymax=3000)

    ax.set_xticks(list(range(21)))
    ax.set_xticklabels(list(range(21)), rotation=45)
    ## rotate x axis label
    ## WRITE Y tick in scientific writing



    filename = Path(path_folder_save)  / f'intensity_aggregated'
    image_format = 'svg'  # e.g .png, .svg, etc.
    file_name = f'{filename}.{image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    image_format = 'png'  # e.g .png, .svg, etc.
    file_name = f'{filename}.{image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    plt.show()

    ##### BACKGROUND AGGREGATED

    fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
    ax.set_xticklabels(list(range(20)))
    ax.plot([1] + list_round_int_xpo1, aggregated_median_background_xpo1, color=COLOR_XPO1,
            marker='o',  ## set spot and line size
            markersize=30,
            linewidth=6)

    ax.plot(list_round_int_kif1c + [20], aggregated_median_background_kif1c, color=COLOR_KIF1C,
            marker='o',  ## set spot and line size
            markersize=30,
            linewidth=6)
    ax.set_ylim(ymin=0, ymax=3000)

    ax.set_xticks(list(range(21)))
    ax.set_xticklabels(list(range(21)), rotation=45)
    ## rotate x axis label
    ## WRITE Y tick in scientific writing
    plt.show()

    filename = Path(path_folder_save)  / f'background_aggregated'
    image_format = 'svg'  # e.g .png, .svg, etc.
    file_name = f'{filename}.{image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    image_format = 'png'  # e.g .png, .svg, etc.
    file_name = f'{filename}.{image_format}'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    plt.show()





