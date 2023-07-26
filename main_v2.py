

import numpy as np

import pandas as pd

from pairing import compute_pair
from tqdm import tqdm

from pathlib import Path

import matplotlib.pyplot as plt







if __name__ == "__main__":



    dict_spots_registered_df_with_cell = np.load("/media/tom/Transcend/autofish/2023-07-04_AutoFISH-SABER/dict_spots_registered_df_r1_with_cell.npy", allow_pickle=True).item()


    # KIF1C 2023-06-28
    #list_round = ["r3", "r6", "r8", "r10", "r12", "r14", "r16", "r18", "r20"]
    #ref_round = "r3"
    #XPO1 2023-06-28
    list_round = ["r2",  "r5", "r8", "r10"]
    ref_round = "r2"
    list_position = dict_spots_registered_df_with_cell.keys()

    dict_distance_matching = {}
    dict_spots_df = {}


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

            df_matching = pd.concat([df_matching, sp1_df[np.isin(sp1_df.index, sp1_index[list_couple_index_sp1])]])
            #breakkkkk
        dict_spots_df[pos] = df_matching



    ### plot histogram of distance
    path_folder_save_plot = "/media/tom/Transcend/autofish/2023-07-04_AutoFISH-SABER/plot_distance_matching/kif1c"

    Path(path_folder_save_plot).mkdir(parents=True, exist_ok=True)

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
            #ref_round = dico_matched_rna[pos][round]['ref_round']
            percentage_rna_match.append(len(dict_distance_matching[pos][round])
                                /
                                len(dict_distance_matching[pos][ref_round]) * 100)
            median_relocalization_distance.append(np.median(dict_distance_matching[pos][round]))


        ### median relocalization distance
        fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
        fig.suptitle(f'median_relocalization_distance, reference is {ref_round}', fontsize=30, x=0.1, ha='left')
        ax.plot(x_list[1:], median_relocalization_distance[1:])
        ax.set_ylabel("median relocalization distance  in [nm]", fontsize=18)
        ax.set_xlabel("round", fontsize=18)
        fig.savefig(path_folder_save_plot + "/" + f'median_relocalization_distance_reference_{ref_round}_{pos}')
        plt.show()


        ### percentage of match cell

        fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
        fig.suptitle(f'percentage_rna_match, reference is {ref_round}_{pos}', fontsize=30, x=0.1, ha='left')
        ax.plot(x_list, percentage_rna_match)
        ax.set_ylabel(f"percentage of rna colocalize from {ref_round}_{pos}", fontsize=18)
        ax.set_xlabel("round", fontsize=18)

        ax.set_ylim(ymin=0)
        fig.savefig(path_folder_save_plot + "/" + f'percentage_rna_match_reference_is_{ref_round}_{pos}')
        plt.show()



        ### nb_rna_match
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

        dict_median_relocalization_distance[pos] = median_relocalization_distance



    nb_rna_match_aggregated = np.sum(nb_rna_match_aggregated, axis = 0)
    percentage_rna_match_aggregated = np.mean(np.array(percentage_rna_match_aggregated), axis = 0)
    median_relocalization_distance_aggregated = np.mean(median_relocalization_distance_aggregated, axis = 0)


    ### median relocalization distance pool

    fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
    fig.suptitle(f'median_relocalization_distance, reference is {ref_round}', fontsize=30, x=0.1, ha='left')
    ax.plot(x_list[1:], median_relocalization_distance_aggregated[1:])
    ax.set_ylabel("median relocalization distance  in pixel", fontsize=18)
    ax.set_xlabel("round", fontsize=18)
    fig.savefig(path_folder_save_plot + "/" + f'aggregated_median_relocalization_distance_reference_{ref_round}')
    plt.show()

    import seaborn as sns
    import pandas as pd

    fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()

    ax = sns.boxplot(data=pd.DataFrame(dict_median_relocalization_distance),
                     orient='v', ax=ax)
    ax.set_ylabel("median relocalization distance  in pixel", fontsize=18)
    fig.suptitle(f'median_relocalization_distance, reference is {ref_round}', fontsize=30, x=0.1, ha='left')
    fig.savefig(path_folder_save_plot + "/" + f'aggregated_median_relocalization_distance_reference__boxplot{ref_round}')
    plt.show()

    ### percentage of match cell pool

    fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
    fig.suptitle(f'median_relocalization_distance, reference is {ref_round}', fontsize=30, x=0.1, ha='left')
    ax.plot(x_list, percentage_rna_match_aggregated)
    ax.set_ylabel(f"aggregated, percentage of rna colocalize from {ref_round} ", fontsize=18)
    ax.set_xlabel("round", fontsize=18)
    ax.set_ylim(ymin=0)
    fig.savefig(path_folder_save_plot + "/" + f'aggregated_percentage_rna_match_reference_is_{ref_round}')
    plt.show()

    ### nb_rna_match pool

    fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()
    fig.suptitle(f' aggregated, nb_rna_match, reference is {ref_round}', fontsize=30, x=0.1, ha='left')
    ax.plot(x_list, nb_rna_match_aggregated)
    ax.set_ylabel(f"number of rna match  colocalize from {ref_round}", fontsize=18)
    ax.set_xlabel("round", fontsize=18)
    ax.set_ylim(ymin=0)
    fig.savefig(path_folder_save_plot + "/" + f'aggregated_nb_rna_match_{ref_round}')
    plt.show()

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

        ################" plot the number of rna matched per cell per ruond
        import seaborn as sns
        import pandas as pd

        fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()

        ax = sns.boxplot(data=pd.DataFrame(dico_pos_RNA),
                         orient='v', ax=ax)
        ax.set_ylabel(f"nb rna per cell", fontsize=24)
        ax.set_title(f"nb rna per cell for position {pos}", fontsize=35)
        ax.get_figure().savefig(path_folder_save_plot + "/" + f'rna_per_cell_boxplot_{pos}.png')
        plt.show()




    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import tifffile

    fig, ax = plt.subplots(figsize=(16, 10))  # plotting density plot for carat using distplot()

    ax = sns.boxplot(data=pd.DataFrame(dico_all_pos_RNA),
                     orient='v', ax=ax, medianprops={"color": "coral"},     boxprops={"facecolor": (.4, .6, .8, .5)},)
    ax.set_ylabel(f"nb rna per cell", fontsize=24)
    ax.set_title(f"nb rna per cell for  all position ", fontsize=35)
    ax.get_figure().savefig(path_folder_save_plot + "/" + f'rna_per_cell_boxplot_all_position.png')
    ax.set_ylim(ymin=0, ymax=600)
    plt.show()



    ##" plot intensity variation per spots position per round

    dico_spots_not_registered = np.load('/media/tom/Transcend/autofish/2023-07-04_AutoFISH-SABER/dict_spots.npy',
                                        allow_pickle=True).item()




    dict_spots_registered_df_with_cell = np.load("/media/tom/Transcend/autofish/2023-07-04_AutoFISH-SABER/dict_spots_registered_df_r1_with_cell.npy",
                                                 allow_pickle=True).item()

    dict_translation = np.load('/media/tom/Transcend/autofish/2023-07-04_AutoFISH-SABER/dict_translation_ref_r1.npy',
                                 allow_pickle=True).item()
    ref_round = 'r1'

    folder_of_rounds = "/media/tom/Transcend/autofish/2023-07-04_AutoFISH-SABER/"
    path_folder_save_plot = "/media/tom/Transcend/autofish/2023-07-04_AutoFISH-SABER/plots_intensity"
    Path(path_folder_save_plot).mkdir(parents=True, exist_ok=True)
    pos = 'pos0'
    df_spots = dict_spots_registered_df_with_cell[pos]
    sp0_ref = df_spots[df_spots['round'] == ref_round]
    sp0_ref = df_spots[df_spots['round'] == ref_round]
    df_matching = pd.DataFrame(columns=df_spots.columns)
    sp0_ref = list(zip(sp0_ref['z'], sp0_ref['y'], sp0_ref['x']))


    #round_to_take = ["r0", "r2",  "r4", "r8", 'r10']
    #round_to_take = ["r0","r1", "r3",  "r4","r7", "r9"]
    empty_round = ["r4"]
    dico_pos_RNA_intensity = {}
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
        dico_pos_RNA_intensity[round_name] = intensity_list

        print(round_name, np.mean(intensity_list))

        #list_round_sorted =sorted(list(dico_pos_RNA_intensity.keys()), key=lambda kv: int(kv[1:]))[:]



    list_round_sorted = round_to_take

    fig, ax = plt.subplots(figsize=(20, 20))  # plotting density plot for carat using distplot()
    df = pd.concat([pd.DataFrame(dico_pos_RNA_intensity[k]) for k in list_round_sorted], axis=1)
    bplot1  = ax.boxplot(df.dropna().values,  patch_artist=True)
    # add colors to the boxplots columns
    #colors_list_random_exa = ['#FF0000', '#FFA500', '#FFFF00', '#008000', '#0000FF', '#4B0082', '#EE82EE'
    #                            , '#FFC0CB', '#000000', '#808080', '#808000', '#00FFFF', '#00FF00', '#800000',
    #                            '#800080', '#008080', '#000080', '#FF00FF', '#C0C0C0', '#FFD700', '#A52A2A', '#00FA9A']
    colors_list_random_exa = ["grey", "blue", "red", "grey", "blue", "red"]
    for patch, color in zip(bplot1 ['boxes'], colors_list_random_exa):
        patch.set_facecolor(color)



    #[pd.DataFrame(dico_pos_RNA_intensity[k]) for k in dico_pos_RNA_intensity]
    #df = pd.concat([pd.DataFrame(dico_pos_RNA_intensity[k]) for k in dico_pos_RNA_intensity], axis=1)
    #ax = sns.boxplot(data=df.item(),
     #                orient='v', ax=ax)
    ax.set_ylim(ymin=0, ymax=4000)
    ax.set_xticklabels(list_round_sorted, fontsize=24)

    ax.set_ylabel(f"intensity ", fontsize=24)
    ax.set_title(f" round at position {pos}", fontsize=35)
    ax.get_figure().savefig(path_folder_save_plot + "/" + f'ref_round_{ref_round}_{pos}.png')
    plt.show()


    ##" plot intensity variation per spots position per round
        for round_name in list_round_sorted:
            intensity_list  = dico_pos_RNA_intensity[round_name]
            print(round_name, np.mean(intensity_list))