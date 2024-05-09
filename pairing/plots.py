#############
## FILE FOR PLOT FUNCTION
###############

import numpy as np
import pylab
import seaborn as sns
import tifffile
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from scipy import ndimage as ndi
from scipy.spatial import distance
from skimage.measure import label
import tqdm
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rc_context({"axes.labelsize" : 45,})


##############################"
# Bead plotting
########################"


def plot_beads_detection(beads_signal,
                         rescale = False,
                         list_list_beads =[],
                         list_color = ["green"],
                         list_label = ['rr'],
                         radius = 6,
                         linewidth = 1,
                         fill=False,
                         figsize=(20, 20),
                         fontsize_legend = 8,
                         normal_dim = 0,
                         mask_dapi_contour = None,
                         mask_cyto_contour = None,
                         dapi_signal = None,
                         fish_cmap = 'Reds'):

    """
    function automatically used in the main
    :param beads_signal:
    :type beads_signal:
    :param rescale:
    :type rescale:
    :param list_list_beads: list of list of coordiante of bead : ex [list_matched_bead, list_unmatched_bead]
    :param color:
    :type color:
    :param radius:
    :type radius:
    :param linewidth:
    :type linewidth:
    :param fill:
    :type fill:
    :return:
    :rtype:
    """

    from matplotlib import pyplot as plt
    from skimage.exposure import rescale_intensity
    from matplotlib.patches import Circle

    input = np.amax(beads_signal, normal_dim)
    if rescale:
        pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
        input = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8)

    if mask_dapi_contour is not None:
        fig, ax = plt.subplots(figsize=figsize)
        #ax.imshow(dapi_signal_input, alpha=0.5, cmap='seismic')
        #ax.imshow(input, alpha=0.5,  cmap='Reds')
        ax.imshow(input)
        Zm = np.ma.masked_where(mask_dapi_contour == 0, mask_dapi_contour)
        ax.imshow(Zm, cmap="spring")
        Zm = np.ma.masked_where(mask_cyto_contour == 0, mask_dapi_contour)
        ax.imshow(Zm, cmap="bwr")
    else:
        fig, ax = plt.subplots(figsize=figsize)

        ax.imshow(input)

    custom_legend = []
    if type(radius) == type([]):
        list_radius = radius
    else:
        list_radius = [radius] * len(list_list_beads)
    for ind in range(len(list_list_beads)):
        color = list_color[ind]
        list_beads = list_list_beads[ind]
        if len(list_beads) == 0:
            continue
        for zxy in list_beads:
            ind_spot_c = np.unique(list(set([0,1,2]) -set([normal_dim])))
            s_cirlce = Circle((zxy[ind_spot_c[1]], zxy[ind_spot_c[0]]), radius=list_radius[ind],
                       color=color, linewidth=linewidth,
                       fill=fill)
            ax.add_patch(s_cirlce)
        custom_legend.append(s_cirlce)
        ax.legend(custom_legend, list_label, fontsize=fontsize_legend)
    return fig, ax



def plot_beads_image_folder(dico_detection_beads,
        first_regex_check="r",
                            channel="",
                            min_distance=[],
                            psf=[],
                            folder_patho=[],
                            radius=9,
                            linewidth=2,
                            fill=False,
                            figsize=(20, 20),
                            fontsize_legend=8,
                            folder_name = "image_bead"):


    from pathlib import Path
    from tqdm import tqdm
    folder_patho = Path(folder_patho)
    for folder_round in tqdm(list(folder_patho.glob("*/"))):
        round_name = str(folder_round).split('/')[-1]

        if first_regex_check != round_name[:len(first_regex_check)]:
            continue

        print(round_name)
        path_folder_image = str(folder_round) + f'/{folder_name}_' + str(psf) + f'_min_distance_bead{min_distance}'
        Path(path_folder_image).mkdir(parents=True, exist_ok=True)

        list_image_round = list(folder_round.glob(f"{first_regex_check}*{channel}*"))

        for beads_signal_path in list_image_round:
            beads_signal = tifffile.imread(beads_signal_path)
            pos = str(beads_signal_path).split('/')[-1].split('_')[1]
            list_beads = dico_detection_beads[round_name][str(pos).split('/')[-1]]["raw_spots"]
            fig, ax = plot_beads_detection(beads_signal,
                                           rescale=True,
                                           list_list_beads=[list_beads],
                                           list_color=["green"],
                                           radius=radius,
                                           linewidth=linewidth,
                                           fill=fill,
                                           figsize=figsize,
                                           fontsize_legend = fontsize_legend,
                                           normal_dim = 0)


            fig.savefig(path_folder_image + "/" + str(beads_signal_path)[-12:-8])

            fig, ax = plot_beads_detection(beads_signal,
                                           rescale=False,
                                           list_list_beads=[list_beads],
                                           list_color=["green"],
                                           radius=radius,
                                           linewidth=linewidth,
                                           fill=fill,
                                           figsize=tuple(np.array(figsize) * np.array([10, 0.5])),
                                           fontsize_legend = fontsize_legend,
                                           normal_dim = 1)
            fig.savefig(path_folder_image + "/" + str(beads_signal_path)[-12:-8]+"z_stack1")
            """fig, ax = plot_beads_detection(beads_signal,
                                           rescale=False,
                                           list_list_beads=[list_beads],
                                           list_color=["green"],
                                           radius=radius,
                                           linewidth=linewidth,
                                           fill=fill,
                                           figsize=tuple(np.array(figsize) * np.array([10, 0.5])),
                                           fontsize_legend = fontsize_legend,
                                           normal_dim = 2)
            fig.savefig(path_folder_image + "/" +str(beads_signal_path).split('/')[-1])"""
        plt.close("all")





def plot_beads_matching_image_folder(
    first_regex_check = "r",
    channel = "ch1",
    folder_patho = "/media/tom/T7/2023-01-19-PAPER-20-rounds/test",
    psf = "",
    min_distance = "",
    round_source = "",
    dico_transform_spots = {},
    final_spots_detection = None,
    radius=6,
    linewidth=1,
    fill=False,
    figsize=(20, 20),
    fontsize_legend=8,
    plot_mode = "bead",
    rescale = True,
    name_file = "image_bead"):

    from pathlib import Path
    from tqdm import tqdm
    folder_patho = Path(folder_patho)
    for folder_round in tqdm(list(folder_patho.glob("*/"))):
        round_name = str(folder_round).split('/')[-1]
        if first_regex_check != round_name[:len(first_regex_check)]:
            continue
        print(round_name)
        path_folder_image = str(folder_round) + '/' + name_file
        Path(path_folder_image).mkdir(parents=True, exist_ok=True)
        list_image_pos = list(folder_round.glob(f"{first_regex_check}*{channel}*"))
        for beads_signal_path in list_image_pos:
            beads_signal = tifffile.imread(beads_signal_path)
            pos = str(beads_signal_path).split("/")[-1].split('_')[1]
            if plot_mode == "bead":
                spots_0_colocalized = dico_transform_spots[round_name][round_source][pos]["spots_0_colocalized"]
                new_spots_1_colocalized = dico_transform_spots[round_name][round_source][pos]["new_spots_1_colocalized"]
                spots_1_colocalized = dico_transform_spots[round_source][round_name][pos]["spots_0_colocalized"]
            if plot_mode == "rna":
                spots_0_colocalized = dico_transform_spots[round_name][round_source][pos][ 'sp0_ref_pair_order']

                new_spots_1_colocalized = dico_transform_spots[round_name][round_source][pos]["sp1_pair_order"]
                spots_1_colocalized = final_spots_detection[round_source][pos]['subpixel_spots'][0]

            list_list_beads  = [spots_0_colocalized, new_spots_1_colocalized]
            ##### after registration
            fig, ax = plot_beads_detection(beads_signal,
                                           rescale=rescale,
                                           list_list_beads=list_list_beads,
                                           list_color=["green", "red"],
                                           radius=radius,
                                           linewidth=linewidth,
                                           fill=fill,
                                           figsize=figsize,
                                           fontsize_legend = fontsize_legend,
                                           normal_dim = 0)

            fig.savefig(path_folder_image + "/" + str(beads_signal_path)[-12:-8] +f"after_registration_{round_name}_{round_source}")

            list_list_beads  = [spots_0_colocalized, spots_1_colocalized]

            fig, ax = plot_beads_detection(beads_signal,
                                           rescale=rescale,
                                           list_list_beads=list_list_beads,
                                           list_color=["green", "red"],
                                           radius=radius,
                                           linewidth=linewidth,
                                           fill=fill,
                                           figsize=figsize,
                                           fontsize_legend = fontsize_legend,
                                           normal_dim = 0)

            fig.savefig(path_folder_image + "/" + str(beads_signal_path)[-12:-8] +f"before_registration_{round_name}_{round_source}")

def plot_spots_matching_image_folder(
    first_regex_check = "r",
    channel = "ch1",
    folder_patho = "/media/tom/T7/2023-01-19-PAPER-20-rounds/test",
    psf = "",
    min_distance = "",
    round_source = "",
    dico_transform_spots = {},
    radius=6,
    linewidth=1,
    fill=False,
    figsize=(20, 20),
    fontsize_legend=8,
    plot_mode = "bead",
    rescale = True):
    return