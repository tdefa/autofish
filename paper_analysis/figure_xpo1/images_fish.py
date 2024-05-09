

#%%

import tifffile
import numpy as np

from pathlib import Path

from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity

if __name__ ==  "__main__":

    folder_save = "/media/tom/Transcend/autofish/img_fish_for_figure"


    img_xpo1_round1 = tifffile.imread("/media/tom/Transcend/autofish/2023-06-28_AutoFISH_22rounds/r1/r1_pos10_ch0.tif")
    img_xpo1_round20 = tifffile.imread("/media/tom/Transcend/autofish/2023-06-28_AutoFISH_22rounds/r21/r21_pos10_ch0.tif")
    img_kif1c_round1 = tifffile.imread("/media/tom/Transcend/autofish/2023-06-28_AutoFISH_22rounds/r3/r3_pos10_ch0.tif")
    img_kif1c_round20 = tifffile.imread("/media/tom/Transcend/autofish/2023-06-28_AutoFISH_22rounds/r22/r22_pos10_ch0.tif")


    img_xpo1_round1_2D = np.max(img_xpo1_round1, axis=0)
    img_xpo1_round20_2D = np.max(img_xpo1_round20, axis=0)
    img_kif1c_round1_2D = np.max(img_kif1c_round1, axis=0)
    img_kif1c_round20_2D = np.max(img_kif1c_round20, axis=0)

    dict_img = {"xpo1_round1": img_xpo1_round1_2D,
                    "xpo1_round20": img_xpo1_round20_2D,
                    "kif1c_round1": img_kif1c_round1_2D,
                    "kif1c_round20": img_kif1c_round20_2D}

    ## plot subset image
    y_min = 50
    y_max = 650
    x_min = 1150
    x_max = 1700

    cmap1 = "Blues"
    cmap20 = "Greens"

    cmap1 = "gray"
    cmap20 = "gray"

    for img_name in dict_img:
        fig, ax = plt.subplots(figsize=(15, 15))
        input = dict_img[img_name]
        if "round20" in img_name:
            cmap = cmap20
        else:
            cmap = cmap1

        if True: # RESCALE
            input = input[y_min:y_max, x_min:x_max]
            pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
            input = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8)

        ax.imshow(input, cmap=  cmap  )

        ax.set(xticklabels=np.array([''] * 2))
        ax.set(yticklabels=np.array([''] * 2))
        ax.axes.get_xaxis().set_visible(False)
        for spline_v in plt.gca().spines.values():
            spline_v.set_visible(False)
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
        plt.tick_params(left=False, bottom=False, right=False)
        plt.show()

        image_format = 'png'  # e.g .png, .svg, etc.
        file_name = Path(folder_save) / f'{img_name}_{cmap}.png'
        fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
        plt.show()
    plt.show()


    ### colocalization image

    # 1) plot with red and green
    # same plot but with registration

    y_min = 50
    y_max = 85
    x_min = 1450
    x_max = 1486

    """y_min = 50
    y_max = 185
    x_min = 1450
    x_max = 1600"""

    fig, ax = plt.subplots(figsize=(15, 15))

    for img_name in ["kif1c_round1", "kif1c_round20"]:
        input = dict_img[img_name]
        if True: # RESCALE
            input = input[y_min:y_max, x_min:x_max]
            if img_name == "kif1c_round1":
                pa_ch1, pb_ch1 = np.percentile(input, (50, 97))
            else:
                pa_ch1, pb_ch1 = np.percentile(input, (50, 99))
            input = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint16)
        if img_name == "kif1c_round1":
            ## plot in red
            ax.imshow(input, cmap="Greens", alpha=0.5)
        else:
            ax.imshow(input, cmap="Reds", alpha=0.5)

    ax.set(xticklabels=np.array([''] * 2))
    ax.set(yticklabels=np.array([''] * 2))
    ax.axes.get_xaxis().set_visible(False)
    for spline_v in plt.gca().spines.values():
        spline_v.set_visible(False)
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    plt.tick_params(left=False, bottom=False, right=False)
    image_format = 'png'  # e.g .png, .svg, etc.
    file_name = Path(folder_save) / f'un_registered_{img_name}_{cmap}.png'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    plt.show()
    #plt.show()

    ### translated images on referencial 1
    #bien


    pos ='pos0'
    ref_round = 'r1'
    dict_translation = np.load('/media/tom/Transcend/autofish/2023-06-28_AutoFISH_22rounds/dict_translation_ref_r1.npy',
                                 allow_pickle=True).item()

    fig, ax = plt.subplots(figsize=(15, 15))
    for img_name in ["kif1c_round1", "kif1c_round20"]:
        input = dict_img[img_name]
        if img_name == "kif1c_round1":
            round_name = 'r3'
            x_translation = dict_translation[pos][ref_round][round_name]['x_translation']
            y_translation = dict_translation[pos][ref_round][round_name]['y_translation']
        else:
            round_name = 'r22'
            x_translation = dict_translation[pos][ref_round][round_name]['x_translation']
            y_translation = dict_translation[pos][ref_round][round_name]['y_translation']
        y_min_t =  round(y_min - y_translation)
        y_max_t = round(y_max - y_translation)
        x_min_t = round(x_min -  x_translation)
        x_max_t = round(x_max -  x_translation)
        if True: # RESCALE
            input = input[y_min_t:y_max_t, x_min_t:x_max_t]
            if img_name == "kif1c_round1":
                pa_ch1, pb_ch1 = np.percentile(input, (50, 97))
            else:
                pa_ch1, pb_ch1 = np.percentile(input, (50, 99))
            input = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8)
        if img_name == "kif1c_round1":
            ## plot in red
            ax.imshow(input, cmap="Greens", alpha=0.5)
        else:
            ax.imshow(input, cmap="Reds", alpha=0.5)
    ax.set(xticklabels=np.array([''] * 2))
    ax.set(yticklabels=np.array([''] * 2))
    ax.axes.get_xaxis().set_visible(False)
    for spline_v in plt.gca().spines.values():
        spline_v.set_visible(False)
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    plt.tick_params(left=False, bottom=False, right=False)
    image_format = 'png'  # e.g .png, .svg, etc.
    file_name = Path(folder_save) / f'registered_{img_name}_{cmap}.png'
    fig.savefig(file_name, format=image_format, dpi=400, bbox_inches='tight')
    plt.show()

    ## plot with registration





    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)  # 5721