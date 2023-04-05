



import argparse
import os
import re
from os import listdir
from os.path import isfile, join
from pathlib import Path

import bigfish
import bigfish.stack as stack
import numpy as np
import tifffile
from matplotlib import pyplot as plt
from matplotlib.patches import RegularPolygon
from scipy import ndimage as ndi
from skimage import measure
from skimage.filters import threshold_otsu
from skimage.measure import label
## from cellpose / bigfish env
from tqdm import tqdm

plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rc_context({"axes.labelsize" : 45,})
## Function to detect the  balls


def compute_cytoplasm_mask(im_fish,
                           maxfilter_kernel = 15,
                           minfilter_kernel = 15,
                           cc_threshold = 20000,
                           ):
    """

    :param im_fish (np.array): fish signal
    :param maxfilter_kernel:
    :param minfilter_kernel:
    :param cc_threshold (int): number of voxel of the smallest connected cytoplasm
    :return:
    """
    threshold_list = []
    for slice in im_fish:
            threshold_list.append(threshold_otsu(slice))
    ### compute binary mask
    binay_mask = np.zeros(im_fish.shape)
    print("step1")
    for index in range(len(im_fish)):
        binay_mask[index] = im_fish[index] > threshold_list[index]
    binay_mask_max_filter = ndi.maximum_filter(binay_mask, size = maxfilter_kernel)
    binay_mask_max_filter = ndi.minimum_filter(binay_mask_max_filter, size=minfilter_kernel)
    ### classify large connected component as cytoplasm
    label_image = measure.label(binay_mask_max_filter)
    unique_label = np.unique(label_image)
    cytoplasm_mask = np.zeros(im_fish.shape)
    print("step2")
    for label in unique_label[1:]:
        if np.sum(label_image == label) > cc_threshold: ##todo find a better threshold criteria
            cytoplasm_mask += (label_image == label).astype(int)
    return cytoplasm_mask




if __name__ == "__main__":
    im_fish = tifffile.imread("/media/tom/T7/2023-01-19-PAPER-20-rounds/round_impair/r1/r1_pos0_ch0.tif")

    cytoplasm_mask = compute_cytoplasm_mask(im_fish,
                           maxfilter_kernel=15,
                           minfilter_kernel=15,
                           cc_threshold=20000,
                           )

    np.save("/media/tom/T7/2023-01-19-PAPER-20-rounds/round_impair/r1/cytoplasm_mask_r1_pos0_ch0.npy", cytoplasm_mask)


    np.save("/media/tom/T7/2023-01-19-PAPER-20-rounds/round_pair/r2/cytoplasm_mask_r2_pos0_ch0.npy", cytoplasm_mask)





