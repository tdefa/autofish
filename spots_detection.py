

#%%
from bigfish.detection import build_reference_spot, compute_snr_spots
from bigfish.detection.spot_detection import (_get_candidate_thresholds,
                                              spots_thresholding)
from tqdm import tqdm
import re
from pathlib import Path
from bigfish.detection import build_reference_spot, compute_snr_spots
import bigfish.stack as stack
import numpy as np
import pylab
import tifffile
from bigfish import detection
#from bigfish.detection import fit_subpixel
from skimage.measure import label, regionprops
import time



def compute_spot_detection_of_folder(path_to_folder = "/media/tom/T7/2023-01-19-PAPER-20-rounds/images",
                                     channel ="ch2",
                                     Key_world = "os",
                                     min_distance=[5, 7, 7],
                                     sigma=[1.35, 1.35, 1.35],
                                     threshold = None,
                                     reference_number_of_spot = None,
                                     beta = 1,
                                     alpha = 0.5,
                                     voxel_size = [],
                                     get_nb_spots_per_cluster = True,
                                     subpixel_loc = True,
                                     spot_radius = [],
                                     mode = "bead"):

    """
    Args:
        path_to_folder (str): path to folder containg all the round folder
        channel (str): channel to analyse
        Key_world (str): key wor
        min_distance ():
        sigma ():
        threshold ():
        reference_number_of_spot ():
        beta ():
        alpha ():
        voxel_size ():
        get_nb_spots_per_cluster ():
        subpixel_loc ():
        spot_radius ():
    Returns:
    """

    dico_detection = {}
    # list the folder round
    folder_patho = Path(path_to_folder)
    print(list(folder_patho.glob("*/")))
    for folder_round in tqdm(list(folder_patho.glob("*/"))):
        round = str(folder_round).split("/")[-1]
        if 'r' != round[0]:
            continue
        print(str(folder_round).split("/")[-1])
        dico_detection[round] = {}
        list_image_round = list(folder_round.glob(f"*{Key_world}*{channel}*")) #this condition should select only relevant folder
        for image_path_pos in tqdm(list_image_round):
            image_name = str(image_path_pos).split('/')[-1]
            print(image_name)
            position = image_name.split('_')[1]
            res_detection = find_rna_subpixel(path_im_fish = image_path_pos,
                                min_distance = min_distance,
                                  sigma = sigma,
                                  threshold = threshold,
                                  reference_number_of_spot=reference_number_of_spot[position] if reference_number_of_spot is not None else None,
                                  beta = beta,
                                  alpha = alpha,
                                  voxel_size = voxel_size,
                                  subpixel_loc = subpixel_loc,
                                  spot_radius = spot_radius,
                                  mode  = mode)
            #dico_detection[round][image_name] = res_detection
            dico_detection[round][position] = res_detection
    return dico_detection
def find_rna_subpixel(path_im_fish = "dync1h1_2_MMStack_Pos0_ch1.tif",
                    min_distance = (5, 7, 7),
                      sigma = [1.35, 1.35,1.35],
                      threshold = None,
                      reference_number_of_spot=None,
                      beta = 1,
                      alpha = 0.5,
                      voxel_size = [],
                      subpixel_loc = True,
                      spot_radius = [],
                      mode  = "bead"):

    """
    Parameters
    ----------
    path_im_fish
    nb_to_keep_focus
    min_distance
    sigma
    channel_in_tiff: channel to run the detection
    threshold
    reference_number_of_spot: int the number of spots we want to detect in the sample
    beta : int or float
        Multiplicative factor for the intensity threshold of a dense region.
        Default is 1. Threshold is computed with the formula: threshold = int(reference_spot.max() * beta)
    alpha : int or float
        Intensity score of the reference spot, between 0 and 1. If 0, reference
        spot approximates the spot with the lowest intensity. If 1, reference
        spot approximates the brightest spot. Default is 0.5.
    Returns
    -------
    """
    assert mode in ["bead", 'rna']
    if type(min_distance) == list:
        min_distance = tuple(min_distance)
    im_fish = tifffile.imread(path_im_fish)
    if im_fish.ndim == 4:
        im_fish = im_fish[0]
        print(path_im_fish)
    print(im_fish.shape)
    # LoG filter
    rna_log = stack.log_filter(im_fish, sigma)
    # local maximum detection
    mask = detection.local_maximum_detection(rna_log, min_distance=min_distance)
    # thresholding
    if threshold is None and reference_number_of_spot is None:
        threshold = detection.automated_threshold_setting(rna_log, mask)
        print('threshold %s' % threshold)
    elif threshold is None and reference_number_of_spot is not None:
        threshold, nb_rna, _, _ = get_exact_threshold(rna_log = rna_log,
                                                      min_distance = min_distance,
                                                      nb_spot_ref=reference_number_of_spot)
        print('threshold %s' % threshold)
        print('detected rna %s with a ref of %s ' % (str(nb_rna), str(reference_number_of_spot)))
    else:
        if threshold is not None and reference_number_of_spot is not None:
            raise Exception(" choose between a thresold or a number of spots")

    spots , _ = detection.spots_thresholding(rna_log, mask, int(threshold))
    print(f'nb spots {len(spots)}')
    if not subpixel_loc:
        return {"raw_spots": spots}

    subpixel_spots = detection.fit_subpixel(im_fish,
                                  spots,
                                       voxel_size=voxel_size,
                                       spot_radius=spot_radius,
                                       )
    print("subpixel_spots")
    if mode == "bead":
        return {"raw_spots": spots,
         "subpixel_spots": subpixel_spots}
    assert mode == 'rna'
    spots_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(
        image=im_fish,
        spots=spots,
        voxel_size = voxel_size,
        spot_radius = spot_radius,
        alpha=alpha,  # alpha impacts the number of spots per candidate region
        beta=beta,  # beta impacts the number of candidate regions to decompose
        gamma=5)
    ### subpixel finting on no-dense region with individual fiting
    ### subpixel finting on dense region with individual fiting
    ind_dense_regions = np.array([s[:3] for s in dense_regions if s[3] == 1])

    #print(ind_dense_regions)

    if len(ind_dense_regions) > 0:
        subpixel_ind_dense_regions = detection.fit_subpixel(im_fish,
                                      ind_dense_regions,
                                      voxel_size=voxel_size,
                                      spot_radius=spot_radius,)
    else:
        subpixel_ind_dense_regions = []
    ### subpixel fiting on dense region with many spots
    many_dense_regions = np.array([s[:3] for s in dense_regions if s[3] > 1])
    print(" check which is the best sigma and raduis to get the best ref spot and sup pixel localization")
    print("for exemple the parameter fitted when decomposing the cluster")
    print("parameter psf hard coded")
    spots_in_regions = None ## not retrun in bigfish 0.6
    spots_out_regions = None  ## not retrun in bigfish 0.6
    return {"raw_spots": spots,
            "subpixel_spots":subpixel_spots,
            "subpixel_ind_dense_regions": subpixel_ind_dense_regions,
            "spots_out_regions": spots_out_regions,
            "spots_in_regions": spots_in_regions,
            "spots_post_decomposition" :spots_post_decomposition,
            "dense_regions": dense_regions,
            "many_dense_regions" : many_dense_regions,
            "reference_spot": reference_spot,
            "fitting_parameters": [voxel_size,spot_radius,  alpha, beta],
            "threshold": threshold}


def get_exact_threshold(rna_log, min_distance, nb_spot_ref):
    """
    Function to detect the same number of spot in all channel
    :param rna_log:
    :param min_distance:
    :param nb_spot_ref: number of spots we want to detect
    :return:
    """
    mask_local_max = detection.local_maximum_detection(rna_log, min_distance=min_distance)
    thresholds = _get_candidate_thresholds(rna_log.ravel())
    # get spots count and its logarithm
    first_threshold = float(thresholds[0])
    spots, mask_spots = spots_thresholding(
        rna_log, mask_local_max, first_threshold, remove_duplicate=False)
    value_spots = rna_log[mask_spots]
    count_spots = [np.count_nonzero(value_spots > t)
                   for t in thresholds]
    for index_th in range(len(count_spots)):
        if count_spots[index_th] < nb_spot_ref:
            break
    mask_local_max = detection.local_maximum_detection(rna_log, min_distance=min_distance)
    for index_th_remove_duplicate in range(index_th, 0,-1):
        mask = (mask_local_max & (rna_log > thresholds[index_th_remove_duplicate]))
        t = time.time()
        cc = label(mask)
        local_max_regions = regionprops(cc)
        spots = []
        for local_max_region in local_max_regions:
            spot = np.array(local_max_region.centroid)
            spots.append(spot)
        print(' time %s  thres %s   nb spots %s' % (str(time.time() - t), str(thresholds[index_th_remove_duplicate]),
                                                    str(len(spots))))
        if len(spots) >= nb_spot_ref:
            break
        #previous_th, previous_len
    ## on peut ameliorer la policy ici

    if len(spots) == nb_spot_ref:
        return thresholds[index_th_remove_duplicate ], count_spots[index_th_remove_duplicate], thresholds, count_spots
    return thresholds[index_th_remove_duplicate], len(spots), thresholds, count_spots





def get_reference_dico(first_dico_detection, mode = "median", pos_key = False):
    """
    :param first_dico_detection (dict):   first_dico_detection[folder][pos_name] = [array of spots]
    :return: the number of spots to detect in each position ref_dico[pos] = nb_spots
    """
    import re
    assert mode == "max" or mode == 'min' or mode == "median"
    ref_dico = {}
    rounds = list(first_dico_detection.keys())
    pos_names = list(first_dico_detection[rounds[0]].keys())
    for name in pos_names:
        print(name)
        pos = re.search(r'pos([0-9]+)', name).group(0)
        print(pos)

        if mode  ==  "max":
            ref_dico[pos] = np.max([len(first_dico_detection[rd][name]["raw_spots"]) for rd in rounds])
        if mode == 'min':
            ref_dico[pos] = np.min([len(first_dico_detection[rd][name]["raw_spots"]) for rd in rounds])
        if mode == "median":
            ref_dico[pos] = np.median([len(first_dico_detection[rd][name]["raw_spots"]) for rd in rounds])

    return ref_dico





















