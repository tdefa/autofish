



import numpy as np
from bigfish.detection import  compute_snr_spots

from pathlib import Path
import tifffile
from tqdm import tqdm

def compute_snr_for_each_spots(spots,
                               im_fish,
                               voxel_size = [270, 108, 108],
                                        snr_psf_yx=300,
                                        snr_psf_z=350,):
    snr_list= []
    for s in spots:
        snr = compute_snr_spots(im_fish, np.array([s]),
                         voxel_size_z=voxel_size[0],
                         voxel_size_yx=voxel_size[-1],
                         psf_yx=snr_psf_yx,
                         psf_z=snr_psf_z)
        snr_list.append(snr)
    return snr_list


def compute_intensity_list(spots,
                           im_fish):
    intensity_list = []
    for s in spots:
        intensity = im_fish[int(round(s[0])),
                            int(round(s[1])),
                            int(round(s[2]))]
        intensity_list.append(intensity)
    return intensity_list



def compute_signal_quality(final_spots_detection,
                       dico_matched_rna,
                        path_folder_img = "/media/tom/T7/2023-01-19-PAPER-20-rounds/round_pair/",
                        channel = "ch0",
                        ref_round = "r2",
                       voxel_size=[270, 108, 108],
                       spot_radius=(400, 300, 300),
                       round_not_to_take = ['r19'],
                        min_filter_size = 10):
    import scipy

    print(f'round_not_to_take: {round_not_to_take}')
    if path_folder_img[-1] != "/":
        path_folder_img += "/"
    from tqdm import tqdm
    ## get list_couple_indie for the reference round
    round_list_sorted = sorted(list(dico_matched_rna.keys()), key=lambda kv: int(kv[1:]))[:-1]
    ref_round_bis = round_list_sorted[round_list_sorted.index(ref_round) + 1]
    dico_signal_quality =  {}
    for r in final_spots_detection.keys():
        if r in round_not_to_take:
            continue
        dico_signal_quality[r] = {}
        for p in final_spots_detection[r].keys():
            dico_signal_quality[r][p] = {}
            print((r,p))

            path_img = list(Path(f'{path_folder_img + r}').glob(f'r*{p}*{channel}.tif*'))
            print(path_img)
            assert len(path_img) == 1
            img = tifffile.imread(path_img[0])

            if r != ref_round:
                list_couple_index_sp1 = dico_matched_rna[r][p]['list_couple_index_sp1']
                spots = final_spots_detection[r][p]['raw_spots'][list_couple_index_sp1]
            else:
                list_couple_index_sp0_ref = dico_matched_rna[ref_round_bis][p]['list_couple_index_sp0_ref']
                spots = final_spots_detection[ref_round][p]['raw_spots'][list_couple_index_sp0_ref]
            intensity_list = compute_intensity_list(
                    spots = spots,
                    im_fish = img)
            snr = compute_snr_spots(image = img,
                                    spots = spots,
                                    voxel_size=voxel_size,
                                    spot_radius=spot_radius)
            dico_signal_quality[r][p]['intensity_list'] = intensity_list
            dico_signal_quality[r][p]['snr_list'] = snr[1]
            dico_signal_quality[r][p]['snr_median'] = snr[0]
            path_mask = list(Path(f'{path_folder_img + r}/cyto_mask2D_3dim/').glob(f'r*{p}*{channel}.tif*'))
            print(path_mask)
            assert len(path_mask) == 1
            mask_cyto = tifffile.imread(path_mask)
            mask_cyto
            img_background = img.copy()
            img_spots_mask = np.zeros(img_background.shape)
            z = spots[:, 0]
            y = spots[:, 1]
            x = spots[:, 2]
            img_spots_mask += 1
            img_spots_mask[z,y,x] = 0
            img_spots_mask = scipy.ndimage.minimum_filter(img_spots_mask, size=10)
            img_background = img_background * (mask_cyto > 1).astype(int)
            img_background = img_background *  img_spots_mask
            np.median(img_background[img_background > 0])

            dico_signal_quality[r][p]['mean_background'] = np.median(img_background[img_background > 0])
            dico_signal_quality[r][p]['median_background'] = np.mean(img_background[img_background > 0])
    return dico_signal_quality






if __name__ ==  "__main__":

    img = tifffile.imread("/media/tom/T7/2023-01-19-PAPER-20-rounds/round_pair/r4/r4_pos0_ch0.tif")

    list_couple_index_sp1 = dico_matched_rna["r4"]["pos0"]['list_couple_index_sp1']

    spots = final_spots_detection["r4"]["pos0"]['raw_spots'][list_couple_index_sp1]

    intensity_list = compute_intensity_list(
            spots = spots,
            im_fish = img)

    snr = compute_snr_spots(image =img,
                      spots = spots,
                      voxel_size= [270, 108, 108],
                      spot_radius = (400, 300, 300))


