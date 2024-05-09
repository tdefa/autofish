import tifffile
from autofish_analysis.registration import folder_translation
import numpy as np
from pathlib import Path

folder_of_rounds = "/media/tom/Transcend/2023-10-06_LUSTRA/"  # works ok
fixed_round_name = "r1"
folder_regex = 'r'
chanel_regex = '*ch0*'
registration_repeat = 5
position_naming = True
sigma_gaussian_filter = 0.8
similarity_metric = "mattes"

dict_translation = folder_translation(
                            folder_of_rounds = "/media/tom/Transcend/2023-10-06_LUSTRA/",  # works ok
                           fixed_round_name = "r1",
                           folder_regex = 'r*',
                           chanel_regex = '*ch0*',
                           registration_repeat = 5,
                           position_naming = True,
                           sigma_gaussian_filter = 0.8,
                            similarity_metric = "mattes"
                                      )

np.save("/media/tom/Transcend/2023-10-06_LUSTRA/8_may_dico_translation.npy", dict_translation)

from autofish_analysis.registration import plot_registrered_image

#dict_translation = np.load("dict_translation = np.load("/media/tom/Transcend/2023-10-06_LUSTRA/garbage/26_oct_dico_translation.npy", allow_pickle = True).item()
#
list_round = list(dict_translation[list(dict_translation.keys())[0]][fixed_round_name].keys())
list_pos = list(dict_translation.keys())


for rd in list_round:
    if rd == fixed_round_name:
        continue
    path_save = Path(folder_of_rounds) / (rd + "/" + "registration")
    path_save.mkdir(exist_ok=True, parents=True)
    for pos in list_pos:
        path_image1 = Path(folder_of_rounds) / (
                    fixed_round_name + '/' + fixed_round_name + "_" + pos + "_ch0.tif")
        path_image2 = Path(folder_of_rounds) / (rd + '/' + rd + "_" + pos + "_ch0.tif")
        fig, ax, image1, image2, shifted_image2, x_translation, y_translation = plot_registrered_image(
            dict_translation=dict_translation,
            path_image1=path_image1,
            path_image2=path_image2,
            plot_napari=False,
            figsize=(15, 15)
        )
        fig.savefig(path_save / Path(path_image2).stem)
        tifffile.imwrite(path_save / (Path(path_image2).stem + "_image1.tif"), image1)
        tifffile.imwrite(path_save / (Path(path_image2).stem + "_image2.tif"), image2)
        tifffile.imwrite(path_save / (Path(path_image2).stem + "_shifted_image2.tif"), shifted_image2)








############# register the spots detection

from autofish_analysis.registration import spots_registration
dict_spots = np.load("/media/tom/Transcend/2023-10-06_LUSTRA/detection_ufish/dict_spots_unregistered.npy", allow_pickle=True).item()

dico_spots_registered_df, dico_spots_registered, missing_data = spots_registration(
    dict_spots = dict_spots,
    dict_translation = dict_translation,
    fixed_round_name=fixed_round_name,
    check_removal=False,
    threshold_merge_limit=0.330,
    scale_xy=0.103,
    scale_z=0.270,
)

## save results
np.save("/media/tom/Transcend/2023-10-06_LUSTRA/dico_spots_registered_df.npy", dico_spots_registered_df)
np.save("/media/tom/Transcend/2023-10-06_LUSTRA/dico_spots_registered.npy", dico_spots_registered)
















