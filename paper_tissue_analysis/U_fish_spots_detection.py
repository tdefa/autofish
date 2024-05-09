

import tifffile
import numpy as np
from skimage import io
from ufish.api import UFish
import time
import argparse
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='test')
    ### path to the image
    parser.add_argument("--path_to_round_folder", type=str,
                        default="/media/tom/Transcend/2023-10-06_LUSTRA/",
                        help="path folder cell semgenation crop")

    parser.add_argument("--path_to_save_res", type=str,
                        default="/media/tom/Transcend/2023-10-06_LUSTRA/detection_ufish/",
                        help="path folder cell semgenation crop")

    parser.add_argument("--port", default=3950)
    parser.add_argument("--mode", default='client')
    parser.add_argument("--host", default='127.0.0.2')

    args = parser.parse_args()

    path_to_round_folder = Path(args.path_to_round_folder)
    path_to_save_res = Path(args.path_to_save_res)

    ufish = UFish()
    ufish.load_weights()
    for path_round in tqdm(list(path_to_round_folder.glob("r*"))):

        path_to_save_res_round = path_to_save_res / path_round.name
        path_to_save_res_round.mkdir(parents=True, exist_ok=True)

        for path_pos in tqdm(list(path_round.glob("r*_pos*ch0.tif"))):

            img = tifffile.imread(path_pos)
            pred_spots, enh_img = ufish.predict(img)
            #pred_spots = np.array([[0, 0, 0]])
            #enh_img = np.array([0, 0, 0])
            np.save(path_to_save_res_round / f"{path_pos.stem}_pred_spots", pred_spots)
            np.save(path_to_save_res_round / f"{path_pos.stem}_enh_img", enh_img)


    print("done ufish")


    ## convert the spots detection into a dictionary of spots dict[round][pos] = spots

    dict_spots = {}
    for path_round in list(path_to_save_res.glob("r*")):
        dict_spots[path_round.name] = {}
        for path_pos in list(path_round.glob("*_pred_spots.npy")):
            dict_spots[path_round.name][path_pos.stem] = np.load(path_pos)

    np.save(path_to_save_res / "dict_spots_unregistered.npy", dict_spots)

