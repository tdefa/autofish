


from pathlib import Path
import argparse
import os
import numpy as np
import tifffile













################## generate the sticing cooridnat in a txt file ##################

if __name__ == "false": ## to check and run

    import imagej
    import scyjava
    from autofish_analysis.utils import macro

    # from utils.macro import STITCHING_MACRO

    scyjava.config.add_option('-Xmx40g')
    if "ij" not in vars():
        ij = imagej.init('sc.fiji:fiji')

    ### define the images to stitch and their position
    img_pos_dico = {"img0": ['pos0', "pos1", "pos2", 'pos3', 'pos4',
                             'pos5', "pos6", "pos7", 'pos8', 'pos8'],
                    }

    import importlib

    from autofish_analysis import stitching

    importlib.reload(stitching)
    from autofish_analysis.stitching import stich_with_image_J

    stich_with_image_J(
        ij,
        STITCHING_MACRO=macro.STITCHING_MACRO,
        img_pos_dico=img_pos_dico,
        stitching_type="[Grid: snake-by-row]",
        order="[Left & Down]",
        grid_size_x=3,
        grid_size_y=3,
        tile_overlap=20,
        image_name="r1_pos{i}_ch0.tif",
        image_path=args.image_path_stiching,
        output_path=args.output_path_stiching)


### use the generate stpot coordiante to stich the images and spots detection


if __name__ == "false": ## to check and run

    from autofish_analysis.stitching import parse_txt_file

    img_name = "img0.txt"
    img_pos_dico = {"img0": ['pos0', "pos1", "pos2", 'pos3', 'pos4',
                             'pos5', "pos6", "pos7", 'pos8', 'pos8'],
                    }
    image_path_stiching = "/media/tom/Transcend/2023-10-06_LUSTRA/r1/"
    folder_of_rounds = '/media/tom/Transcend/2023-10-06_LUSTRA/'
    dict_stitch_all_img = {}
    for img_name in img_pos_dico:
        path_txt = Path(image_path_stiching) / f"TileConfiguration.registered_{img_name}.txt"
        dico_stitch = parse_txt_file \
            (path_txt=path_txt,
             image_name_regex="_pos", )
        dict_stitch_all_img[img_name] = dico_stitch
    np.save(f"{folder_of_rounds}dict_stitch_img.npy", dict_stitch_all_img)


### stitch the spots detection


if __name__ == "false": ## to check and run

    from autofish_analysis.stitching import stitch_spots_detection

    dico_bc_gene0 = {
        'r1': "Rtkn2",  # bc1
        'r2': "Lamp3",  # bc3
        'r3': "Pecam1",  # bc4
        'r4': "Ptprb",  # bc5
        'r5': "Pdgfra",  # bc6
        'r6': "Chil3",  # bc7
        'r7': "unknow",  # bc1
        'r8': "Rtkn2",  # bc3
        'r9': "Lamp3",  # bc4
        'r10': "Pecam1",  # bc7
    }


    dict_bc_gene= {
        'r1': "r1",  # bc1
        'r2': "r2",  # bc3
        'r3': "r3",  # bc4
        'r4': "r4",  # bc5
        'r5': "r5",  # bc6
        'r6': "r6",  # bc7
        'r7': "r7",  # bc7
        'r8': "r8",  # bc7
        'r9': "r9",  # bc7
        'r10': "r10",  # bc7
    }

    dico_spots_registered = np.load(
        f"/media/tom/Transcend/2023-10-06_LUSTRA/dico_spots_registered.npy",
        allow_pickle=True).item()
    dico_spots_registered_df = np.load("/media/tom/Transcend/2023-10-06_LUSTRA/dico_spots_registered_df.npy",
        allow_pickle=True).item()
    dict_stitch_img = np.load(f"/media/tom/Transcend/2023-10-06_LUSTRA/dict_stitch_img.npy",
                              allow_pickle=True).item()

    from autofish_analysis.stitching import stich_dico_spots

    dico_spots_registered_stitch_df = stich_dico_spots(
        dict_spots_registered_df=dico_spots_registered_df,
        dict_stitch_img=dict_stitch_img,
        dict_round_gene=dict_bc_gene,
        image_shape=[38, 2048, 2048],
        nb_tiles_x=3,
        nb_tiles_y=3,
    )
    ## save results
    np.save("/media/tom/Transcend/2023-10-06_LUSTRA/dico_spots_registered_stitch_df.npy",
            dico_spots_registered_stitch_df)

    dico_spots_registered_stitch_df["img0"].to_csv("/media/tom/Transcend/2023-10-06_LUSTRA/input_comseg/img0.csv")



## stitching fish images and mask
if __name__ == "false": ## to check and run

    from autofish_analysis.stitching import stich_from_dico_img, stich_segmask

    ## stitch fish

    dico_stitch = dict_stitch_img[image_name]
    path_mask = "/media/tom/Transcend/2023-10-06_LUSTRA/r2"
    final_masks = stich_from_dico_img(dico_stitch,
                                      # np.load(f"/media/tom/T7/Stitch/acquisition/2mai_dico_stitch.npy",allow_pickle=True).item()
                                        path_mask=path_mask,
                                      regex="*_ch0*tif*",
                                      image_shape=image_shape,
                                      nb_tiles_x=nb_tiles_x,
                                      nb_tiles_y=nb_tiles_y,
                                      )
    ## save
    tifffile.imwrite(f"{path_mask}/mask_stitch.tif", final_masks)
    tifffile.imwrite(f"{path_mask}/mask.tif_stitch_mip", np.amax(final_masks, 0))


    dico_stitch = dict_stitch_img[image_name]
    path_mask = "/media/tom/Transcend/2023-10-06_LUSTRA/r1"
    final_masks = stich_from_dico_img(dico_stitch,
                                      # np.load(f"/media/tom/T7/Stitch/acquisition/2mai_dico_stitch.npy",allow_pickle=True).item()
                                        path_mask=path_mask,
                                      regex="*_ch1*tif*",
                                      image_shape=image_shape,
                                      nb_tiles_x=nb_tiles_x,
                                      nb_tiles_y=nb_tiles_y,
                                      )
    ## save
    tifffile.imwrite(f"{path_mask}/mask_ch1__stitch.tif", final_masks)
    tifffile.imwrite(f"{path_mask}/mask_ch1_stitch_mip.tif", np.amax(final_masks, 0))


    ## stitch mask


    stich_segmask(
        dict_stitch_img = dict_stitch_img,
                  path_mask="/media/tom/Transcend/2023-10-06_LUSTRA/segmentation_mask/50",
                  path_to_save_mask="/media/tom/T7/stich0504/segmentation_mask_stitch",
                  image_shape=[40, 2048, 2048],
                  nb_tiles_x=3,
                  nb_tiles_y=3,
                  iou_threshold=0.25,
                  margin=1000,
                  )

    mask_ch1_stitch = tifffile.imread(f"/media/tom/Transcend/2023-10-06_LUSTRA/r1/mask_ch1__stitch.tif")
    mask_ch1_stitch = mask_ch1_stitch.astype(np.uint8)
    tifffile.imwrite(f"/media/tom/Transcend/2023-10-06_LUSTRA/r1/mask_ch1_stitch_int8.tif", mask_ch1_stitch, dtype=np.uint8)










