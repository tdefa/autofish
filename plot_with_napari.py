

import napari
import numpy as np
import tifffile


###




if __name__ == '__main__':
    final_spots_detection = np.load("/media/tom/T7/2023-01-19-PAPER-20-rounds/test/final_spots_detection.npy",
            allow_pickle=True).item()

    im_fish = tifffile.imread("/media/tom/T7/2023-01-19-PAPER-20-rounds/test2/r1/r1_pos0_ch0.tif")
    spots_list = final_spots_detection['r1']['pos0']["raw_spots"]
    viewer = napari.Viewer()
    viewer.add_image(im_fish, name='fish')
    if len(spots_list) > 0:
        viewer.add_points(spots_list, size=3,
                          name='spots',
                          edge_color='red',
                          face_color='red',
                          )



    im_fish = tifffile.imread("/media/tom/T7/2023-01-19-PAPER-20-rounds/test2/r2/r2_pos0_ch0.tif")
    spots_list = final_spots_detection['r2']['pos0']["raw_spots"]
    viewer = napari.Viewer()
    viewer.add_image(im_fish, name='fish')
    if len(spots_list) > 0:
        viewer.add_points(spots_list, size=3,
                          name='spots',
                          edge_color='red',
                          face_color='red',
                          )


