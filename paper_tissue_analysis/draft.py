import pandas as pd
import tifffile
import numpy as np
from skimage import io
from ufish.api import UFish
import time



img = tifffile.imread("/media/tom/Transcend/2023-10-06_LUSTRA/r4/r4_pos2_ch0.tif")
enh_img = np.load("/media/tom/Transcend/2023-10-06_LUSTRA/detection_ufish/r4/r4_pos2_ch0_enh_img.npy")
pred_spots = np.load('/media/tom/Transcend/2023-10-06_LUSTRA/detection_ufish/r4/r4_pos2_ch0_pred_spots.npy')


ufish = UFish()
ufish.load_weights()



t = time.time()
pred_spots, enh_img = ufish.predict(img)
print(time.time()-t)


import napari
viewer = napari.Viewer()
viewer.add_image(img)
viewer.add_image(enh_img)
viewer.add_points(pred_spots, size=25)
np.save("pred_spots", pred_spots)
np.save("enh_img", enh_img)


#####################"
# vizualization
#####################


img = tifffile.imread("/media/tom/Transcend/2023-10-06_LUSTRA/r8/r8_pos1_ch0.tif")
enh_img = np.load("/media/tom/Transcend/2023-10-06_LUSTRA/detection_ufish/r8/r8_pos1_ch0_enh_img.npy")
pred_spots = np.load('/media/tom/Transcend/2023-10-06_LUSTRA/detection_ufish/r8/r8_pos1_ch0_pred_spots.npy')

import napari
viewer = napari.Viewer()
viewer.add_image(img)
viewer.add_image(enh_img)
viewer.add_points(pred_spots, size=4, edge_color='red', face_color='red')
np.save("pred_spots", pred_spots)
np.save("enh_img", enh_img)

#####################"
# vizualization comseg input
#####################


import napari
import tifffile
import numpy as np
from pathlib import Path
import pandas as pd
import numpy as np

df = pd.read_csv("/media/tom/Transcend/2023-10-06_LUSTRA/input_comseg/img0.csv")
mask = np.load("/media/tom/Transcend/2023-10-06_LUSTRA/input_comseg/mask/img0.npy")

import napari
viewer = napari.Viewer()
#viewer.add_image(mask)
# select x y and z of gene r1
gene = "r1"
list_x = list(df[df["gene"] == gene].x)
list_y = list(df[df["gene"] == gene].y)
list_z = list(df[df["gene"] == gene].z)

## add spots in napari
spots = np.array([list_z, list_y, list_x]).T
viewer.add_points(spots, size=50, edge_color='red', face_color='red')


gene = "r8"
list_x = list(df[df["gene"] == gene].x)
list_y = list(df[df["gene"] == gene].y)
list_z = list(df[df["gene"] == gene].z)

## add spots in napari
spots = np.array([list_z, list_y, list_x]).T
viewer.add_points(spots, size=50, edge_color='green', face_color='green')