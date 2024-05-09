


import matplotlib.pyplot as plt
import numpy as np
import tifffile

from piscis import Piscis
from piscis.data import load_datasets
from piscis.downloads import download_dataset
from piscis.utils import pad_and_stack

download_dataset('20230905', '')



test_ds = load_datasets('20230905', load_train=False, load_valid=False, load_test=True)['test']
images = pad_and_stack(test_ds['images'])
test_ds['images'] = images
coords = test_ds['coords']


model = Piscis(model_name='20230905')

images_test =  tifffile.imread('/media/tom/Transcend1/2023-10-06_LUSTRA/r1/r1_pos0_ch0.tif')

coords_pred, y = model.predict(images_test, threshold=1.0, intermediates=True)