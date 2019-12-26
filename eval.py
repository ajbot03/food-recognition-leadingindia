import os
import sys
import time
import numpy as np
import skimage.io
import keras.backend
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
#  
# A quick one liner to install the library 
# !pip install git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

from pycocotools.coco import COCO
from cocoeval import COCOeval
from pycocotools import mask as maskUtils

import coco #a slightly modified version

from mrcnn.evaluate import build_coco_results, evaluate_coco
from mrcnn.dataset import MappingChallengeDataset
from mrcnn import visualize


import zipfile
import urllib.request
import shutil
import glob
import tqdm
import random

files = glob.glob(os.path.join('/content/drive/My Drive/test_images' , "*.jpg"))
ALL_FILES=[]
_buffer = []
for _idx, _file in enumerate(files):
    if len(_buffer) == config.IMAGES_PER_GPU * config.GPU_COUNT:
        ALL_FILES.append(_buffer)
        _buffer = []
    else:
        _buffer.append(_file)

if len(_buffer) > 0:
    ALL_FILES.append(_buffer)

_final_object = []
for files in tqdm.tqdm(ALL_FILES):
    images = [skimage.io.imread(x) for x in files]
    if(len(images)!= config.IMAGES_PER_GPU):
        images = images + [images[-1]]*(config.BATCH_SIZE - len(images))
    predoctions = model.detect(images, verbose=0)
    for _idx, r in enumerate(predoctions):
        if(_idx < len(files)):
            _file = files[_idx]
            image_id = int(_file.split("/")[-1].replace(".jpg",""))
            for _idx, class_id in enumerate(r["class_ids"]):
                if class_id > 0:
                    mask = r["masks"].astype(np.uint8)[:, :, _idx]
                    bbox = np.around(r["rois"][_idx], 1)
                    bbox = [float(x) for x in bbox]
                    _result = {}
                    _result["image_id"] = image_id
                    _result["category_id"] = id_category[class_id]
                    _result["score"] = float(r["scores"][_idx])
                    _mask = maskUtils.encode(np.asfortranarray(mask))
                    _mask["counts"] = _mask["counts"].decode("UTF-8")
                    _result["segmentation"] = _mask
                    _result["bbox"] = [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]
                    _final_object.append(_result)

