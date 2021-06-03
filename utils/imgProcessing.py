import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import skimage.io
import math
import time
import cv2

def rectangle_from_poly(x_coords, y_coords):

    l = min(x_coords)
    r = max(x_coords)
    t = min(y_coords)
    b = max(y_coords)
    rect = [l, r, t, b]
    return rect

def add_object(image, polygons, finalImgSz):

    object_arr = []

    # for each ROI in image
    for itm in range(len(polygons)):
        x_coords = polygons[itm]['all_points_x']
        y_coords = polygons[itm]['all_points_y']

        box = rectangle_from_poly(x_coords, y_coords)

        #snip
        tmp_img = image[box[2]:box[3], box[0]:box[1], :]
        # cv2.imshow('image', tmp_img)
        # cv2.waitKey(0)

        # rescale
        img = cv2.resize(tmp_img, dsize=(finalImgSz, finalImgSz), interpolation=cv2.INTER_CUBIC)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        object_arr.append(img)

    return object_arr



def load_data(groundTruthFile=None, numChannels=3, finalImgSz=64, dataset_dir=None):
    """
    creates the ndarray of the data for embedding
    :param dataset_dir:
    :param subset:
    :param groundTruthFile:
    :param numChannels:
    :return:
    """
    subset="train"

    if numChannels == 4:
        import custom_parse_rgbz as parse_txt
    else:
        import custom_parse_rgb as parse_txt
        #z_path = z_imgpath, z_size = a['zDepth_size'], z_name = a['zDepth_filename']

    if dataset_dir == None and groundTruthFile == None:
        print("ERROR: No Data Found")
        return

    """load dataset and prepare for embedding
    """
    data = []
    labels = []

    # Train or validation dataset?

    datafile = groundTruthFile

    print("datafile: ", datafile)

    # check to see if groundTruthFile is .txt or .json
    extension = os.path.splitext(groundTruthFile)[1]
    if extension == ".json":
        annotations = json.load(open(datafile))
    elif extension == ".txt":  # load txt file
        annotations, dataset_dir = parse_txt.load(datafile)
        print('Number of images found: ', len(annotations))
    else:
        print("ERROR: GroundTruthFile type is not .txt or .json")
        return

    annotations = list(annotations.values())  # don't need the dict keys

    # The VIA tool saves images in the JSON even if they don't have any
    # annotations. Skip unannotated images.
    annotations = [a for a in annotations if a['regions']]

    # Add images
    for a in annotations:
        # Get the x, y coordinaets of points of the polygons that make up
        # the outline of each object instance. These are stores in the
        # shape_attributes (see json format above)
        # The if condition is needed to support VIA versions 1.x and 2.x.
        # f type(a['regions']) is dict:
        #     polygons = [r['shape_attributes'] for r in a['regions'].values()]
        #     names = [r['region_attributes']['name'] for r in a['regions'].values()]
        #     print("is a dict")
        # else:
        print(a['filename'])

        names = [r['region_attributes']['name'] for r in a['regions']]
        if extension == ".json":
            image_path = os.path.join(dataset_dir, a['filename'])
        else:
            image_path = a['img_path']

        # load_mask() needs the image size to convert polygons to masks.
        # Unfortunately, VIA doesn't include it in JSON, so we must read
        # the image. This is only managable since the dataset is tiny.

        image = np.array(skimage.io.imread(image_path))
        #image = np.transpose(image, (1, 0, 2))
        if len(image.shape) < 3:
            image = np.r_[image[None, :], image[None, :], image[None,:]]
            image = image.transpose(1, 2, 0)

        image = image[:,:,[2, 1, 0]]
        # cv2.imshow('image', image[:,:,[2, 1, 0]])
        # cv2.waitKey(0)

        try:
            z_imgpath = os.path.join(dataset_dir, a['zDepth_path'], a['zDepth_filename'])
            image_z = np.array(skimage.io.imread(z_imgpath))
            try:
                image_z.shape[2]
            except Exception:
                image_z = np.r_[image_z[None,:], image_z[None,:], image_z[None,:]]
                image_z = image_z.transpose(1,2,0)
            image = np.append(image, image_z, axis=2)
            image = image[:,:,0:4]
        except Exception:
            bp = 0

        name_dict = {"plane": 1, "glider": 2, "kite": 3, "quadcopter": 4, "eagle": 5}
        #name_dict = {"plane": 1, "glider": 2, "quadcopter": 4, "eagle": 5}

        name_id = []
        for itm in names:
            try:
                name_id.append(name_dict[itm])
            except:
                for k in range(a['regions'].__len__() - 1, -1, -1):
                    if a['regions'][k]['region_attributes']['name'] == itm:
                        del a['regions'][k]

        polygons = [r['shape_attributes'] for r in a['regions']]

        # if a['regions']:
        #     new_obj = add_object(image, polygons, finalImgSz)
        #     data.extend(new_obj[:])
        #     labels.extend(name_id[:])

        data.extend([image])

    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int64)

def read_data(images, reshape=True, dtype=np.float32):

    if reshape:
        assert images.shape[3] > 1, "Images do not have 3 or 4 channels"
        images = images.reshape(images.shape[0], images.shape[1]*images.shape[2]*images.shape[3])

    if dtype == np.float32:
        if images.dtype == np.float32:
            images = np.multiply(images, 1.0/255.0)
        elif images.dtype == np.uint8:
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)

    return images


def create_sprite(data):

    #from medium.com "How to visualize feature vectors with sprites and TensorFlow's TensorBoard by Andrew B. Martin

    # For B&W or greyscale images
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0,0), (0,0), (0,0))
    data = np.pad(data, padding, mode='constant', constant_values=0)

    # Tile Images into sprite
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3, 4))

    data = data.reshape((n * data.shape[1], n* data.shape[3]) + data.shape[4:])

    return data

