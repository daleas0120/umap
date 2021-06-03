# Custom Text Parser for mrcnn project

import re
import pandas as pd
import ntpath
import os
#from sys import platform
from matplotlib import collections
#from mrcnn import visualize

def load(filepath: object) -> object:
    nested_dict = lambda: collections.defaultdict(nested_dict)
    annotations = {}
    # open the file and read through it line by line

    with open(filepath, 'r') as file_object:
        line = file_object.readline()
        while line[0] == "#" or line == "\n":
            line = file_object.readline()
        paths = line.split(",")
        num_paths = len(paths)
        for x in paths:
            x.strip()
        line = file_object.readline()
        platform = os.environ['PLATFORM']
        print("Platform: "+platform)
        if platform == "SL02319_WIN":
            img_dir = paths[0]
        elif platform == "SL02319":
            img_dir = paths[2]
        elif platform == "SL023L1_WIN":
            img_dir = paths[2]
        elif platform == ("LAMBDA1" or "LAMBDA2"):
            img_dir = paths[2]
        else:
            img_dir = paths[1]

        print('Getting data from '+img_dir)
        img_dir.strip("\n")

        while line:
            if len(line) > 0 and line[0] != "#" and line != "\n":
                d = {}  # creates dict for this file

                parts = line.split("{")
                # rgb_img,z_img,region_info = line.split(",")
                # handle rgb_image
                img_files = parts[0].split(",")

                rgb_img = img_files[0]
                rgb_img.strip()  # remove trailing white spaces
                path, name = ntpath.split(rgb_img)
                img_path = os.path.join(img_dir, rgb_img)
                img_size = os.path.getsize(img_path)


                d['filename'] = name
                d['size'] = img_size
                d['img_path'] = img_path

                # print(name)
                parts.remove(parts[0])

                regions_list = []
                gt_boxes = []

                while len(parts) != 0:

                    tmp_data = parts[0].split("}")
                    tmp_data = tmp_data[0].split(",")
                    label = tmp_data[len(tmp_data) - 1]
                    tmp_data.remove(tmp_data[len(tmp_data) - 1])

                    tmp_dict = {}
                    shape_attributes = {'name': "polygon"}

                    if len(tmp_data) > 0:
                        x_coords=[]
                        y_coords=[]
                        if len(tmp_data) == 4:
                            x_1 = int(tmp_data[0])
                            y_1 = int(tmp_data[1])
                            x_2 = int(tmp_data[2]) + x_1
                            y_2 = int(tmp_data[3]) + y_1
                            x_coords = [x_1, x_2, x_1, x_2]
                            y_coords = [y_1, y_1, y_2, y_2]
                        else:
                            for i in range(int(len(tmp_data)/2)):
                                x_coords.append(int(tmp_data[0]))
                                tmp_data.remove(tmp_data[0])
                            while len(tmp_data) != 0:
                                y_coords.append(int(tmp_data[0]))
                                tmp_data.remove(tmp_data[0])
                            x_1 = min(x_coords)
                            y_1 = min(y_coords)
                            x_2 = max(x_coords)
                            y_2 = max(y_coords)

                        if len(x_coords) != len(y_coords):
                            print("ERROR: Number of x_coords does not equal number of y_coords")
                            return

                        shape_attributes['all_points_x'] = x_coords
                        shape_attributes['all_points_y'] = y_coords
                        tmp_dict['shape_attributes'] = shape_attributes

                        gt_boxes.append([y_1, x_1, y_2, x_2])

                    region_attributes = {'name': label}
                    tmp_dict['region_attributes'] = region_attributes
                    regions_list.append(tmp_dict)

                    d.update({'regions': regions_list})
                    parts.remove(parts[0])

                    file_attributes = {}
                    d.update({'file_attributes':file_attributes})
                    annotations.update({img_path:d})

            line = file_object.readline()
    data = [annotations, img_dir]
    return data