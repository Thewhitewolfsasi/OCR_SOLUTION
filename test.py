"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT
from torch.cuda import is_available as cuda_available
from collections import OrderedDict

import h5py 
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, InputLayer, Reshape, MaxPooling2D, Flatten, Activation
from keras import layers
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.utils import plot_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,precision_score, recall_score, f1_score, precision_recall_curve, auc
from pathlib import Path
import cv2
from PIL import Image, ImageFile
# from tensorflow.keras.models import load_model
import matplotlib.image as matimage
# load model
model = load_model('E:\INTERNSHIP\SRI TECH ENGG\CRAFT-pytorch\models\model_distributed1.h5')
print("Model is loaded")



def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default=r'weights\craft_ic15_20k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
parser.add_argument('--char', default=False, type=str2bool, help='Character level split')


args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)


# if not os.path.isdir(result_folder):
#     os.mkdir(result_folder)

from math import atan2, degrees

# def rotate_image(image, angle):
#     # Rotate the image around its center
#     height, width = image.shape[:2]
#     center = (width // 2, height // 2)
#     rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
#     return rotated_image

# def rotate_point(point, center, angle):
#     # Rotate a point around a center by a specified angle
#     x, y = point
#     angle_rad = np.radians(angle)
#     new_x = int((x - center[0]) * np.cos(angle_rad) - (y - center[1]) * np.sin(angle_rad) + center[0])
#     new_y = int((x - center[0]) * np.sin(angle_rad) + (y - center[1]) * np.cos(angle_rad) + center[1])
#     return new_x, new_y


# def show_final_image(image, bboxes, save_path=None, original_filename=None):
#     coordinates_dict = {}

#     # Initialize rotated_image before the loop
#     rotated_image = image.copy()

#     for i, bbox in enumerate(bboxes):
#         # Convert bbox to integers
#         bbox = np.array(bbox)  # Convert to NumPy array for easier manipulation
#         bbox = bbox.astype(int)

#         # Extract coordinates
#         x_coords = bbox[:, 0]
#         y_coords = bbox[:, 1]

#         # Calculate the rotation angle of the bounding box
#         angle = degrees(atan2(y_coords[1] - y_coords[0], x_coords[1] - x_coords[0]))

#         # Rotate the image
#         rotated_image = rotate_image(rotated_image, angle)

#         # Rotate the bounding box points
#         center = (rotated_image.shape[1] // 2, rotated_image.shape[0] // 2)
#         rotated_bbox = np.array([rotate_point(point, center, -angle) for point in bbox])

#         # Draw rotated bounding box on the image
#         cv2.polylines(rotated_image, [rotated_bbox], isClosed=True, color=(0, 255, 0), thickness=2)

#         # Crop the rotated image based on the rotated bounding box
#         xmin, xmax = min(rotated_bbox[:, 0]), max(rotated_bbox[:, 0])
#         ymin, ymax = min(rotated_bbox[:, 1]), max(rotated_bbox[:, 1])
#         cropped_img = rotated_image[ymin:ymax, xmin:xmax]

#         # Save the cropped image if a save path is provided
#         if save_path:
#             # Create a filename based on the box serial number and original filename
#             filename = f'{os.path.basename(original_filename)}_Cropped_Box_{i + 1}'
#             cv2.imwrite(os.path.join(save_path, filename), cropped_img)

#         # Store coordinates in the dictionary
#         coordinates_dict[f'{os.path.basename(original_filename)}_Box_{i + 1}'] = {
#             'x_coords': x_coords.tolist(),
#             'y_coords': y_coords.tolist(),
#             'xmin': xmin,
#             'ymin': ymin,
#             'xmax': xmax,
#             'ymax': ymax,
#             'filename': filename if save_path else None  # Include the filename in the dictionary
#         }

#     # Display the rotated image with bounding boxes
#     cv2.imshow('Final Rotated Image with Bounding Boxes', rotated_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     return coordinates_dict

# def rotate_image(image, angle):
#     center = (image.shape[1] // 2, image.shape[0] // 2)
#     rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
#     return rotated_image

# def rotate_point(point, center, angle):
#     angle_rad = np.deg2rad(angle)
#     x, y = point[0] - center[0], point[1] - center[1]
#     new_x = x * np.cos(angle_rad) - y * np.sin(angle_rad) + center[0]
#     new_y = x * np.sin(angle_rad) + y * np.cos(angle_rad) + center[1]
#     return int(new_x), int(new_y)

def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return rotated_image

def rotate_point(point, center, angle):
    angle_rad = np.deg2rad(angle)
    x, y = point[0] - center[0], point[1] - center[1]
    new_x = x * np.cos(angle_rad) - y * np.sin(angle_rad) + center[0]
    new_y = x * np.sin(angle_rad) + y * np.cos(angle_rad) + center[1]
    return int(new_x), int(new_y)

def calculate_overlap_percentage(box1, box2):
    # Calculate the area of the intersection rectangle
    x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    intersection_area = x_overlap * y_overlap

    # Calculate the area of both rectangles
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the percentage overlap
    overlap_percentage = (intersection_area / min(area_box1, area_box2)) * 100

    return overlap_percentage

def extract_bbox_coordinates(bbox):
    # Convert bbox to NumPy array for easier manipulation
    bbox = np.array(bbox)

    # Extract x and y coordinates
    x_coords = bbox[:, 0]
    y_coords = bbox[:, 1]

    # Calculate xmin, ymin, xmax, ymax
    xmin = int(np.min(x_coords))
    ymin = int(np.min(y_coords))
    xmax = int(np.max(x_coords))
    ymax = int(np.max(y_coords))

    return xmin, ymin, xmax, ymax

def detect_and_rotate_characters(image, word_bboxes, char_bboxes):
    for word_bbox in word_bboxes:
        # Ensure the word_bbox has at least 4 points
        if len(word_bbox) < 4:
            print(f"Invalid word_bbox: {word_bbox}")
            continue

        # Convert word_bbox to integers
        word_bbox = np.array(word_bbox)  # Convert to NumPy array for easier manipulation
        word_bbox = word_bbox.astype(int)

        # Extract coordinates
        x_coords = word_bbox[:, 0]
        y_coords = word_bbox[:, 1]

        # Calculate the rotation angle of the word bounding box
        angle = degrees(atan2(y_coords[1] - y_coords[0], x_coords[1] - x_coords[0]))

        rotated_image_copy = rotate_image(image.copy(), angle)

        print(f"Rotation angle for word from character func: {angle} degrees")

        # Rotate the character bboxes within the word bbox
        for char_bbox in char_bboxes:
            # Check if the center of the character bbox is within the word bbox
            if len(char_bbox) < 2:
                print(f"Invalid char_bbox: {char_bbox}")
                continue

            char_center_x = (char_bbox[0][0] + char_bbox[0][1]) / 2  # Update this line
            center = (rotated_image_copy.shape[1] // 2, rotated_image_copy.shape[0] // 2)

            if (
                word_bbox[0][0] <= char_center_x <= word_bbox[2][0]
                and word_bbox[0][1] <= char_bbox[0][1] <= word_bbox[2][1]
            ):
                # Calculate the relative angle of the character bbox to the word bbox
                char_relative_angle = degrees(atan2(char_bbox[0][1] - word_bbox[0][1], char_bbox[0][0] - word_bbox[0][0]))

                # Calculate the absolute angle of the character bbox by adding the word angle
                rotated_char_relative_angle = char_relative_angle + angle

                # Rotate the character bbox based on the absolute angle np.array([rotate_point(point, center, -angle) for point in bbox])
                rotated_char_bbox = np.array([rotate_point(point, center, rotated_char_relative_angle)] for point in char_bbox)
                print(f"{np.array([po for po in char_bbox])}")
                # rotated_image = cv2.polylines(rotated_image_copy, [rotated_char_bbox], isClosed=True, color=(0, 255, 0), thickness=2)
                # Perform further processing or visualization as needed
                print(f"Rotated character bbox: {type(rotated_char_bbox) }:{rotated_char_bbox}")
                # cv2.imshow('Final Rotated Image with Bounding Boxes', rotated_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def get_text_from_model(image):
    print("\nTEXT FROM MODEL")
    text = ''
    result_dict = {'0':0,'1':1, '2':2, '3':3, '4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'a': 11, 'b': 12, 'c': 13, 'd': 14, 'e': 15, 'f': 16, 
    'g': 17, 'h': 18, 'i': 19, 'j': 20, 'k': 23, 'l': 22, 'm': 21, 'n': 24, 'o': 25, 'p': 26, 'q': 27, 'r': 28, 's': 29, 't': 30, 
    'u': 31, 'v': 32, 'w': 33, 'x': 34, 'y': 35, 'z': 36} #m and k value is interchanged and 10 value is missing
    print(image.shape)
    
    cv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resize_img = cv2.resize(cv_rgb, (32,32))
    print(resize_img.shape)
    arr_img = np.asarray(resize_img)
    print(arr_img.shape)
    cv2.imwrite(r"E:\INTERNSHIP\SRI TECH ENGG\arr_img.jpg",arr_img )
    read = cv2.imread(r"E:\INTERNSHIP\SRI TECH ENGG\arr_img.jpg")
    gray1 = rgb2gray(read) #32,32
    print(arr_img)
    print(gray1)
    # gray = cv2.imread(resize_img,'0')
    gray = cv2.cvtColor(arr_img, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    add_channels = np.expand_dims(gray1, axis=-1) #32,32,1
    print(add_channels.shape)
    img_array = add_channels.astype('float32') / 255
    img_a = np.expand_dims(np.array(img_array), axis=0) #1,32,32,1
    print(img_array.shape)
    print(img_a.shape)
    predict_arr = model.predict(img_a)
    predict_val = predict_arr[-1].round()
    print(predict_val)
    indx = [i for i, value in enumerate(predict_val) if value == 1]
    print(indx)
    if len(indx)==0:
        text = '?'
    else:
        val = [i for i in result_dict if result_dict[i]==indx[0]]
        text = val[0]
    # cv2.imshow('arr_img',arr_img)
    # cv2.imshow('cv_rgb',cv_rgb)
    # cv2.imshow('resize_img',resize_img)
    # cv2.imshow('gray_img',gray)
    # cv2.imshow('gray_img1',gray1)
    # plt.imshow(gray1)
    # plt.show()
    # cv2.imshow('add_channels',add_channels)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return text

def show_final_image(image, bboxes, save_path=None, original_filename=None, expand_size = 10):
    coordinates_dict = {}

    # Initialize rotated_image before the loop
    rotated_image = image.copy()

    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # # Display the original image with bounding boxes
    # axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # axes[0].set_title('Original Image with Bounding Boxes')

    # for bbox in bboxes:
    #     rect = patches.Polygon(bbox, closed=True, edgecolor='red', linewidth=2, fill=False)
    #     axes[0].add_patch(rect)

    for i, bbox in enumerate(bboxes):
        # Convert bbox to integers
        bbox = np.array(bbox)  # Convert to NumPy array for easier manipulation
        bbox = bbox.astype(int)

        # Extract coordinates
        x_coords = bbox[:, 0]
        y_coords = bbox[:, 1]

        #########
        #og image bounding box
        ogxmin, ogxmax = min(x_coords) - expand_size, max(x_coords) + expand_size
        ogymin, ogymax = min(y_coords) - expand_size, max(y_coords) + expand_size

        # Ensure the bounds are within the image dimensions
        ogxmin = max(0, ogxmin)
        ogymin = max(0, ogymin)
        ogxmax = min(rotated_image.shape[1], ogxmax)
        ogymax = min(rotated_image.shape[0], ogymax)
        #########

        # Calculate the rotation angle of the bounding box
        angle = degrees(atan2(y_coords[1] - y_coords[0], x_coords[1] - x_coords[0]))

        # Rotate a copy of the original image
        rotated_image_copy = rotate_image(image.copy(), angle)

        # Rotate the bounding box points
        center = (rotated_image_copy.shape[1] // 2, rotated_image_copy.shape[0] // 2)
        print(f"Rotation angle for word from show_image: {angle} degrees" )
        og_rotated_bbox = np.array([point for point in bbox])
        
        rotated_bbox = np.array([rotate_point(point, center, -angle) for point in bbox])

        # Draw rotated bounding box on the rotated image
        saveimage = rotated_image_copy.copy()
        cv2.polylines(saveimage, [rotated_bbox], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(rotated_image, [og_rotated_bbox],isClosed=True, color=(0, 255, 255), thickness=1)
        # cv2.imwrite(f'/home/sasi/Music/result/saveimage_{i}.jpg',saveimage)
        # Crop the rotated image based on the rotated bounding box
        # xmin, xmax = min(rotated_bbox[:, 0]), max(rotated_bbox[:, 0])
        # ymin, ymax = min(rotated_bbox[:, 1]), max(rotated_bbox[:, 1])

        # Crop the rotated image based on the rotated bounding box by increase size by 10 pixels in all directions
        xmin, xmax = min(rotated_bbox[:, 0]) - expand_size, max(rotated_bbox[:, 0]) + expand_size
        ymin, ymax = min(rotated_bbox[:, 1]) - expand_size, max(rotated_bbox[:, 1]) + expand_size

        # Ensure the bounds are within the image dimensions
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(rotated_image_copy.shape[1], xmax)
        ymax = min(rotated_image_copy.shape[0], ymax)

        cropped_img = rotated_image_copy[ymin:ymax, xmin:xmax]
        # cropped_img = rotated_image_copy[xmin:xmax, ymin:ymax]

        og_croppedimg = rotated_image[ogymin:ogymax, ogxmin:ogxmax]
        # print(type(og_croppedimg))
        # print(type(cropped_img))
        # cv2.imshow('cropped img',cropped_img)
        # cv2.imshow('ogcropped img',og_croppedimg)
        # base1 = os.path.basename(original_filename)
        # filename1 = f'{os.path.splitext(base1)[0]}{os.path.splitext(base1)[1]}'
        # print(f'FILENAME --- {filename1}')
        # saveimage1 = cv2.imread(os.path.join(save_path, filename1))
        # cv2.imshow('rotated_image',rotated_image)
        predicted_text = get_text_from_model(cropped_img)
        # saveimage1 = cv2.imread('/home/sasi/Music/result/textimg.jpeg')
        text_img = cv2.putText(rotated_image,predicted_text,(ogxmin,ogymin), cv2.FONT_HERSHEY_PLAIN,
                                1.5, (0,0,0),2, cv2.LINE_8)
        # cv2.imshow('text_img',text_img)
        # cv2.imwrite('/home/sasi/Music/result/textimg.jpeg',text_img)
        # cv2.imwrite(os.path.join(save_path, filename1), text_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the cropped image if a save path is provided
        if save_path:
            # Create a filename based on the box serial number and original filename
            base = os.path.basename(original_filename)
            filename = f'{os.path.splitext(base)[0]}_Box_{i + 1}{os.path.splitext(base)[1]}'
            # cv2.imwrite(os.path.join(save_path, filename), saveimage)
            
            

        # Store coordinates in the dictionary with respect to the original image
        original_x_coords = np.array(x_coords, dtype= np.int64).tolist()
        original_y_coords = np.array(y_coords, dtype= np.int64).tolist()
        rotated_bbox_list = np.array(rotated_bbox, dtype= np.int64).tolist()
        rotated_x_coords = np.array(rotated_bbox[:, 0], dtype= np.int64).tolist()
        rotated_y_coords = np.array(rotated_bbox[:, 1], dtype= np.int64).tolist()
        
        coordinates_dict[f'{os.path.splitext(base)[0]}_Box_{i + 1}'] = {
            'filename': filename if save_path else None,
            'original_x_coords': original_x_coords,
            'original_y_coords': original_y_coords,
            'rotated_x_coords': [point[0] for point in rotated_bbox_list],
            'rotated_y_coords': [point[1] for point in rotated_bbox_list],
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
        }
    
    if save_path and (len(bboxes)!=0):
        cv2.imwrite(os.path.join(save_path, filename), rotated_image)

    # Display the rotated image with bounding boxes using matplotlib
    # axes[1].imshow(cv2.cvtColor(rotated_image_copy, cv2.COLOR_BGR2RGB))
    # axes[1].set_title('Rotated Image with Bounding Boxes')

    # # Plot the rotated bounding boxes
    # for rotated_bbox in [bbox for bbox in coordinates_dict.values()]:
    #     rect = patches.Polygon(
    #         list(zip(rotated_bbox['rotated_x_coords'], rotated_bbox['rotated_y_coords'])),
    #         closed=True, edgecolor='red', linewidth=2, fill=False)
    #     axes[1].add_patch(rect)

    # plt.show()

    return coordinates_dict

def get_image_stats(img_path):#, lbl):
    img = cv2.imread(img_path)
    filename, file_ext = os.path.splitext(os.path.basename(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)
    no_text = gray * ((gray/blurred)>0.99)                     # select background only
    no_text[no_text<10] = no_text[no_text>20].mean()           # convert black pixels to mean value
    no_bright = no_text.copy()
    no_bright[no_bright>220] = no_bright[no_bright<220].mean() # disregard bright pixels
    # cv2.imshow("no_text", no_text)
    # cv2.waitKey(0)
    # print(lbl)
    std = no_bright.std()
    print('STD:', std)
    bright = (no_text>220).sum()
    print('Brigth pixels:', bright)
    plt.figure()
    plt.hist(no_text.reshape(-1,1), 25)
    plt.title(filename)
    plt.show()
    
    if std>25:
        print("!!! Detected uneven illumination")
    if no_text.mean()<200 and bright>8000:
        print("!!! Detected glare")

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, chara, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, chara, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]
    
    # print(f"\ntest.py ratio_w -- \n{ratio_w}\n")
    # print(f"\ntest.py ratio_h -- \n{ratio_h}\n")
    # print(f"\ntest.py boxes -- \n{boxes}\n")

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


# if __name__ == '__main__':
#     # load net
#     net = CRAFT()     # initialize

#     print('Loading weights from checkpoint (' + args.trained_model + ')')
#     # if args.cuda:
#     #     net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
#     # else:
#     #     net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

#     # if args.cuda:
#     #     net = net.cuda()
#     #     net = torch.nn.DataParallel(net)
#     #     cudnn.benchmark = False
#     print('Loading weights from checkpoint (' + args.trained_model + ')')
#     device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
#     net = net.to(device)
#     if device.type == 'cuda' and args.cuda:
#         net = torch.nn.DataParallel(net)
#         cudnn.benchmark = False



#     # net.eval()

#     # LinkRefiner
#     refine_net = None
#     if args.refine:
#         from refinenet import RefineNet
#         refine_net = RefineNet()
#         print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
#         if args.cuda:
#             refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
#             refine_net = refine_net.cuda()
#             refine_net = torch.nn.DataParallel(refine_net)
#         else:
#             refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

#         refine_net.eval()
#         args.poly = True
if __name__ == '__main__':

    def convert_to_builtin_type(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            raise TypeError("Object of type {} is not serializable".format(type(obj)))
            
    # load net
    net = CRAFT()     # initialize
    train_model = r'E:\INTERNSHIP\SRI TECH ENGG\CRAFT-pytorch\weights\craft_ic15_20k.pth'

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    
    device = torch.device('cuda' if cuda_available() and args.cuda else 'cpu')
    if device.type == 'cuda':
        print("cu")
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
        net = net.to(device)
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    else:
        print("cp")
        print(str(args.trained_model))
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
        print

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        print(device.type)
        if device.type == 'cuda':
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.to(device)
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))
            print("Running in cpu")

        refine_net.eval()
        args.poly = True

    t = time.time()
    image_list1 = []

    # capture = cv2.VideoCapture(0)
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # while True:
    #     (grabbed, frame) = capture.read()

    #     if not grabbed:
    #         break

    #     # Resize frame
    #     width = 1500
    #     height = 1000
    #     dim = (width, height)
    #     frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    #     cv2.imshow('RGB', frame)
        

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         print('turning off the camera')
    #         break

    #     if cv2.waitKey(1) == ord('h') or cv2.waitKey(1) == ord('H'):
    #         image_path = r'E:\INTERNSHIP\SRI TECH ENGG\merge_pic.jpg'
    #         # capture.set(cv2.CAP_PROP_BRIGHTNESS, 255/2) # Setting Exposure
    #         (grabbed, frame) = capture.read() # Updating frame
    #         print('frame captured')
    #         if grabbed:
    #             cv2.imshow('RGB', frame) #Display
    #             cv2.imwrite(image_path, frame)
    #             print('image captured')
            

        # load data
    for k, image_path in enumerate(image_list):
        # print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        # get_image_stats(image_path)
        image = imgproc.loadImage(image_path)
        rgbimg = matimage.imread(image_path)
        print('image is processing')
        # grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a colormap for thermal effect (e.g., COLORMAP_JET)
        # thermal_image = cv2.applyColorMap(grey_img, cv2.COLORMAP_JET)

            

        # show_final_image(image, bboxes)
        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, args.char, refine_net)
        print(f'Text box is obtained from image {bboxes}')

            # # Extract text from each bounding box
            # for i, bbox in enumerate(bboxes):
            #     try:
            #         # Extract text from the cropped region of the image using OCR
            #         extracted_text = extract_text(image, bbox)

            #         # Print or store the extracted text
            #         print(f"Text from bounding box {i + 1}: {extracted_text}")

            #         # You can save the extracted text or perform any further processing here
            #     except Exception as e:
            #         print(f"Error processing bounding box {i + 1}: {str(e)}")
                

            # save score text
        result_folder = '/home/sasi/Documents/wha/normal/engine/result_jet/'
        result_folder1 = '/home/sasi/intern/Bush_load/darken_text/result/'
        result_folder2 = r'e:\INTERNSHIP\SRI TECH ENGG\output\result folder'
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder2 + "\\res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

            # file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)
            # Specify the path to save the annotated image
        save_path = '/home/sasi/intern/tesseract/engineimg/wordimg/'
        save_path1 = '/home/sasi/Music/textfolder/'
        save_path2 = r'e:\INTERNSHIP\SRI TECH ENGG\output\model a to z'
        save_path3 = r'e:\INTERNSHIP\SRI TECH ENGG\output\save folder1'

            # Show the final image with warped bounding boxes and save coordinates
        coordinates_dict = OrderedDict()
        coordinates_dict = show_final_image(image, bboxes, save_path2, original_filename=image_path, expand_size = 2)
        print('coordinate is saved')
            # print(type(coordinates_dict))
            # print(f"Coordinates Dictionary for Image {k + 1}:", coordinates_dict)
            

            # json_file_path = "/home/sasi/intern/dict1.json"
            # with open(json_file_path, "w") as f:
            #     json.dump(coordinates_dict, f, default=convert_to_builtin_type)

            # # Later, to read the dictionary back
            # with open(json_file_path, "r") as f:
            #     loaded_coordinates_dict = json.load(f)

            # Now, 'loaded_coordinates_dict' contains the dictionary loaded from the JSON file
            # print(loaded_coordinates_dict)
        
# capture.release() 
# # Destroy all the windows 
# cv2.destroyAllWindows() 

print("elapsed time : {}s".format(time.time() - t))