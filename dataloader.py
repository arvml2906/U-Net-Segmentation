import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_data(path, split=0.1):
    image_files = os.listdir(image_path)
    mask_files = os.listdir(mask_path)
    # Extract common base names from mask filenames
    mask_base_names = [fName.split(".png")[0] for fName in mask_files]

    # Create a list of image file names using mask base names
    image_file_names = [fName.split("_mask")[0] for fName in mask_base_names]

    image_files = sorted(image_file_names)
    mask_files = sorted(mask_base_names)
    split_size = int(len(images) * split)

    train_x, valid_x = train_test_split(
        image_files, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(
        mask_files, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(
        train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(
        train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def getData(X_shape, flag="train"):
    im_array = []
    mask_array = []

    if flag == "test":
        for x, y in tqdm(testing_files):
            im = cv2.resize(
                cv2.imread(
                    os.path.join(
                        image_path,
                        x + ".png")),
                (X_shape,
                 X_shape))[
                :,
                :,
                0]
            mask = cv2.resize(
                cv2.imread(
                    os.path.join(
                        mask_path,
                        y + ".png")),
                (X_shape,
                 X_shape))[
                :,
                :,
                0]

            im_array.append(im)
            mask_array.append(mask)

        return im_array, mask_array

    if flag == "train":
        for x, y in tqdm(training_files):
            im = cv2.resize(
                cv2.imread(
                    os.path.join(
                        image_path,
                        x + ".png")),
                (X_shape,
                 X_shape))[
                :,
                :,
                0]
            mask = cv2.resize(
                cv2.imread(
                    os.path.join(
                        mask_path,
                        y + ".png")),
                (X_shape,
                 X_shape))[
                :,
                :,
                0]

            im_array.append(im)
            mask_array.append(mask)

        return im_array, mask_array

    if flag == "val":
        for x, y in tqdm(validation_files):
            im = cv2.resize(
                cv2.imread(
                    os.path.join(
                        image_path,
                        x + ".png")),
                (X_shape,
                 X_shape))[
                :,
                :,
                0]
            mask = cv2.resize(
                cv2.imread(
                    os.path.join(
                        mask_path,
                        y + ".png")),
                (X_shape,
                 X_shape))[
                :,
                :,
                0]

            im_array.append(im)
            mask_array.append(mask)

        return im_array, mask_array
