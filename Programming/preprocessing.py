
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow_datasets as tfds

def read_and_preprocess_image(filename, input_shape):
    image = Image.open(filename)
    image = image.resize((input_shape[1], input_shape[0]))  # 1st width, 2nd height
    image = np.array(image) / 255.0  # normalize as above
    return image

# def adjust_bboxes(bboxes, original_shape, input_shape):
#     height_ratio = original_shape[0] / input_shape[0]
#     width_ratio = original_shape[1] / input_shape[1]
    
#     adjusted_bboxes = bboxes.astype(np.float32).copy()
#     adjusted_bboxes[:, [0, 1]] = (adjusted_bboxes[:, [0, 1]] / width_ratio).astype(np.int32)
#     adjusted_bboxes[:, [2, 3]] = (adjusted_bboxes[:, [2, 3]] / height_ratio).astype(np.int32)
    
#     return adjusted_bboxes

def data_generator(grouped_data, input_shape):
    for filename, bboxes, labels in grouped_data:
        image = read_and_preprocess_image(filename, input_shape)
        yield image, (bboxes.astype(np.int64), labels.astype(np.int64))

def load_virtual_kitti_dataset(csv_filename, input_shape, split_ratio = 0.8, model_type = "2D"):
    # Load data from CSV
    padded_data = pd.read_csv(csv_filename)
    
    grouped_data = []
    for filename, group in padded_data.groupby('filename'):
        bboxes = group[['left', 'right', 'top', 'bottom']].values
        labels = group['label'].values
        grouped_data.append((filename, bboxes, labels))

    dataset = tf.data.Dataset.from_generator(lambda: data_generator(grouped_data, input_shape),
                                             output_signature=(tf.TensorSpec(shape=input_shape, dtype=tf.float32),
                                                               (tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                                                                tf.TensorSpec(shape=(None,), dtype=tf.int64))))
    
    num_samples = len(grouped_data)
    train_size = int(num_samples * split_ratio)
    
    # Shuffle the dataset before splitting
    dataset = dataset.shuffle(num_samples, reshuffle_each_iteration=False)

    # Split into training and testing datasets
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    return train_dataset, test_dataset



def preprocess_image(image, bbox, label, input_shape):
    # Filter out non-car objects (car: type = 0)
    
    car_indices = tf.where(label == 0)[:, 0]

    bbox = tf.gather(bbox, car_indices)
    label = tf.gather(label, car_indices)
   
    # filter out images with more objects inside the image
    num_objects = tf.shape(bbox)[0]
    max_objects = 63
    # select max number of bboxes and orientation from the dataset
    bbox = tf.cond(num_objects <= max_objects,
                   lambda: bbox,
                   lambda: bbox[:max_objects])
    
    label = tf.cond(num_objects <= max_objects,
                   lambda: label,
                   lambda: label[:max_objects])
    
    image = tf.image.resize(image, (input_shape[0], input_shape[1]))
    image = image / 255.0 

    # Handling multiple objects in a single image
    # Pad bounding boxes and orientations to a fixed number (16)
    bbox = tf.pad(bbox, [[0, 63 - tf.shape(bbox)[0]], [0, 0]], constant_values=0)
    label = tf.pad(label, [[0, 63 - tf.shape(label)[0]]], constant_values=1)

    # tf.print("Padded Label content:", label, summarize=-1)
    # tf.print("Padded BBox content:", bbox, summarize=-1)

    return image, (bbox, label)

def preprocess(example, input_shape):
    # access relevant information only
    image = example['image']
    bbox = example['objects']['bbox']
    label = example['objects']['type']

    # tf.print("Original Label content:", label, summarize=-1)
    # tf.print("Original BBox content:", bbox, summarize=-1)
    # Preprocess each image, bbox, label
    image, (bbox, label) = preprocess_image(image, bbox, label, input_shape)
    
    return image, (bbox, label)


def load_kitti_dataset(split, input_shape):
    # Load dataset
    dataset = tfds.load('kitti', split=split)

    # Apply preprocessing function to dataset
    dataset = dataset.map(lambda example: preprocess(example, input_shape))
   # dataset = dataset.filter(lambda image, labels: image is not None)


    return dataset


def load_combined_dataset(csv_filename, kitti_split, input_shape, split_ratio, model_type):
    kitti_train = load_kitti_dataset(kitti_split[0], input_shape)
    kitti_test = load_kitti_dataset(kitti_split[1], input_shape)
    
    vkitti_train, vkitti_test = load_virtual_kitti_dataset(csv_filename, input_shape, split_ratio, model_type)
    
    combined_train = kitti_train.concatenate(vkitti_train)
    combined_test = kitti_test
    
    return combined_train, combined_test