
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

def adjust_bboxes(bboxes, original_shape, input_shape):
   
    adjusted_bboxes = bboxes.astype(np.float32).copy()
    adjusted_bboxes[:, [0, 1]] = (adjusted_bboxes[:, [0, 1]] / original_shape[0]).astype(np.float32)
    adjusted_bboxes[:, [2, 3]] = (adjusted_bboxes[:, [2, 3]] / original_shape[1]).astype(np.float32)
    
    return adjusted_bboxes

def data_generator(grouped_data, input_shape, original_shape):
    for filename, bboxes, labels in grouped_data:
        base_path = "/content/drive/MyDrive/EAI_data/"
        corrected_filename = base_path + filename.replace("\\", "/")
        image = read_and_preprocess_image(corrected_filename, input_shape)
        bboxes = adjust_bboxes(bboxes, original_shape, input_shape)
        yield image, (bboxes.astype(np.float32), labels.astype(np.int32))

def load_virtual_kitti_dataset(csv_filename, input_shape, original_shape, split_ratio=(0.7, 0.15, 0.15)):
    padded_data = pd.read_csv(csv_filename)
    
    grouped_data = []
    for filename, group in padded_data.groupby('filename'):
        bboxes = group[['left', 'right', 'top', 'bottom']].values
        labels = group['label'].values
        grouped_data.append((filename, bboxes, labels))

    dataset = tf.data.Dataset.from_generator(lambda: data_generator(grouped_data, input_shape, original_shape),
                                             output_signature=(tf.TensorSpec(shape=input_shape, dtype=tf.float32),
                                                               (tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                                                                tf.TensorSpec(shape=(None,), dtype=tf.int32))))
    
    num_samples = len(grouped_data)
    train_size = int(num_samples * split_ratio[0])
    val_size = int(num_samples * split_ratio[1])
    test_size = num_samples - train_size - val_size
    
    dataset = dataset.shuffle(num_samples, reshuffle_each_iteration=False)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)

    return train_dataset, val_dataset, test_dataset

# def adjust_bboxes_kitti(bboxes, input_shape):
#     height_ratio = input_shape[0]
#     width_ratio = input_shape[1] 
    
#     # Convert bboxes to float32 tensor if not already
#     bboxes = tf.cast(bboxes, tf.float32)
    
#     # Adjust bounding boxes
#     adjusted_bboxes = bboxes
#     adjusted_bboxes = tf.stack([
#         adjusted_bboxes[:, 0]  * height_ratio,  # x1
#         adjusted_bboxes[:, 1] *width_ratio,  # x2
#         adjusted_bboxes[:, 2] *height_ratio, # y1
#         adjusted_bboxes[:, 3] * width_ratio  # y2
#     ], axis=-1)
    
#     return tf.cast(adjusted_bboxes, tf.int32)

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
    
    # print(image, label, bbox)

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

def load_kitti_dataset(train_split, val_split, test_split, input_shape):
    train_dataset = tfds.load('kitti', split=train_split)
    val_dataset = tfds.load('kitti', split=val_split)
    test_dataset = tfds.load('kitti', split=test_split)

    # Apply preprocessing
    train_dataset = train_dataset.map(lambda example: preprocess(example, input_shape))
    val_dataset = val_dataset.map(lambda example: preprocess(example, input_shape))
    test_dataset = test_dataset.map(lambda example: preprocess(example, input_shape))

    return train_dataset, val_dataset, test_dataset


def load_combined_dataset(csv_filename, kitti_split, input_shape, original_shape, vkitti_train_ratio):
    kitti_train = load_kitti_dataset(kitti_split[0], kitti_split[1], kitti_split[2], input_shape)
    vkitti_train, vkitti_val, vkitti_test = load_virtual_kitti_dataset(csv_filename, input_shape, original_shape, split_ratio=(vkitti_train_ratio, 0.15, 0.15))
    
    combined_train = kitti_train[0].concatenate(vkitti_train)
    combined_val = kitti_train[1].concatenate(vkitti_val)
    combined_test = kitti_train[2].concatenate(vkitti_test)
    
    return combined_train, combined_val, combined_test