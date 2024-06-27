
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


def data_generator(grouped_data, input_shape):
    for filename, bboxes, labels in grouped_data:
        image = read_and_preprocess_image(filename, input_shape)
        yield image, (bboxes.astype(np.int32), labels.astype(np.int32))

def load_virtual_kitti_dataset(image_folder, bbox_file, pose_file, input_shape, split_ration = 0.8, model_type = "2D"):
    image_filenames = sorted([os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith('.jpg')])

    bbox_data = pd.read_csv(bbox_file, delim_whitespace=True)
    print(bbox_data["frame"])

    # Read pose data and select required columns
    pose_data = pd.read_csv(pose_file, delim_whitespace=True)

    print(bbox_data)
    grouped_data = []
    for filename in image_filenames:
        frame_id = os.path.splitext(os.path.basename(filename))[0]
        frame_id_original = int(frame_id.split('_')[-1])

        # Selecting camera 0's bounding boxes for the given frame
        if frame_id_original in bbox_data[bbox_data['cameraID'] == 0]['frame'].values:
            bboxes = bbox_data[(bbox_data['frame'] == frame_id_original) & (bbox_data['cameraID'] == 0)][['left', 'right', 'top', 'bottom']].values
        else:
            bboxes = np.zeros((1, 4), dtype=np.float32)
        grouped_data.append((filename, bboxes))
    
    padded_data = []
    max_objects = 16  # Maximum number of objects per image (bboxes and orientations)
    for filename, bboxes in grouped_data:
        padded_bboxes = np.zeros((max_objects, 4), dtype=np.float32)
        padded_labels = np.ones((max_objects,), dtype=np.int32) 
               
        num_bboxes = min(max_objects, bboxes.shape[0])
        
        padded_bboxes[:num_bboxes, :] = bboxes[:num_bboxes, :]
        padded_labels[:num_bboxes] = 0  
              
        padded_data.append((filename, padded_bboxes, padded_labels))

    dataset = tf.data.Dataset.from_generator(lambda: data_generator(padded_data, input_shape),
                                             output_signature=(tf.TensorSpec(shape=input_shape, dtype=tf.float32),
                                                               (tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                                                                tf.TensorSpec(shape=(None,), dtype=tf.int32))))
    
    num_samples = len(padded_data)
    train_size = int(num_samples * 0.8)
    
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
    max_objects = 15
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
    bbox = tf.pad(bbox, [[0, 16 - tf.shape(bbox)[0]], [0, 0]], constant_values=0)
    label = tf.pad(label, [[0, 16 - tf.shape(label)[0]]], constant_values=1)

    return image, (bbox, label)

def preprocess(example, input_shape):
    # access relevant information only
    image = example['image']
    bbox = example['objects']['bbox']
    label = example['objects']['type']
    
    # Preprocess each image, bbox, label
    image, (bbox, label) = preprocess_image(image, bbox, label, input_shape)

    return image, (bbox, label)

def load_kitti_dataset(split, input_shape, model_type = "2D"):
    # Load dataset
    dataset = tfds.load('kitti', split=split)

    # Apply preprocessing function to dataset
    dataset = dataset.map(lambda example: preprocess(example, input_shape))
   # dataset = dataset.filter(lambda image, labels: image is not None)


    return dataset