
from visualization import visualize_ground_truth_vKITTI, visualize_ground_truth_KITTI
from model_objdetection import VoxelRCNN2D, VoxelRCNN3D, create_model
from preprocessing import load_virtual_kitti_dataset, load_kitti_dataset, load_combined_dataset
import tensorflow_datasets as tfds
import tensorflow as tf

import argparse


def main(dataset_type, model_type):
    # Input image shape
    input_shape = (224, 224, 3)
    
    # load kitti datasetpi
    if dataset_type == "kitti":
        train_dataset = load_kitti_dataset('train[:80%]', input_shape)
        test_dataset = load_kitti_dataset('train[80%:]', input_shape) 
        train_dataset = train_dataset.padded_batch(64, padded_shapes=([224,224, 3], ([63, 4], [63])))
        test_dataset = test_dataset.padded_batch(64, padded_shapes=([224,224, 3], ([63, 4], [63])))
        visualize_ground_truth_KITTI(test_dataset)
        
    # load virtual kitti dataset
    elif dataset_type == "vkitti": 
        csv_file =  r"C:\Arbeitsordner\Abgaben_repo\padded_data_vkitti_scene2.csv"
        split_ratio = 0.8
        # image_folder = r'C:\Arbeitsordner\Abgaben_repo\vkitti_2.0.3_rgb\Scene01\15-deg-left\frames\rgb\Camera_0'
        # bbox_file = r'C:\Arbeitsordner\Abgaben_repo\vkitti_2.0.3_textgt\Scene01\15-deg-left\bbox.txt'
        # pose_file = r'C:\Arbeitsordner\Abgaben_repo\vkitti_2.0.3_textgt\Scene01\15-deg-left\pose.txt'
        
        train_dataset, test_dataset = load_virtual_kitti_dataset(csv_file, input_shape, split_ratio, model_type)
        visualize_ground_truth_vKITTI(train_dataset)
        train_dataset = train_dataset.batch(32)
        test_dataset = test_dataset.batch(32)
        
    elif dataset_type == "combined":
        csv_file = r"C:\Arbeitsordner\Abgaben_repo\padded_data_vkitti_scene2.csv"
        kitti_split = ['train[:80%]', 'train[80%:]']
        train_dataset, test_dataset = load_combined_dataset(csv_file, kitti_split, input_shape, 0.8, model_type)
        train_dataset = train_dataset.batch(32)
        test_dataset = test_dataset.batch(32)

    if model_type == "2D":
        model = create_model(input_shape)
        # model = VoxelRCNN2D(input_shape)    
    elif model_type == "3D": 
        model = create_model(input_shape)
        # model = VoxelRCNN3D(input_shape)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss={'bbox_reshaped': 'mse', 'label_reshaped': 'binary_crossentropy'},
                  metrics={'bbox_reshaped': 'mae', 'label_reshaped': tf.keras.metrics.BinaryAccuracy(threshold=0.5)})

    
    # Train the model
    model.fit(train_dataset, epochs=5, validation_data=test_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object Detection with VoxelRCNN')
    parser.add_argument('--dataset', type=str, default='kitti', choices=['kitti', 'vkitti', 'combined'],
                        help='Dataset type: kitti, vkitti or combined')
    parser.add_argument('--complexity', type=str, default='2D', choices=['2D', '3D'],
                        help='Model complexity: 2D or 3D')
    args = parser.parse_args()

    # Call main function with parsed arguments
    main(args.dataset, args.complexity)
    