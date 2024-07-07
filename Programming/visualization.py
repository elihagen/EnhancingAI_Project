
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np

def visualize_ground_truth_vKITTI(dataset):
    fig, axes = plt.subplots( 1, figsize=(10, 10))

    for i, (image, (bbox, orientation)) in enumerate(dataset.take(1)):
        axes.imshow(image.numpy())
        axes.set_title(f"Sample {i+1}")
        axes.axis('off')

        for bbox_label in bbox.numpy():
            if np.all(bbox_label == 0):
                continue
            left, right, top, bottom = bbox_label
            bbox_width = right - left
            bbox_height = bottom - top
    
            rect = patches.Rectangle((left, top), bbox_width, bbox_height, linewidth=3, edgecolor='r', facecolor='none')
            axes.add_patch(rect)

    plt.tight_layout()
    plt.show()
    
    

def visualize_ground_truth_KITTI(dataset):
    # Function to plot images with ground truth bounding boxes
    for image, (bbox_labels, orientation_labels) in dataset.take(1):
        print("Image: ", image[0], "bbox: ", bbox_labels[0])
        # Plot image with ground truth bounding boxes
        plot_image_with_bbox(image[0], bbox_labels[0], orientation_labels[0])

def plot_image_with_bbox(image, bboxes, orientations):
    height, width, _ = image.shape
    fig, ax = plt.subplots(1)
    
    ax.imshow(image)
    
    for bbox in bboxes:
        if tf.reduce_all(tf.equal(bbox, 0)):
            continue
        
        # KITTI bbox format: [left, top, right, bottom]        
        ymin = bbox[0] * height
        xmin = bbox[1] * width
        ymax = bbox[2] * height
        xmax = bbox[3] * width

        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        print("ymax", height - ymax)
        print("ymin", height - ymin)

        rect = patches.Rectangle((xmin, height - ymax), bbox_width, bbox_height, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    plt.show()
