
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np

def visualize_ground_truth_vKITTI(dataset, input_size):
    # dataset = dataset.shuffle(buffer_size=len(list(dataset)))
    fig, axes = plt.subplots(1, 10, figsize=(25, 10))  # Create 5 subplots in a row

    for i, (image, (bbox, orientation)) in enumerate(dataset.take(10)):
        
        width, height, _ = input_size
        ax = axes[i]
        ax.imshow(image.numpy())
        ax.set_title(f"Sample {i+1}")
        ax.axis('off')

        for bbox_label in bbox.numpy():
            if np.all(bbox_label == 0):
                continue
        
            left = bbox_label[0] * height
            right = bbox_label[1] * width
            top = bbox_label[2] * height
            bottom = bbox_label[3] * width
            bbox_width = right - left
            bbox_height = bottom - top
    
            rect = patches.Rectangle((left, top), bbox_width, bbox_height, linewidth=3, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    plt.tight_layout()
    plt.show()
    
    

def visualize_ground_truth_KITTI(dataset):
    # Function to plot images with ground truth bounding boxes
    for image, (bbox_labels, orientation_labels) in dataset.take(1):
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
        
        rect = patches.Rectangle((xmin, height - ymax), bbox_width, bbox_height, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    plt.show()


# Plot results after training
def predict_and_plot(model, test_dataset):
    log_dir = "/home/student/e/ehagensieker/programming/HPC_files/Programming/Colab"
    
    for i, (images, (true_boxes, true_labels)) in enumerate(test_dataset.take(1)):
        # Make predictions
        predictions = model.predict(images)
        
        # Extract predicted bounding boxes and labels
        pred_boxes = predictions[0]  # Adjust the index based on your model output
        pred_labels = predictions[1] # Adjust the index based on your model output
        
        # Iterate over a few samples to plot
        for j in range(min(5, len(images))):  # Number of samples to plot
            # Convert TensorFlow tensors to NumPy arrays (only if needed)
            image = images[j].numpy()  
            image = np.clip(image * 255, 0, 255).astype(np.uint8)  # Scale and convert to uint8
            
            true_box = true_boxes[j] 
            true_label = true_labels[j] 
            
            # Plot image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)

            height, width, _ = image.shape
            # Draw true bounding boxes
            # for box in true_box:
 
            #     ymin = box[0] * height
            #     xmin = box[1] * width
            #     ymax = box[2] * height
            #     xmax = box[3] * width
              
            #     plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
            #                                        edgecolor='green', facecolor='none', linestyle='--', linewidth=2))
            #     plt.text(xmin, ymin, 'True', color='green', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

            # Draw predicted bounding boxes
            for box in pred_boxes[j]:
                ymin = box[0] * height
                xmin = box[1] * width
                ymax = box[2] * height
                xmax = box[3] * width
              
                plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                                   edgecolor='red', facecolor='none', linestyle='-', linewidth=2))
                #plt.text(xmin, ymin, 'Pred', color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

            plt.title(f'Sample {j}')
            
            # Save the plot
            image_filename =  log_dir + "/sample_" + str(i) + "_image_" + str(j) + ".png"
            plt.savefig(image_filename)
            plt.close()
