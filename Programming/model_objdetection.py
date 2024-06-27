import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import ResNet50


def VoxelRCNN2D(input_shape, num_classes=1):
    # For demonstration, using a simple model - should use MobileNetV2 as above
    backbone = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)

    # Remove the top classification layers
    backbone_output = backbone.output

    # Add custom head for object detection
    x = layers.GlobalAveragePooling2D()(backbone_output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)

    # Output layers for bounding box regression and orientation prediction
    bbox_output = tf.keras.layers.Dense(16 * 4, name='bbox_output')(x)  
    label_output = tf.keras.layers.Dense(16, activation='sigmoid', name='label_output')(x) 

    # Reshape the outputs
    bbox_output = tf.keras.layers.Reshape((16, 4), name="bbox_reshaped")(bbox_output)
    label_output = tf.keras.layers.Reshape((16,), name="label_reshaped")(label_output)

    # Define the model
    model = tf.keras.Model(inputs=backbone.input, outputs=[bbox_output, label_output])

    return model


def VoxelRCNN3D(input_shape, num_classes=1):
    # Replace this with Voxel R-CNN backbone or a similar 3D convolutional backbone
    # For demonstration, use a simpler backbone (MobileNetV2) and adjust to your needs
    backbone = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)

    # Remove the top classification layers
    backbone_output = backbone.output

    # Add custom head for object detection
    x = layers.GlobalAveragePooling2D()(backbone_output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)

    # Output layers for bounding box regression and orientation prediction
    bbox_output = layers.Dense(16 * 4, name='bbox_output')(x)  # 16 bboxes, each with 4 coordinates (x, y, w, h)
    orientation_output = layers.Dense(16 * 3, name='orientation_output')(x)  # 16 orientations, each with 3 dimensions

    # Reshape the outputs
    bbox_output = layers.Reshape((16, 4), name="bbox_reshaped")(bbox_output)
    orientation_output = layers.Reshape((16, 3), name="orientation_reshaped")(orientation_output)

    # Define the model
    model = tf.keras.Model(inputs=backbone.input, outputs=[bbox_output, orientation_output])

    return model