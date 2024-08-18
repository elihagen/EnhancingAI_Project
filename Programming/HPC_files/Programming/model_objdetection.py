import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout



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

def create_model(input_shape):
    # Load VGG16 as the backbone
    vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))
    vgg.trainable = False

    # Flatten the output of VGG16
    flatten = vgg.output
    flatten = Flatten()(flatten)

    # Bounding box head
    bboxHead = Dense(128, activation="relu")(flatten)
    bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(32, activation="relu")(bboxHead)
    bboxHead = Dense(63 * 4, activation="linear", name="bounding_box")(bboxHead)
    bboxHead = layers.Reshape((63, 4), name="bbox_reshaped")(bboxHead)

    # Label head
    softmaxHead = Dense(512, activation="relu")(flatten)
    softmaxHead = Dropout(0.5)(softmaxHead)
    softmaxHead = Dense(512, activation="relu")(softmaxHead)
    softmaxHead = Dropout(0.5)(softmaxHead)
    softmaxHead = Dense(63, activation="sigmoid", name="class_label")(softmaxHead)
    softmaxHead = layers.Reshape((63,), name="label_reshaped")(softmaxHead)

    # Define the model
    model = Model(inputs=vgg.input, outputs=[bboxHead, softmaxHead])

    return model
