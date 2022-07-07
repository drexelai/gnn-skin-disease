"""
Author: QTVo
Pretrained VGG16 U-Net
Code reference from: https://idiotdeveloper.com/
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)

    return x

def build_vgg16_unet(input_shape):
    inputs = Input(input_shape)
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)

    conv1 = vgg16.get_layer("block1_conv2").output         
    conv2 = vgg16.get_layer("block2_conv2").output         
    conv3 = vgg16.get_layer("block3_conv3").output       
    conv4 = vgg16.get_layer("block4_conv3").output        
    connected_layer = vgg16.get_layer("block5_conv3").output       
    decoder_1 = decoder_block(connected_layer, conv4, 256)                    
    decoder_2 = decoder_block(decoder_1, conv3, 128)                    
    decoder_3 = decoder_block(decoder_2, conv2, 64)                    
    decoder_4 = decoder_block(decoder_3, conv1, 32)                      

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(decoder_4)

    model = Model(inputs, outputs, name="VGG16_U-Net")
    return model

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_vgg16_unet(input_shape)
    model.summary()