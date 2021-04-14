#from .load_data import COVIDGR
import argparse
import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()

config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.flatten(y_true)
    y_pred_f = tf.keras.flatten(y_pred)
    intersection = tf.keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.keras.sum(y_true_f) + tf.keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

parser = argparse.ArgumentParser(description='Process directory images.')
parser.add_argument('--directory',"-d", type=str,default="data/COVIDGR-09-04/Revised/",
                    help='Directory to process')
parser.add_argument('--segment-model',"-sm", type=str,default="COVID-XAI/segmentation_model/unet_lung_seg.hdf5",
                    help='Segmentation model to process')
parser.add_argument('--output',"-o", type=str,default="data/COVIDGR-09-04/Revised-croped/",
                    help='Output directory of croped images')

def crop_numbers(mascara):

    top,bot = 0,len(mascara)
    left,rigth = 0,len(mascara[0])

    for i in range(len(mascara)):
        if np.max(mascara[i,:]) > 0.5:
            top = i
            break
    for i in range(len(mascara)-1,-1,-1):
        if np.max(mascara[i,:]) > 0.5:
            bot = i
            break
    
    for i in range(len(mascara[0])):
        if np.max(mascara[:,i]) > 0.5:
            left = i
            break
    for i in range(len(mascara[0])-1,-1,-1):
        if np.max(mascara[:,i]) > 0.5:
            rigth = i
            break
    rango1 = int(bot-top)
    rango2 = int(rigth-left)
    '''
    if rango1 < 100:
        top,left =  top-50,left+50
    

    if rango2 < 100:
        left,rigth = left-50,rigth-50
    '''
    return max(top,0),min(bot,len(mascara)),max(left,0),min(rigth,len(mascara[0]))


if __name__ == "__main__":
    args = parser.parse_args()
    directory = args.directory
    output_dir = args.output
    sm_name = args.segment_model

    sm = tf.keras.models.load_model(sm_name,custom_objects={"dice_coef_loss":dice_coef_loss,"dice_coef":dice_coef})
    images_names = os.listdir(directory)
    clahe = cv2.createCLAHE(clipLimit = 5)
    for i,image_name in enumerate(images_names):
        source = directory+image_name
        out = output_dir+image_name

        image = cv2.imread(source,cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image,(512,512))/255
        image = np.expand_dims(image,-1)
        prediction = sm(np.array([image]))

        image = image[:,:,0]
        prediction = prediction[0,:,:,0]
        
        top,bot,left,rigth = crop_numbers(prediction)
        
        
        image = image[top:bot,left:rigth]
        image = cv2.resize(image,(512,512))
        image = clahe.apply( np.array(image*254,dtype=np.uint8))

        plt.imsave(f"{output_dir}{image_name}",image,cmap="gray")
        
