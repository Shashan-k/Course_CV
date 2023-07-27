
### "Image colorization using Autoencoders. Transfer learning using VGG-16."
### "Importing Necessary Libraries"
from keras.preprocessing.image import img_to_array, load_img
import os
import numpy as np
from skimage.transform import resize
from tensorflow.keras.utils import load_img
from ML_Pipeline.EncoderLayers import EncoderLayers
from ML_Pipeline.LoadData import LoadData
from ML_Pipeline.DecoderLayers import DecoderLayers
from ML_Pipeline.Inference import Inference

"""
Training Phase
"""
def trainingPhase():

    #Load Data
    datagen = LoadData()
    X, Y = datagen.getData()
    
    # Setup the Encoding Layer
    encoder_layer = EncoderLayers()
    encoded = encoder_layer.getfeatures(X)

    # Setup the Decoding Layer
    decoder_layer = DecoderLayers()
    decoder_layer.fit(encoded, Y)


"""
Inference Phase 
"""
def inferencePhase():
    # Setup the Encoding Layer
    encoder_layer = EncoderLayers()
    encode_model = encoder_layer.newmodel

    # Setup the Decoding Layer
    decoder_layer = DecoderLayers()
    model = decoder_layer.load_model()

    # taking Test images
    testpath = "C:/Users/88693/Desktop/Megha/ComputerVision/Image_Colorization_Project/Image_Colorization/modular_code/Input/dataset/test/"
    files = os.listdir(testpath)
    print(files)
    inf = Inference()

    # Inferencing and pasting the output
    for idx, file in enumerate(files):
        print(file)
        test = img_to_array(load_img(testpath+file))
        inf.processImg(idx, test, encode_model, model)
    

    # Evaluating our output - PSNR
    original_images_path = 'C:/Users/88693/Desktop/Megha/ComputerVision/Image_Colorization_Project/Image_Colorization/modular_code/Input/dataset/groundtruth/'
    colorized_images_path = 'C:/Users/88693/Desktop/Megha/ComputerVision/Image_Colorization_Project/Image_Colorization/modular_code/Output/generated_images/'

    psnr_values = []
    for idx, file in enumerate(files):
        original_image = load_img(original_images_path + file)
        colorized_image = load_img(colorized_images_path + str(idx) + '.jpg')
        
        original_image_array = img_to_array(original_image) / 255.0
        colorized_image_array = img_to_array(colorized_image) / 255.0
        
        original_image_array = resize(original_image_array, (224, 224), anti_aliasing=True)
        
        psnr = inf.calculate_psnr(original_image_array, colorized_image_array)
        psnr_values.append(psnr)

    average_psnr = np.mean(psnr_values)
    print("Average PSNR:", average_psnr)


    # Evaluating our output - SSIM
    ssim_values = []
    for idx, file in enumerate(files):
        original_image = load_img(original_images_path + file)
        colorized_image = load_img(colorized_images_path + str(idx) + '.jpg')
        
        original_image_array = img_to_array(original_image) / 255.0
        colorized_image_array = img_to_array(colorized_image) / 255.0
        
        original_image_array = resize(original_image_array, (224, 224), anti_aliasing=True)
        colorized_image_array = resize(colorized_image_array, (224, 224), anti_aliasing=True)
        
        ssim_score = inf.calculate_ssim(original_image_array, colorized_image_array)
        ssim_values.append(ssim_score)

    average_ssim = np.mean(ssim_values)
    print("Average SSIM:", average_ssim)

### Training the model ###
# trainingPhase()

### Perform inference ###
inferencePhase()




