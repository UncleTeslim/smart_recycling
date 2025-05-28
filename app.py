import gradio as gr
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers, models
from keras.models import load_model


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model = load_model("models/finetuned_waste_classifier_mobilenetv2_v1.keras", compile=False)

class_names = ['batteries', 'clothes', 'e-waste', 'glass', 'light bulbs',
               'metal', 'organic', 'paper', 'plastic']
IMG_SIZE = 224

def classify_image(img):
    if img is None:
        return None
    
    if isinstance(img, np.ndarray):
        image_np = img
    else:
        image_np = np.array(img)
    
    img = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))

    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))

    label = class_names[class_idx]
    return {label: confidence}




# Gradio Interface
interface = gr.Interface(
    fn=classify_image,
    inputs=[
        gr.Image(label="Upload Waste Image or Use Webcam", image_mode='RGB')
    ],
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    live=True,
    title="Smart Waste Classifier",
    description="Upload an image or use your webcam to classify recyclable materials (e.g., plastic, paper, glass). Model: fine-tuned MobileNetV2.",
)


interface.launch(share=True)