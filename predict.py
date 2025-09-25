import argparse
from PIL import Image
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import json
from utility import process_image

parser = argparse.ArgumentParser(description='Predict flower class from an image using a trained model.')
parser.add_argument('image_path', type=str, help='Path of the image to predict')
parser.add_argument('model', type=str, help='Path to the trained Keras model')
parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
parser.add_argument('--category_names', type=str, default='label_map.json', help='Path to JSON file mapping labels to flower names')
args = parser.parse_args()

def predict(image_path, model, top_k=5):
    im = Image.open(image_path)
    image_array = np.asarray(im)
    processed_image = process_image(image_array)
    image_batch = np.expand_dims(processed_image, axis=0)
    probabilities = model.predict(image_batch)[0]
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_probs = probabilities[top_indices]
    return top_indices, top_probs

if __name__ == '__main__':
    custom_objects = {'KerasLayer': hub.KerasLayer}
    model = tf.keras.models.load_model(args.model, custom_objects=custom_objects)

    with open(args.category_names, 'r') as f:
        class_names = json.load(f)

    top_preds, top_probs = predict(args.image_path, model, args.top_k)

    print(f'{"Prediction": <30}{"":<10}Probability')
    for pred, prob in zip(top_preds, top_probs):
        print(f'{class_names[str(pred)]:<30}{"":<10}{prob:.4f}')
