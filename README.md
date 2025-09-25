# Flower Classifier

A deep learning image classifier built with TensorFlow that predicts flower types from images. The model returns the top K most likely classes along with their probabilities.

## Features

- Predict flower type from a single image using a trained Keras model.
- Return top K predictions.
- Map numeric class labels to flower names via a JSON file.
- Command-line interface for easy usage.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/IbrahimHashhash/flower-classifier
cd flower-classifier
```
## Usage 
- Basic usage:
python predict.py /path/to/image saved_model.h5
- Return top 3 predictions:
python predict.py /path/to/image saved_model.h5 --top_k 3
- Use a JSON file for flower names:
python predict.py /path/to/image saved_model.h5 --category_names label_map.json
