# Fruit Freshness Classifier

Classifies a fruit photo as fresh or spoiled across 8 fruit types, with a Streamlit app for predictions.

## Why this exists
Built during the AtliQ Technologies virtual data science internship (Dec 2025 - Jan 2026), to replace manual freshness inspection.

## What it does
- Trains a custom CNN (`task.ipynb`) and a ResNet50 transfer-learning model (`transfer.ipynb`)
- Predicts 16 classes: fresh or spoiled banana, lemon, lulo, mango, orange, strawberry, tamarillo and tomato
- ResNet50 reached 97.8% validation accuracy and is saved as `fruit_classifier_resnet50.keras`
- `app.py` lets you upload a photo and shows the predicted class and confidence

## Project structure
```
fruit_classifier_cnn/
├── fruits/                           Images, one folder per class
├── task.ipynb                        Custom CNN training
├── transfer.ipynb                    ResNet50 transfer learning
├── fruit_classifier_resnet50.keras   Trained model
├── app.py                            Streamlit prediction app
└── requirements.txt                  Python dependencies
```
Folder names use `F_` for fresh and `S_` for spoiled.

## Getting started
Requirements: Python 3.10+.
```bash
git clone https://github.com/abhijha8287/fruit_classifier_cnn.git
cd fruit_classifier_cnn
pip install -r requirements.txt
streamlit run app.py
```
To retrain, open `transfer.ipynb` in Jupyter and run all cells.

## Tech stack
Python, TensorFlow/Keras (ResNet50), NumPy, Pillow, Streamlit

## Notes
The repository is large (about 16,000 images). Use `git clone --depth 1` for a faster download.
