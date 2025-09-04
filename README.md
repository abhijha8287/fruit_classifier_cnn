# 🍎 Fruit Freshness Classifier CNN

A deep learning project that classifies fruits as fresh or spoiled using Convolutional Neural Networks. This project implements both a custom CNN and a transfer learning approach using ResNet50, deployed as an interactive web application using Streamlit.

## 🌟 Features

- **Dual Model Implementation**: Custom CNN and ResNet50 transfer learning
- **Binary Classification**: Determines if fruit is fresh or spoiled
- **8 Fruit Types Supported**: Banana, Lemon, Lulo, Mango, Orange, Strawberry, Tamarillo, Tomato
- **Interactive Web Interface**: Easy-to-use Streamlit application
- **Pre-trained Model**: Ready-to-use trained ResNet50 model included

## 📁 Project Structure

```
fruit_classifier_cnn/
├── app.py                          # Streamlit web application
├── task.ipynb                      # Custom CNN training notebook
├── transfer.ipynb                  # ResNet50 transfer learning notebook
├── requirements.txt                # Python dependencies
├── fruit_classifier_resnet50.keras # Pre-trained ResNet50 model
└── fruits/                         # Dataset directory
    ├── F_Banana/                   # Fresh banana images
    ├── F_Lemon/                    # Fresh lemon images
    ├── F_Lulo/                     # Fresh lulo images
    ├── F_Mango/                    # Fresh mango images
    ├── F_Orange/                   # Fresh orange images
    ├── F_Strawberry/               # Fresh strawberry images
    ├── F_Tamarillo/                # Fresh tamarillo images
    ├── F_Tomato/                   # Fresh tomato images
    ├── S_Banana/                   # Spoiled banana images
    ├── S_Lemon/                    # Spoiled lemon images
    ├── S_Lulo/                     # Spoiled lulo images
    ├── S_Mango/                    # Spoiled mango images
    ├── S_Orange/                   # Spoiled orange images
    ├── S_Strawberry/               # Spoiled strawberry images
    ├── S_Tamarillo/                # Spoiled tamarillo images
    └── S_Tomato/                   # Spoiled tomato images
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/abhijha8287/fruit_classifier_cnn.git
   cd fruit_classifier_cnn
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

4. **Access the web interface**
   - Open your browser and navigate to `http://localhost:8501`
   - Upload a fruit image and get instant freshness classification!

## 📊 Model Performance

### Custom CNN Model (task.ipynb)
- **Architecture**: 2 Convolutional layers with MaxPooling, Dense layers
- **Input Size**: 256x256 pixels
- **Training Results**: 
  - Final Training Accuracy: ~95.7%
  - Final Validation Accuracy: ~82.2%
  - Training completed in 10 epochs

### Transfer Learning Model (transfer.ipynb)
- **Base Model**: ResNet50 pre-trained on ImageNet
- **Input Size**: 224x224 pixels (ResNet50 standard)
- **Training Results**:
  - Final Training Accuracy: ~99.5%
  - Final Validation Accuracy: ~99.8%
  - Training completed in just 3 epochs
  - **Superior performance with faster training**

## 🛠️ Technical Implementation

### Custom CNN Architecture
- Rescaling layer (normalization)
- Conv2D (32 filters, 3x3 kernel) + ReLU
- MaxPooling2D
- Conv2D (64 filters, 3x3 kernel) + ReLU
- MaxPooling2D
- Flatten
- Dense (128 neurons) + ReLU
- Dense (16 classes) + Softmax

### Transfer Learning Architecture
- ResNet50 base (frozen weights)
- GlobalAveragePooling2D
- Dense (128 neurons) + ReLU
- Dropout (0.3)
- Dense (16 classes) + Softmax

### Dataset Information
- **Total Images**: 16,000
- **Classes**: 16 (8 fruits × 2 conditions)
- **Training Split**: 80% (12,800 images)
- **Validation Split**: 20% (3,200 images)
- **Batch Size**: 32

## 🖼️ Web Application Features

The Streamlit app (`app.py`) provides:
- **User-friendly Interface**: Clean, colorful design
- **Image Upload**: Support for JPG, JPEG, PNG formats
- **Real-time Prediction**: Instant classification results
- **Visual Feedback**: 
  - Green success message for fresh fruits
  - Red error message for spoiled fruits
- **Supported Fruits Display**: Clear list of compatible fruit types
- **Image Preprocessing**: Automatic resizing and normalization

## 📋 Dependencies

- `streamlit`: Web application framework
- `tensorflow`: Deep learning framework
- `keras`: High-level neural network API
- `pillow`: Image processing library
- `numpy`: Numerical computing library

## 🎯 Usage Examples

1. **Web Application**: Upload any image of the supported fruits
2. **Jupyter Notebooks**: 
   - `task.ipynb`: Train your own custom CNN
   - `transfer.ipynb`: Implement transfer learning with ResNet50

## 🔬 Model Training

### To train the custom CNN:
```python
# Open and run task.ipynb
# The notebook includes:
# - Data loading and preprocessing
# - Model architecture definition
# - Training loop with 10 epochs
# - Performance evaluation
```

### To train with transfer learning:
```python
# Open and run transfer.ipynb
# The notebook includes:
# - ResNet50 base model loading
# - Custom classifier head
# - Efficient training (3 epochs)
# - Model saving
```

## 🎨 Supported Fruit Types

The model can classify the freshness of these fruits:
- 🍌 **Banana**
- 🍋 **Lemon**
- 🫒 **Lulo**
- 🥭 **Mango**
- 🍊 **Orange**
- 🍓 **Strawberry**
- 🍅 **Tamarillo**
- 🍅 **Tomato**

## 📈 Performance Comparison

| Model | Training Accuracy | Validation Accuracy | Training Time | Epochs |
|-------|------------------|-------------------|---------------|--------|
| Custom CNN | 95.7% | 82.2% | ~50 minutes | 10 |
| ResNet50 Transfer | 99.5% | 99.8% | ~35 minutes | 3 |

**Winner**: ResNet50 transfer learning provides superior accuracy with faster training!

## 🚀 Future Improvements

- [ ] Add more fruit varieties
- [ ] Implement data augmentation for better generalization
- [ ] Deploy to cloud platforms (Heroku, AWS, etc.)
- [ ] Add confidence score display
- [ ] Implement batch processing
- [ ] Add mobile app version

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Abhishek Jha** - [@abhijha8287](https://github.com/abhijha8287)

## 🙏 Acknowledgments

- TensorFlow and Keras teams for the excellent deep learning frameworks
- Streamlit team for the intuitive web app framework
- ResNet50 architecture authors for the pre-trained model
- The open-source community for inspiration and resources

---

**Made with ❤️ using TensorFlow, Keras, and Streamlit**
