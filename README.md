# Image Classification Project

A deep learning project that classifies images into two categories: **Happy** and **Sad**. This project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to perform binary image classification.

## 📁 Project Structure

```
Image_Classification/
├── data/                    # Dataset directory
│   ├── happy/              # Happy images
│   └── sad/                # Sad images
├── models/                  # Trained model files
│   └── imageclassifier.h5  # Saved model
├── logs/                   # Training logs for TensorBoard
├── Getting Started.ipynb   # Main Jupyter notebook
├── happy-people.jpg        # Sample happy image
├── sad.jpg                 # Sample sad image
└── README.md              # This file
```

## 🚀 Features

- **Binary Image Classification**: Classifies images as either "Happy" or "Sad"
- **CNN Architecture**: Uses a deep convolutional neural network
- **Data Preprocessing**: Automatic image validation and preprocessing
- **Model Training**: Complete training pipeline with validation
- **Performance Monitoring**: TensorBoard integration for training visualization
- **Model Evaluation**: Precision, Recall, and Accuracy metrics
- **Inference**: Ready-to-use prediction functionality

## 🛠️ Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook

## 📦 Installation

1. **Clone or download the project**
2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venvdl
   source .venvdl/bin/activate  # On Windows: .venvdl\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install tensorflow opencv-python numpy matplotlib jupyter
   ```

## 🎯 Usage

### Running the Notebook

1. **Start Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

2. **Open `Getting Started.ipynb`** and run the cells sequentially

### Project Workflow

The notebook follows this workflow:

1. **Setup**: Install dependencies and configure GPU memory
2. **Data Cleaning**: Remove corrupted or invalid images
3. **Data Loading**: Load images from the `data/` directory
4. **Data Preprocessing**: Scale pixel values (0-255 → 0-1)
5. **Data Splitting**: Split into train (70%), validation (20%), test (10%)
6. **Model Building**: Create CNN architecture
7. **Training**: Train the model with TensorBoard logging
8. **Evaluation**: Assess model performance
9. **Testing**: Make predictions on new images
10. **Model Saving**: Save the trained model


## 🧠 Model Architecture

The CNN model consists of:

- **Input Layer**: 256x256x3 RGB images
- **Convolutional Layers**: 3 Conv2D layers with ReLU activation
- **Pooling Layers**: MaxPooling2D after each Conv2D layer
- **Flatten Layer**: Converts feature maps to 1D vector
- **Dense Layers**: 256 neurons with ReLU, then 1 neuron with sigmoid
- **Total Parameters**: ~3.7 million

## 📊 Performance

The model achieves excellent performance:

- **Precision**: 1.0
- **Recall**: 1.0
- **Accuracy**: 1.0

## 📈 Training Visualization

Use TensorBoard to visualize training progress:

```bash
tensorboard --logdir=logs
```

Then open your browser to `http://localhost:6006`




