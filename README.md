# Magnetic Tile Defect Detection

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/Oumllack/Drilling-Computer-Vision-Project.git)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Project Overview
This project implements a deep learning-based solution for detecting and classifying defects in magnetic tiles. The system uses a Convolutional Neural Network (CNN) based on ResNet50 architecture to identify five different types of defects: Free, Fray, Crack, Break, and Blowhole.

## Problem Statement
Magnetic tiles are crucial components in various industrial applications. Detecting defects in these tiles is essential for maintaining product quality and preventing equipment failures. Manual inspection is time-consuming and prone to human error. This project aims to automate the defect detection process using computer vision and deep learning techniques.

## Dataset
The project uses the Magnetic Tile Defect dataset, which contains images of magnetic tiles with various types of defects. The dataset is organized into five categories:
- MT_Free: Defect-free tiles
- MT_Fray: Tiles with frayed edges
- MT_Crack: Tiles with cracks
- MT_Break: Tiles with breaks
- MT_Blowhole: Tiles with blowholes

## Technical Implementation

### Architecture
- Base Model: ResNet50 (pretrained on ImageNet)
- Modifications:
  - Added dropout (0.5) for regularization
  - Custom fully connected layer for 5-class classification
  - Input size: 224x224 pixels

### Data Preparation
- Split ratio: 70% training, 15% validation, 15% testing
- Data augmentation:
  - Random horizontal flips
  - Random rotations (±10 degrees)
  - Color jittering
  - Random affine transformations

### Training Process
- Optimizer: Adam
- Learning rate: 0.001
- Weight decay: 1e-4
- Batch size: 32
- Number of epochs: 20
- Early stopping based on validation accuracy

## Results
The model achieved impressive results on the test set:

### Overall Performance
- Global Accuracy: 90.00%

### Per-Class Performance
- MT_Free: 88.18%
- MT_Fray: 86.67%
- MT_Crack: 95.00%
- MT_Break: 88.57%
- MT_Blowhole: 94.44%

The model shows particularly strong performance in detecting cracks and blowholes, which are critical defects in magnetic tiles.

## Project Structure
```
.
├── src/
│   ├── prepare_data.py      # Data preparation and splitting
│   ├── train.py            # Training script
│   ├── test_model.py       # Model evaluation
│   └── training/
│       ├── model.py        # Model architecture
│       ├── dataset.py      # Dataset class
│       └── trainer.py      # Training utilities
├── data/
│   ├── train/             # Training data
│   ├── val/              # Validation data
│   └── test/             # Test data
└── models/               # Saved model checkpoints
```

## Setup and Usage

### Prerequisites
- Python 3.8+
- PyTorch
- torchvision
- tqdm
- PIL

### Installation
```bash
# Clone the repository
git clone https://github.com/Oumllack/Drilling-Computer-Vision-Project.git
cd Drilling-Computer-Vision-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation
```bash
python src/prepare_data.py
```

### Training
```bash
python src/train.py --data-dir data --model-dir models --num-epochs 20 --batch-size 32 --learning-rate 0.001 --weight-decay 1e-4 --save-interval 5
```

### Testing
```bash
python src/test_model.py
```

## Deployment
The model can be deployed in various ways:

### Local Deployment
1. Save the trained model:
```bash
python src/save_model.py --model-path models/best_model.pth --output-path models/deployed_model.pth
```

2. Use the model for inference:
```bash
python src/inference.py --model-path models/deployed_model.pth --image-path path/to/image.jpg
```

### Web Deployment
A Flask API is provided for web deployment:
```bash
python src/app.py
```
The API will be available at `http://localhost:5000`

### Docker Deployment
Build and run the Docker container:
```bash
docker build -t defect-detection .
docker run -p 5000:5000 defect-detection
```

## Future Improvements
1. Implement data balancing techniques to improve performance on underrepresented classes
2. Experiment with different architectures (EfficientNet, Vision Transformer)
3. Add real-time inference capabilities
4. Implement a web interface for easy model deployment
5. Add support for new defect types

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- The Magnetic Tile Defect dataset providers
- PyTorch team for the excellent deep learning framework
- The open-source community for various tools and libraries used in this project 
