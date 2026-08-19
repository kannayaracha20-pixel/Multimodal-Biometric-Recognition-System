# Multimodal Biometric Recognition System

## Overview

The **Multimodal Biometric Recognition System** is a deep learning-based authentication framework that combines multiple biometric traits—**Face**, **Iris**, and **Finger Vein**—to provide accurate, secure, and reliable user identification. By integrating information from multiple biometric modalities, the system overcomes the limitations of unimodal biometric systems and improves overall recognition performance.

## Features

* Multi-biometric authentication using:

  * Face Recognition
  * Iris Recognition
  * Finger Vein Recognition
* Deep learning-based feature extraction
* Biometric fusion for enhanced accuracy
* Secure identity verification
* Reduced spoofing and impersonation risks
* High recognition performance and reliability

## Problem Statement

Traditional biometric systems that rely on a single biometric trait often suffer from issues such as noise, spoofing attacks, and environmental variations. This project addresses these challenges by integrating multiple biometric traits into a unified authentication system, improving security and recognition accuracy.

## Objectives

* Develop a robust multimodal biometric authentication framework.
* Extract discriminative features from face, iris, and finger vein images.
* Perform feature-level or score-level fusion.
* Improve recognition accuracy and system reliability.
* Enhance security against biometric spoofing attacks.

## System Architecture

1. Data Collection
2. Image Preprocessing
3. Feature Extraction
4. Feature Fusion
5. Classification and Matching
6. Authentication Decision

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## Dataset

The system utilizes biometric datasets containing:

* Face Images
* Iris Images
* Finger Vein Images

Datasets may be obtained from publicly available biometric repositories or institutional datasets used for research purposes.

## Methodology

### Image Preprocessing

* Image resizing
* Noise removal
* Normalization
* Contrast enhancement

### Feature Extraction

Deep learning models are used to extract meaningful features from each biometric modality.

### Fusion Strategy

Features obtained from face, iris, and finger vein modalities are combined to improve recognition performance.

### Classification

The fused feature vector is used for identity verification and classification.

## Project Structure

```text
Multimodal-Biometric-Recognition-System/
│
├── Dataset/
├── Models/
├── Preprocessing/
├── Training/
├── Testing/
├── Results/
├── Images/
├── requirements.txt
├── README.md
└── main.py
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/kannayaracha20-pixel/Multimodal---Biometric-Recognition-System.git
```

### Navigate to Project Directory

```bash
cd Multimodal---Biometric-Recognition-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the main application:

```bash
python main.py
```

Train the model:

```bash
python train.py
```

Test the model:

```bash
python test.py
```

## Results

The multimodal approach demonstrates improved authentication performance compared to single-modal biometric systems by leveraging complementary information from multiple biometric traits.

### Performance Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* False Acceptance Rate (FAR)
* False Rejection Rate (FRR)

## Applications

* Secure Access Control
* Banking and Financial Authentication
* Healthcare Systems
* Government Identification Systems
* Border Security
* Smart Surveillance

## Future Enhancements

* Real-time biometric authentication
* Mobile deployment
* Cloud-based biometric verification
* Advanced deep learning architectures
* Liveness detection mechanisms

## Author

**Racha Murali Krishna**

B.Tech – Computer Science Engineering (AI & DS)
Sri Indu Institute of Engineering and Technology

## License

This project is developed for academic and research purposes.
