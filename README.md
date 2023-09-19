# Image-Forgery-Detection-through-Error-Level-Analysis-and-Convolutional-Neural-Networks

# Overview
Digital image forgery detection is a critical task in the field of image forensics, aiming to identify manipulated regions within images and preserve the integrity of visual content. This repository presents a comprehensive framework for detecting image forgeries by incorporating Error Level Analysis (ELA) in conjunction with prominent Convolutional Neural Network (CNN) architectures, including VGG-16, VGG-19, ResNet-50, Xception, and EfficientNetV3.

# Table of Contents
* Introduction
* Key Features
* Dataset
* Proposed Methodology
  * ELA - Error Level Analysis
  * How ELA Works
* Advantages of ELA compared to Patch-Level Feature Extraction
* Results

# Introduction
This repository contains the implementation and research findings of a comprehensive framework for detecting digital image forgeries using Error Level Analysis (ELA) in conjunction with prominent Convolutional Neural Network (CNN) architectures. Image forgery detection is a critical task in image forensics, and our approach combines the strengths of ELA and CNNs to achieve accurate and efficient forgery detection.

![alt text](https://github.com/Khizar-Baig/Image-Forgery-Detection-through-Error-Level-Analysis-and-Convolutional-Neural-Networks/assets/59732957/00862f75-3ded-4a45-b4eb-8c367f890b25?raw=true)

# Key Features
* Integration of Error Level Analysis (ELA) with prominent CNN architectures.
* Comparison between ELA and patch-level feature extraction techniques.
* Extensive experiments on the CASIA1 dataset to evaluate framework performance.
* Metrics include loss, accuracy, recall, precision, F1-score, and computational time.
* Demonstrated superiority of ELA over patch-level techniques.
* Efficient and feasible approach suitable for real-world applications.

# Dataset
The framework is evaluated on the CASIA1 dataset. 

# Proposed Methodology
## ELA - Error Level Analysis
Error Level Analysis (ELA) is a forensic technique used to detect digital image manipulations or alterations. It analyzes inconsistencies in error levels introduced during the compression and recompression of an image. ELA is integrated with well-known CNN models, including VGG-16, VGG-19, ResNet-50, Xception, and EfficientNetV3, to enhance forgery detection precision and effectiveness.

## How ELA Works
1. Conversion to a lossy format: The original image is converted to a lossy format like JPEG, introducing compression artifacts.
2. Resaving and recompression: The JPEG image is resaved with the same compression level, amplifying differences.
3. Error level calculation: ELA calculates error level differences between the original and recompressed images.
4. Visualization of error levels: The calculated error levels are represented using a color map, highlighting potential manipulation areas.
   
# Advantages of ELA compared to Patch-Level Feature Extraction
* Non-specific Detection: ELA detects various types of manipulations, such as copy-pasting, image splicing, and retouching.
* Global Analysis: Analyzes the entire image, ensuring comprehensive coverage of manipulations.
* Simplicity and Computational Efficiency: Requires less computational resources and is faster to execute compared to patch-level techniques.
* Initial Screening Tool: Quickly identifies suspicious regions, guiding further detailed analysis.
* Accessibility: Applicable to images of different formats and sizes without extensive preprocessing.

# Result Analysis
Our research findings reveal essential insights into the effectiveness of our proposed framework for digital image forgery detection:

* Robust Detection: Our framework, combining ELA with CNN architectures, demonstrates robust detection capabilities, effectively identifying manipulated regions in digital images.
* Model Comparison: Through a detailed analysis of performance metrics, we compare different CNN architectures and highlight their varying strengths in forgery detection.
* Efficiency: Our framework proves to be efficient and scalable, making it suitable for real-world applications that demand timely image analysis.
* Comprehensive Analysis: The use of a wide range of performance metrics and visual aids allows for a comprehensive evaluation, enabling researchers and practitioners to make informed decisions.


