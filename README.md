#**Overview**#
This MATLAB project detects and segments brain tumors from MRI images using image processing and a Support Vector Machine (SVM) classifier. It also calculates the tumor area automatically and provides a simple GUI for users to upload and analyze MRI scans easily.

**Methodology**
1)Input MRI Image
2)Preprocessing: Convert to grayscale and resize
3)Segmentation: Apply Otsu’s thresholding and morphological operations
4)Feature Extraction: Calculate mean and standard deviation
5)Classification: Use SVM to detect tumor presence
6)Visualization: Display original image, segmented tumor, and outline

**Metrics:** 
Show accuracy, precision, recall, and F1-score in bar graph

**Features**
1)Automatic tumor detection and area calculation
2)Three-panel visualization (Brain, Tumor Mask, Detected Tumor)
3)GUI for easy image upload and analysis
4)Performance metrics with graphical representation

**Requirements**
1)MATLAB R2021a or later
2)Image Processing Toolbox
3)Statistics and Machine Learning Toolbox

**How to Run**
1)Place MRI images in yes (tumor) and no (non-tumor) folders
2)Open mri_tumor_detection.m in MATLAB
3)Run the script and upload an image using the GUI

View detection results and model metrics
