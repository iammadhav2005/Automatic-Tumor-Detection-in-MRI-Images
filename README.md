# Automatic-Tumor-Detection-in-MRI-Images
The model is medical application for detection of brain tumor from MRI images. In addition, segmentation of brain tumor and area calculation is done automatically for examination of MRI's image. With this application, even an ordinary person can precisely study the MRI images.

This repository includes source code for tumor segmentation and its area calculation - mri_tumor_detection.m

**Methodology:**

1)Taking an MRI Image as input
2)Thresholding the image using Basic Global algorithm & Otsu’s Method
3)Processing image properties in detail using regionprops and Morphological Operation
4)Confirmation of tumor based on density and area
5)If no Tumor end processing
6)Else segmentation of tumor in the image by particular border and calculate area
