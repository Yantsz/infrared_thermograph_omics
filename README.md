# Introduction

This code repository is developed based on the methods described in "Feasibility of Infrared Thermography Omics: A Study on Facial and Palmar Thermography in Metabolic Syndrome." It is intended for creating predictive models for diagnosing Metabolic Syndrome using facial and palmar infrared thermography. The aim is to explore the potential of using infrared imaging for diagnosing metabolic diseases. The regions of interest (ROI) chosen for this study are the face and palms due to their ease of access, ability to protect patient privacy effectively, and sensitive response to metabolic changes.

Data from 196 individuals (98 Metabolic Syndrome patients and 98 healthy subjects) were collected at Dongzhimen Hospital and divided into a training set of 137 (68 patients, 69 healthy) and a validation set of 59 (30 patients, 29 healthy) using a stratified random sampling method at a 7:3 ratio. A total of 1656 radiomic features were extracted from both ROIs using pyradiomics, and after Pearson correlation analysis, two-sample t-tests, and Lasso regression for feature selection, 7 key radiomic features were chosen to build the logistic regression model.

In the training cohort, the facial and palmar radiomics model showed excellent diagnostic performance with an AUC of 0.90 (95% CI: 0.84-0.95). In the independent validation set, the model maintained a stable performance with an AUC of 0.89 (95% CI: 0.81-0.98). The calibration curve in the validation cohort demonstrated good consistency, indicating no significant deviation from the ideal probability (Spiegelhalter’s test, z=-0.745, p=0.456). The decision curve analysis indicated that the radiomics model offers a significant advantage in distinguishing between normal individuals and those with Metabolic Syndrome across a broader range of prediction probabilities (0-0.83) compared to assuming all participants are either healthy or have the syndrome.

## Repository Objectives
To provide detailed implementation of the facial and palmar infrared radiomics model development for reviewing our research paper.
To enable researchers to validate or further improve the Metabolic Syndrome radiomics model based on facial and palmar infrared thermography on larger datasets.
This repository is intended for use by researchers evaluating the manuscript and the development and validation of the model. Note that this code is not intended for generating predictions for clinical decisions or any other clinical use. Users assume all risks associated with using this model.


Our code runs on Python 3.11, and third-party libraries can be installed using the following command with pip:
pip install -r requirements.txt
Please note: Pyradiomics requires C++ support. You need to install Microsoft C++ build tools first.
