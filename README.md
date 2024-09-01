# Traffic Accident Analysis

## Overview

This repository contains code for analyzing traffic accident data. The analysis includes data cleaning, feature encoding, univariate and bivariate analysis, predictive modeling, and visualization of results. The goal is to understand the patterns and factors contributing to traffic accidents and to build predictive models based on this data.

## Features

- **Data Cleaning**: Handles missing values and prepares the dataset for analysis.
- **Feature Encoding**: Converts categorical features to numerical formats for modeling.
- **Univariate Analysis**: Plots the distribution of categorical features.
- **Bivariate Analysis**: Examines relationships between accident severity and various features.
- **Predictive Modeling**: Trains and evaluates Random Forest, Gradient Boosting, and Support Vector Machine models.
- **Feature Importance**: Analyzes the importance of features using Random Forest.
- **Segmentation Analysis**: Examines accident severity across different segments like age, gender, and cause of accident.

## Dataset

The dataset should be in CSV format and should include the following features:
- `Age_band_of_driver`
- `Sex_of_driver`
- `Educational_level`
- `Vehicle_driver_relation`
- `Driving_experience`
- `Lanes_or_Medians`
- `Types_of_Junction`
- `Road_surface_type`
- `Light_conditions`
- `Weather_conditions`
- `Type_of_collision`
- `Vehicle_movement`
- `Pedestrian_movement`
- `Cause_of_accident`
- `Accident_severity`

## Requirements

Ensure you have the following Python packages installed:
- `pandas`
- `seaborn`
- `matplotlib`
- `numpy`
- `plotly`
- `scikit-learn`

You can install the necessary packages using `pip`:
```bash
pip install pandas seaborn matplotlib numpy plotly scikit-learn

Usage
Load the dataset: Ensure your dataset is named cleaned.csv and located in the same directory as the script.

Run the script: Execute the Python script to perform the analysis.
Copy code
python RTA_analysis.py
View results: The script will generate various plots and output metrics to the console, including:

Distribution of categorical features
Bivariate analysis of accident severity by different features
Model evaluation metrics for Random Forest, Gradient Boosting, and Support Vector Machine models
Feature importance from the Random Forest model
Segmentation analysis based on different features
Code Explanation
Data Cleaning: Rows with missing values are dropped.
Feature Encoding:
Ordinal Encoding: For features with a natural ordering like Driving_experience.
Label Encoding: For other categorical features.
Visualization:
Univariate Analysis: Histograms of feature distributions.
Bivariate Analysis: Histograms of accident severity by different features using plotly.express.
Predictive Modeling:
Models are trained and evaluated with accuracy, confusion matrix, and ROC-AUC score.
Feature Importance: Uses Random Forest to show which features are most important for predicting accident severity.
Segmentation Analysis: Looks at how accident severity varies by categories like age band, sex, and cause of accident.
Contributing
Feel free to open issues or submit pull requests if you have suggestions or improvements. Please make sure to follow the coding style and include relevant tests with your contributions.
