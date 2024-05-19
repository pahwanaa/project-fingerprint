**MBTI Personality Prediction with Machine Learning**

This repository contains a machine learning model for predicting MBTI (Myers-Briggs Type Indicator) personality types based on fingerprint images.

## Overview

The Myers-Briggs Type Indicator (MBTI) is a widely used tool for understanding personality preferences. This project utilizes machine learning techniques to predict MBTI personality types using fingerprint images as input data.

## Dataset

The dataset consists of fingerprint images categorized into different types, along with the associated MBTI personality types. The dataset is preprocessed to extract Local Binary Pattern (LBP) features from the images, which are then used to train the machine learning model.

## Model Training

The machine learning model is trained using a Random Forest Classifier. Grid Search Cross Validation is employed to optimize hyperparameters such as the number of estimators, maximum depth, and maximum features. The model is trained on the preprocessed fingerprint image data to predict the corresponding MBTI personality types.

## Usage

1. **Install Dependencies**: Ensure all dependencies are installed by running:
    ```
    pip install -r requirements.txt
    ```

2. **Preprocess Data**: The dataset is preprocessed to extract LBP features from fingerprint images.

3. **Train Model**: Execute the `TrainModelLBP.py` script to train the machine learning model:
    ```
    python TrainModelLBP.py
    ```

4. **Predict MBTI Personality**: Use the trained model to predict MBTI personality types from fingerprint images by running the `Test.py` script:
    ```
    python Test.py --input_image_path <path_to_image>
    ```

    Replace `<path_to_image>` with the path to the fingerprint image for which you want to predict the MBTI personality type.

## Results

The trained model achieves high accuracy in predicting MBTI personality types from fingerprint images. Evaluation metrics such as accuracy and classification report are provided to assess the model's performance.

## Conclusion

This project demonstrates the application of machine learning in predicting MBTI personality types based on fingerprint images. The trained model can be utilized in various applications such as personality assessment, forensic analysis, and biometric identification.

