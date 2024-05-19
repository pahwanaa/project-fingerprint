import os
import cv2
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

def compute_lbp(image):
    radius = 1
    num_points = 8 * radius
    lbp = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            center = image[i, j]
            code = 0
            code |= (image[i-1, j-1] > center) << 7
            code |= (image[i-1, j] > center) << 6
            code |= (image[i-1, j+1] > center) << 5
            code |= (image[i, j+1] > center) << 4
            code |= (image[i+1, j+1] > center) << 3
            code |= (image[i+1, j] > center) << 2
            code |= (image[i+1, j-1] > center) << 1
            code |= (image[i, j-1] > center) << 0
            lbp[i, j] = code
    return lbp[1:-1, 1:-1]

def preprocess_image_lbp(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    lbp_image = compute_lbp(img)
    
    resized_img = cv2.resize(lbp_image, (180, 180)) 
    
    flattened_img = resized_img.flatten()  
    return flattened_img

with open('Dataset\MBTI_Intents.json', encoding='utf-8') as f:
    mbti_data = json.load(f)

X_personality = []
y_personality = []

for entry in mbti_data:
    fingerprint_type = entry["type"]
    
    # Extract the personalities associated with the fingerprint type
    personalities = entry["personality"]
    
    # Iterate through each personality and add it to the labels
    for personality in personalities:
        X_personality.append(fingerprint_type)
        y_personality.append(personality)

# Convert lists to numpy arrays
X_personality = np.array(X_personality)
y_personality = np.array(y_personality)

dataset_path = "Dataset/Fingerprint_IMG"

X_images_lbp = []
y_images_lbp = []

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    for image_file in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
        image_path = os.path.join(class_path, image_file)
        features = preprocess_image_lbp(image_path)
        X_images_lbp.append(features)
        y_images_lbp.append(class_name)

X_images_lbp = np.array(X_images_lbp)
y_images_lbp = np.array(y_images_lbp)

# Apply SMOTE for class imbalance
smote = SMOTE(random_state=32)
X_resampled_lbp, y_resampled_lbp = smote.fit_resample(X_images_lbp, y_images_lbp)

X_train_lbp, X_test_lbp, y_train_lbp, y_test_lbp = train_test_split(X_resampled_lbp, y_resampled_lbp, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
rf_classifier_lbp = RandomForestClassifier(random_state=42)

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [None, 10, 50, 100],
    'max_features': ['sqrt', 'log2']
}

# Perform Grid Search Cross Validation
grid_search_lbp = GridSearchCV(estimator=rf_classifier_lbp, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_lbp.fit(X_train_lbp, y_train_lbp)

# Get the best parameters from the grid search
best_params_lbp = grid_search_lbp.best_params_

# Initialize Random Forest Classifier with best parameters
best_rf_classifier_lbp = RandomForestClassifier(**best_params_lbp, random_state=42)

# Train the model with the best parameters
best_rf_classifier_lbp.fit(X_train_lbp, y_train_lbp)

y_pred_train_lbp = best_rf_classifier_lbp.predict(X_train_lbp)
y_pred_test_lbp = best_rf_classifier_lbp.predict(X_test_lbp)

# Calculate accuracy
train_accuracy_lbp = accuracy_score(y_train_lbp, y_pred_train_lbp)
test_accuracy_lbp = accuracy_score(y_test_lbp, y_pred_test_lbp)

print("Train Accuracy:", train_accuracy_lbp)
print("Test Accuracy:", test_accuracy_lbp)

print("\nClassification Report:")
print(classification_report(y_test_lbp, y_pred_test_lbp))

# Save the LBP model
joblib.dump(best_rf_classifier_lbp, 'Model/Opersonalities.joblib')
