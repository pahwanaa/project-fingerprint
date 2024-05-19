import cv2
from collections import Counter
import numpy as np
import joblib
import json


model = joblib.load('Model/Opersonalities.joblib')

# Function to compute Local Binary Pattern (LBP) features
def compute_lbp(image):
    radius = 1
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

def preprocess_input_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Compute LBP features
    lbp_image = compute_lbp(img)
    
    resized_img = cv2.resize(lbp_image, (180, 180)) 
    
    flattened_img = resized_img.flatten()  
    return flattened_img

# Function to predict fingerprint pattern from input image
def predict_fingerprint_pattern(image_path, model):
    input_features = preprocess_input_image(image_path)
    
    prediction = model.predict([input_features])
    
    return prediction[0]

# Function to map fingerprint pattern to personality type
def map_pattern_to_personality(pattern):
    # Load personality dataset
    with open('Dataset/MBTI_Intents.json', encoding='utf-8') as f:
        personality_data = json.load(f)
    
    # Search for pattern in dataset
    for entry in personality_data:
        if entry["type"] == pattern:
            # Return personality associated with the pattern
            return entry["personality"]
    
    # If pattern not found, return None
    return None

# Function to predict fingerprint pattern and personality from input image
def predict_fingerprint_pattern_and_personality(image_path, model):
    # Predict fingerprint pattern
    predicted_pattern = predict_fingerprint_pattern(image_path, model)
    
    # Map pattern to personality
    predicted_personalities = map_pattern_to_personality(predicted_pattern)
    
    # Count the occurrences of each personality
    personality_counts = Counter(predicted_personalities)
    
    # Get the most common personality
    most_common_personality = personality_counts.most_common(1)[0][0]
    
    return predicted_pattern, most_common_personality

input_image_path = 'Source\Test_IMG\Fingerprint_Test.jpg'

# Predict fingerprint pattern and personality from input image
predicted_pattern, predicted_personality = predict_fingerprint_pattern_and_personality(input_image_path, model)

print("Predicted Fingerprint Pattern:", predicted_pattern)
print("Predicted Personality:", predicted_personality)
