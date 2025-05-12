import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("perfume_dataset.csv")
features = ['scent_type', 'gender', 'longevity', 'sillage'] + \
           ['Citrus', 'Bergamot', 'Lemon', 'Rose', 'Jasmine', 'Lavender', 'Sandalwood', 'Vanilla', 'Musk']
X = df[features]
y = df['perfume_name']

# Encode categorical features
le_scent = LabelEncoder()
le_gender = LabelEncoder()
X.loc[:, 'scent_type'] = le_scent.fit_transform(X['scent_type'])
X.loc[:, 'gender'] = le_gender.fit_transform(X['gender'])

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Helper function to prepare user input
def prepare_user_input(scent_type, gender, longevity, sillage):
    input_data = pd.DataFrame({
        'scent_type': [scent_type],
        'gender': [gender],
        'longevity': [longevity],
        'sillage': [sillage],
        'Citrus': [1 if scent_type in ['Citrus', 'Fresh', 'Aquatic'] else 0],
        'Bergamot': [1 if scent_type in ['Citrus', 'Fresh'] else 0],
        'Lemon': [1 if scent_type in ['Citrus', 'Fresh', 'Aquatic'] else 0],
        'Rose': [1 if scent_type in ['Floral', 'Fruity'] else 0],
        'Jasmine': [1 if scent_type in ['Floral', 'Oriental'] else 0],
        'Lavender': [1 if scent_type in ['Floral', 'Fresh'] else 0],
        'Sandalwood': [1 if scent_type in ['Woody', 'Oriental'] else 0],
        'Vanilla': [1 if scent_type in ['Gourmand', 'Oriental'] else 0],
        'Musk': [1 if scent_type in ['Oriental', 'Gourmand'] else 0]
    })
    input_data['scent_type'] = le_scent.transform(input_data['scent_type'])
    input_data['gender'] = le_gender.transform(input_data['gender'])
    return input_data

# Load test data
test_data = pd.read_csv("test_data.csv")

# Run tests
print("Running tests...")
for idx, row in test_data.iterrows():
    scent_type = row['scent_type']
    gender = row['gender']
    longevity = int(row['longevity'])
    sillage = int(row['sillage'])
    expected_perfumes = row['expected_perfumes'].split(',')

    # Prepare input
    input_data = prepare_user_input(scent_type, gender, longevity, sillage)

    # Predict
    probabilities = model.predict_proba(input_data)[0]
    top_indices = np.argsort(probabilities)[::-1]
    top_perfumes = [model.classes_[i] for i in top_indices]
    top_scores = probabilities[top_indices]

    # Filter by gender and select top 6
    recommendations = []
    count = 0
    for perfume, score in zip(top_perfumes, top_scores):
        if count >= 6:
            break
        perfume_data = df[df['perfume_name'] == perfume].iloc[0]
        if (gender == 'Unisex' and perfume_data['gender'] == 'Unisex') or \
           (gender == 'Female' and perfume_data['gender'] in ['Female', 'Unisex']) or \
           (gender == 'Male' and perfume_data['gender'] in ['Male', 'Unisex']):
            recommendations.append(perfume)
            count += 1

    # Compare with expected
    match = all(perfume.strip() in [p.strip() for p in expected_perfumes] for perfume in recommendations) and len(recommendations) == 6
    print(f"Test {idx+1}: {scent_type}, {gender}, Longevity={longevity}, Sillage={sillage}")
    print(f"Recommended: {recommendations}")
    print(f"Expected: {expected_perfumes}")
    print(f"Pass: {match}\n")