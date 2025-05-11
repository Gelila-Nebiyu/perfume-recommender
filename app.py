import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Load and prepare dataset
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

# Generate visualizations
def generate_graphs(predictions, probabilities, recommendations, scent_type):
    os.makedirs('static/graphs', exist_ok=True)
    
    # Set Seaborn style for modern look
    sns.set(style="whitegrid", font_scale=1.3)
    gradient_colors = sns.color_palette("blend:#9333ea,#f472b6", n_colors=6)  # Purple to pink gradient
    
    # 1. Horizontal bar plot for recommendations
    plt.figure(figsize=(10, 6))
    bars = plt.barh(predictions[:6], probabilities[:6], color=gradient_colors, edgecolor='black')
    plt.title('Top 6 Perfume Recommendations', fontsize=18, weight='bold', pad=20, color='#4b0082')
    plt.xlabel('Recommendation Score', fontsize=12)
    plt.ylabel('Perfume Name', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Highlight top perfume
    bars[0].set_color('#c026d3')  # Bright magenta for top perfume
    
    # Add score labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{probabilities[i]:.2f}',
                 ha='left', va='center', fontsize=10, color='black')
    
    plt.tight_layout()
    plt.savefig('static/graphs/recommendations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature importance plot (sorted)
    plt.figure(figsize=(10, 6))
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = feature_importance[sorted_idx]
    
    bars = plt.bar(sorted_features, sorted_importance, color=gradient_colors, edgecolor='black')
    plt.title('Feature Importance in Recommendations', fontsize=18, weight='bold', pad=20, color='#4b0082')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Annotate top feature
    max_bar = bars[0]
    plt.text(max_bar.get_x() + max_bar.get_width()/2, max_bar.get_height(), f'{max_bar.get_height():.2f}',
             ha='center', va='bottom', fontsize=10, color='black', weight='bold')
    
    plt.tight_layout()
    plt.savefig('static/graphs/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Pie chart for scent type distribution
    plt.figure(figsize=(8, 8))
    scent_types = [rec['scent_type'] for rec in recommendations]
    scent_counts = pd.Series(scent_types).value_counts()
    labels = scent_counts.index
    sizes = scent_counts.values
    explode = [0.1 if i == 0 else 0 for i in range(len(labels))]  # Highlight most common scent
    
    plt.pie(sizes, labels=labels, colors=gradient_colors[:len(labels)], autopct='%1.1f%%', startangle=140,
            textprops={'fontsize': 12}, explode=explode, shadow=True)
    plt.title('Scent Type Distribution in Recommendations', fontsize=18, weight='bold', pad=20, color='#4b0082')
    
    plt.tight_layout()
    plt.savefig('static/graphs/scent_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input
        scent_type = request.form['scent_type']
        gender = request.form['gender']
        longevity = int(request.form['longevity'])
        sillage = int(request.form['sillage'])
        
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
            if perfume_data['gender'] == gender or gender == 'Unisex':
                recommendations.append({
                    'name': perfume,
                    'brand': perfume_data['brand'],
                    'scent_type': perfume_data['scent_type'],
                    'gender': perfume_data['gender'],
                    'longevity': perfume_data['longevity'],
                    'sillage': perfume_data['sillage'],
                    'score': round(score, 2)
                })
                count += 1
        
        # Generate graphs
        generate_graphs([rec['name'] for rec in recommendations], [rec['score'] for rec in recommendations], recommendations, scent_type)
        
        return render_template('results.html', recommendations=recommendations)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)