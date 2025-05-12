import pandas as pd
import numpy as np

# Define possible values
perfume_names = [
    "Chanel No. 5", "Dior Sauvage", "Creed Aventus", "Tom Ford Oud Wood", "Jo Malone Peony & Blush Suede",
    "YSL Libre", "Gucci Bloom", "Lancôme La Vie Est Belle", "Maison Margiela Replica Jazz Club",
    "Byredo Gypsy Water", "Chloé Nomade", "Armani Sì", "Versace Eros", "Prada Candy",
    "Bvlgari Omnia Crystalline", "Hermès Terre d’Hermès", "Givenchy L’Interdit", "Montblanc Explorer",
    "Penhaligon’s Juniper Sling", "Amouage Reflection", "Narciso Rodriguez For Her", "Le Labo Santal 33",
    "Diptyque Philosykos", "Tiziana Terenzi Kirke", "Atelier Cologne Clémentine California",
    "Frédéric Malle Portrait of a Lady", "Guerlain Mon Guerlain", "Maison Francis Kurkdjian Baccarat Rouge 540",
    "Acqua di Parma Colonia", "L’Artisan Parfumeur Mandarina Corsica", "Dolce & Gabbana Light Blue",
    "Burberry Her", "Viktor & Rolf Flowerbomb", "Jean Paul Gaultier Scandal", "Aesop Erémia",
    "Maison Margiela Replica By the Fireplace", "Byredo Rose Noir", "Chanel Coco Mademoiselle",
    "Dior Miss Dior", "YSL Black Opium", "Tom Ford Black Orchid", "Jo Malone Basil & Neroli",
    "Creed Silver Mountain Water", "Hermès Un Jardin Sur Le Nil", "Lancôme Trésor",
    "Guerlain Shalimar", "Armani Code", "Versace Bright Crystal", "Prada La Femme", "Bvlgari Goldea"
]
brands = [
    "Chanel", "Dior", "Creed", "Tom Ford", "Jo Malone", "YSL", "Gucci", "Lancôme", "Maison Margiela",
    "Byredo", "Chloé", "Armani", "Versace", "Prada", "Bvlgari", "Hermès", "Givenchy", "Montblanc",
    "Penhaligon’s", "Amouage", "Narciso Rodriguez", "Le Labo", "Diptyque", "Tiziana Terenzi",
    "Atelier Cologne", "Frédéric Malle", "Guerlain", "Maison Francis Kurkdjian", "Acqua di Parma",
    "L’Artisan Parfumeur", "Dolce & Gabbana", "Burberry", "Viktor & Rolf", "Jean Paul Gaultier", "Aesop"
]
scent_types = ["Floral", "Woody", "Citrus", "Oriental", "Fresh", "Fruity", "Gourmand", "Aquatic"]
genders = ["Female", "Male", "Unisex"]
top_notes = ["Citrus", "Bergamot", "Lemon"]
heart_notes = ["Rose", "Jasmine", "Lavender"]
base_notes = ["Sandalwood", "Vanilla", "Musk"]

# Generate synthetic dataset with balanced genders and scents
np.random.seed(42)
n_samples = 200  # Increased size for variety
data = {
    "perfume_name": np.random.choice(perfume_names, n_samples, replace=True),
    "brand": np.random.choice(brands, n_samples),
    "gender": np.random.choice(genders, n_samples, p=[0.33, 0.33, 0.34]),  # Balanced distribution
    "scent_type": np.random.choice(scent_types, n_samples, p=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]),  # Balanced scents
    "longevity": np.random.randint(1, 6, n_samples),
    "sillage": np.random.randint(1, 6, n_samples),
}
for note in top_notes + heart_notes + base_notes:
    data[note] = np.random.randint(0, 2, n_samples)

# Ensure gender-specific perfumes align with realistic assignments
for i in range(n_samples):
    perfume = data["perfume_name"][i]
    if perfume in ["Chanel No. 5", "YSL Libre", "Gucci Bloom", "Lancôme La Vie Est Belle", "Chanel Coco Mademoiselle", "Dior Miss Dior", "Guerlain Mon Guerlain", "Givenchy L’Interdit"]:
        data["gender"][i] = "Female"
    elif perfume in ["Dior Sauvage", "Creed Aventus", "Versace Eros", "Montblanc Explorer", "Hermès Terre d’Hermès", "Armani Code"]:
        data["gender"][i] = "Male"
    elif perfume in ["Jo Malone Basil & Neroli", "Atelier Cologne Clémentine California", "Acqua di Parma Colonia", "Creed Silver Mountain Water"]:
        data["gender"][i] = "Unisex"

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("perfume_dataset.csv", index=False)
print("Dataset generated and saved as perfume_dataset.csv")