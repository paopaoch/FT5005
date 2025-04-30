import pandas as pd
import numpy as np

# Charger les données existantes
try:
    original_df = pd.read_csv('data/stacking_data.csv')
    print("Loaded original data successfully")
except Exception as e:
    print(f"Error loading original data: {e}")
    # Si le chargement échoue, créer les données originales manuellement
    original_df = pd.DataFrame([
        ['BAC', 'Q1 2001', 123, 0.4, 123, 0.5, 123, 0.5],
        ['JPM', 'Q1 2001', 123, 0.4, 456, 0.8, 456, 0.8],
        ['WFC', 'Q1 2001', 123, 0.4, 789, 0.25, 789, 0.25]
    ], columns=['ticker', 'quarter_year', 'actual_log_rev', 'actual_car', 
               'log_revenue_prediction_1', 'CAR_prediction_1', 
               'log_revenue_prediction_2', 'CAR_prediction_2'])

# Définir des tickers et des périodes pour augmenter les données
# Réduire le nombre de tickers et d'années pour avoir environ 100 lignes
tickers = ['BAC', 'JPM', 'WFC', 'C', 'GS']  # Réduit à 5 tickers
quarters = ['Q1', 'Q2', 'Q3', 'Q4']
years = list(range(2018, 2023))  # Réduit à 5 ans (2018-2022)

# Créer un DataFrame vide pour les données augmentées
data = []

# Générer des données synthétiques pour chaque combinaison ticker-trimestre-année
for ticker in tickers:
    # Définir des paramètres de base spécifiques à chaque ticker pour plus de réalisme
    base_rev = np.random.uniform(5, 9)  # log de revenu de base entre 5 et 9
    rev_growth = np.random.uniform(0.01, 0.05)  # croissance trimestrielle entre 1% et 5%
    rev_volatility = np.random.uniform(0.05, 0.2)  # volatilité entre 5% et 20%
    
    base_car = np.random.uniform(-0.1, 0.1)  # CAR de base entre -10% et 10%
    car_volatility = np.random.uniform(0.02, 0.1)  # volatilité entre 2% et 10%
    
    # Définir les performances des modèles (erreurs relatives)
    model1_rev_error = np.random.uniform(-0.05, 0.05)
    model2_rev_error = np.random.uniform(-0.07, 0.07)
    
    model1_car_error = np.random.uniform(-0.03, 0.03)
    model2_car_error = np.random.uniform(-0.04, 0.04)
    
    time_idx = 0
    for year in years:
        for quarter in quarters:
            time_idx += 1
            quarter_year = f"{quarter} {year}"
            
            # Calculer le revenu réel avec une tendance et un bruit
            actual_rev = base_rev + rev_growth * time_idx + np.random.normal(0, rev_volatility)
            
            # Calculer le CAR réel avec un bruit
            actual_car = base_car + np.random.normal(0, car_volatility)
            
            # Prédictions des modèles (avec des erreurs différentes)
            rev_pred1 = actual_rev * (1 + model1_rev_error + np.random.normal(0, 0.01))
            rev_pred2 = actual_rev * (1 + model2_rev_error + np.random.normal(0, 0.015))
            
            car_pred1 = actual_car + model1_car_error + np.random.normal(0, 0.02)
            car_pred2 = actual_car + model2_car_error + np.random.normal(0, 0.025)
            
            # Ajouter la ligne au DataFrame
            data.append([
                ticker, 
                quarter_year, 
                round(actual_rev, 2), 
                round(actual_car, 4),
                round(rev_pred1, 2), 
                round(car_pred1, 4),
                round(rev_pred2, 2), 
                round(car_pred2, 4)
            ])

# Créer le DataFrame augmenté
synthetic_df = pd.DataFrame(data, columns=original_df.columns)

# Calculer combien de lignes nous avons générées
expected_rows = len(tickers) * len(years) * len(quarters)
print(f"Expected rows: {expected_rows}")

# Si nécessaire, tronquer le DataFrame pour avoir exactement 100 lignes
if len(synthetic_df) > 100:
    synthetic_df = synthetic_df.iloc[:100]
elif len(synthetic_df) < 100:
    # Si nous avons moins de 100 lignes, dupliquer certaines avec un peu de bruit
    rows_to_add = 100 - len(synthetic_df)
    extra_rows = []
    for i in range(rows_to_add):
        # Prendre une ligne aléatoire
        random_row = synthetic_df.iloc[np.random.randint(0, len(synthetic_df))].copy()
        # Ajouter un peu de bruit aux valeurs numériques
        random_row['actual_log_rev'] = round(random_row['actual_log_rev'] * (1 + np.random.normal(0, 0.01)), 2)
        random_row['actual_car'] = round(random_row['actual_car'] + np.random.normal(0, 0.005), 4)
        random_row['log_revenue_prediction_1'] = round(random_row['log_revenue_prediction_1'] * (1 + np.random.normal(0, 0.01)), 2)
        random_row['CAR_prediction_1'] = round(random_row['CAR_prediction_1'] + np.random.normal(0, 0.005), 4)
        random_row['log_revenue_prediction_2'] = round(random_row['log_revenue_prediction_2'] * (1 + np.random.normal(0, 0.01)), 2)
        random_row['CAR_prediction_2'] = round(random_row['CAR_prediction_2'] + np.random.normal(0, 0.005), 4)
        extra_rows.append(random_row)
    
    if extra_rows:
        extra_df = pd.DataFrame(extra_rows)
        synthetic_df = pd.concat([synthetic_df, extra_df], ignore_index=True)

# Sauvegarder les données augmentées
synthetic_df.to_csv('data/stacking_data_expanded_small.csv', index=False)

# Afficher quelques statistiques sur les données générées
print(f"Generated {len(synthetic_df)} rows of synthetic data")
print(f"Unique tickers: {synthetic_df['ticker'].nunique()}")
print(f"Time range: {synthetic_df['quarter_year'].min()} to {synthetic_df['quarter_year'].max()}")

# Afficher un échantillon
print("\nSample of generated data:")
print(synthetic_df.head(10))
