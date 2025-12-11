"""
Module utils - Fonctions d'analyse ML avancee avec prediction future
Logique complete de code essaye.py pour predire 2025-2027
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def charger_donnees():
    """Charge et prepare les donnees DVF avec feature engineering"""
    print("📂 Chargement et nettoyage des données...\n")
    
    df = pd.read_csv(r'.\data\dvf.csv', sep=',', skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df['Departement'] = df['INSEE_COM'].astype(str).str[:2]
    
    # Nettoyage
    df_clean = df.dropna(subset=['annee', 'Prixm2Moyen', 'nb_mutations']).copy()
    df_clean = df_clean[(df_clean['Prixm2Moyen'] > 100) & (df_clean['Prixm2Moyen'] < 20000)]
    df_clean = df_clean[df_clean['nb_mutations'] > 0]
    
    print(f"  ✓ Lignes nettoyées : {len(df_clean):,}")
    print(f"  ✓ Départements disponibles : {df_clean['Departement'].nunique()}\n")
    
    return df_clean

def creer_features_departement(df_dept):
    """Cree les features avancees pour un departement"""
    df_agg = df_dept.groupby('annee').agg({
        'Prixm2Moyen': 'mean',
        'SurfaceMoy': 'mean',
        'nb_mutations': 'sum',
        'NbMaisons': 'sum',
        'NbApparts': 'sum'
    }).reset_index()
    
    # Feature engineering avance
    df_agg['Annee_idx'] = df_agg['annee'] - df_agg['annee'].min()
    df_agg['PropMaisons'] = df_agg['NbMaisons'] / (df_agg['NbMaisons'] + df_agg['NbApparts'] + 1e-6)
    df_agg['Prix_MA3'] = df_agg['Prixm2Moyen'].rolling(window=3, min_periods=1).mean()
    df_agg['Prix_Growth'] = df_agg['Prixm2Moyen'].pct_change().fillna(0) * 100
    df_agg['Prix_Lag1'] = df_agg['Prixm2Moyen'].shift(1).fillna(df_agg['Prixm2Moyen'].iloc[0])
    
    return df_agg.fillna(0)

def calculer_metriques_avancees(y_true, y_pred, y_train_mean):
    """Calcule des metriques avancees pour evaluer la qualite du modele"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # Direction Accuracy
    if len(y_true) > 1:
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        direction_accuracy = np.mean(true_direction == pred_direction) * 100
    else:
        direction_accuracy = np.nan
    
    # Amelioration vs modele naif
    naive_rmse = np.sqrt(mean_squared_error(y_true, np.full_like(y_true, y_train_mean)))
    improvement_vs_naive = ((naive_rmse - rmse) / (naive_rmse + 1e-10)) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'direction_accuracy': direction_accuracy,
        'improvement_vs_naive': improvement_vs_naive
    }

def filtrer_donnees_par_zone(df, user_input):
    """Filtre les donnees selon la zone demandee"""
    if user_input.upper() == "FRANCE":
        df_filtered = df.copy()
        zone_name = "France entiere"
    elif len(user_input) == 5 and user_input.isdigit():
        # Code postal -> departement
        dept_code = user_input[:2]
        df_filtered = df[df['Departement'] == dept_code].copy()
        zone_name = f"Departement {dept_code}"
    elif user_input.isdigit() and len(user_input) <= 3:
        # Code departement
        dept_code = user_input.zfill(2)
        df_filtered = df[df['Departement'] == dept_code].copy()
        zone_name = f"Departement {dept_code}"
    else:
        # Par defaut: France
        df_filtered = df.copy()
        zone_name = user_input.title()
    
    if df_filtered.empty:
        print(f"Aucune donnee trouvee pour: {user_input}")
        return None, None
        
    print(f"Donnees filtrees: {len(df_filtered)} lignes pour {zone_name}")
    return df_filtered, zone_name

def calculer_statistiques_avancees(df_filtered, zone_name):
    """Calcule toutes les statistiques comme dans code essaye.py"""
    # Agregation par annee
    df_agg = df_filtered.groupby('annee').agg({
        'Prixm2Moyen': 'mean',
        'SurfaceMoy': 'mean',
        'nb_mutations': 'sum',
        'NbMaisons': 'sum',
        'NbApparts': 'sum'
    }).reset_index()
    
    stats = {
        'zone': zone_name,
        'nb_mutations': int(df_filtered['nb_mutations'].sum()),
        'prix_moyen_m2': float(df_filtered['Prixm2Moyen'].mean()),
        'surface_moyenne': float(df_filtered['SurfaceMoy'].mean()) if 'SurfaceMoy' in df_filtered.columns else 0.0
    }
    
    # Evolution temporelle
    evolution = df_agg.set_index('annee')['Prixm2Moyen'].to_dict()
    stats['evolution'] = {int(year): float(price) for year, price in evolution.items()}
    
    # Distribution des prix
    stats['prix_distribution'] = df_filtered['Prixm2Moyen'].values
    
    # Repartition par type
    if 'NbMaisons' in df_agg.columns and 'NbApparts' in df_agg.columns:
        maisons = df_agg['NbMaisons'].sum()
        apparts = df_agg['NbApparts'].sum()
        if maisons > 0 or apparts > 0:
            stats['repartition'] = {
                'Maisons': int(maisons),
                'Appartements': int(apparts)
            }
    
    return stats, df_agg

def entrainer_modeles_ml(df_agg):
    """Entraine les modeles ML comme dans code essaye.py"""
    if len(df_agg) < 5:
        return None
        
    try:
        # Features simples mais efficaces
        X = df_agg[['annee']].copy()
        y = df_agg['Prixm2Moyen'].copy()
        
        # Ajout de features si disponibles
        if 'SurfaceMoy' in df_agg.columns:
            X['SurfaceMoy'] = df_agg['SurfaceMoy'].fillna(df_agg['SurfaceMoy'].mean())
        if 'nb_mutations' in df_agg.columns:
            X['nb_mutations'] = df_agg['nb_mutations'].fillna(0)
        
        # Division train/test
        if len(X) > 3:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Modeles
        models = {
            'Regression Lineaire': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }
        
        predictions = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            predictions[name] = {
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'r2': float(r2_score(y_test, y_pred))
            }
        
        return predictions
        
    except Exception as e:
        print(f"Erreur ML: {e}")
        return None

def predire_departement_complet(df_clean, code_departement):
    """Prediction ML avancee pour un departement avec predictions futures"""
    # Filtrer le departement
    df_dept = df_clean[df_clean['Departement'] == code_departement]
    
    if len(df_dept) < 5:
        return None
    
    # Creer les features
    df_agg = creer_features_departement(df_dept)
    
    if len(df_agg) < 5:
        return None
    
    print(f"⚙️ Feature engineering en cours...")
    print(f"🧠 Entraînement de 5 modèles ML pour prédire 2025-2027...")
    
    # Features et cible
    feature_columns = ['Annee_idx', 'SurfaceMoy', 'nb_mutations', 'PropMaisons', 
                       'Prix_MA3', 'Prix_Growth', 'Prix_Lag1']
    
    X = df_agg[feature_columns].values
    y = df_agg['Prixm2Moyen'].values
    annees = df_agg['annee'].values
    
    # Split temporel pour prediction future
    n_years = len(df_agg)
    if n_years < 6:
        train_size = max(int(n_years * 0.75), n_years - 2)
        test_size = n_years - train_size
    else:
        test_size = max(int(n_years * 0.2), 2)
        train_size = n_years - test_size
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    annees_train = annees[:train_size]
    annees_test = annees[train_size:]
    
    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.array([])
    
    # Modeles ML avances
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1, max_iter=10000),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    }
    
    y_train_mean = np.mean(y_train)
    results = {}
    
    # Entrainement et evaluation de tous les modeles
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        
        # Predictions train
        y_train_pred = model.predict(X_train_scaled)
        train_metrics = calculer_metriques_avancees(y_train, y_train_pred, y_train_mean)
        
        # Predictions test
        if len(X_test_scaled) > 0:
            y_test_pred = model.predict(X_test_scaled)
            test_metrics = calculer_metriques_avancees(y_test, y_test_pred, y_train_mean)
        else:
            y_test_pred = np.array([])
            test_metrics = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'mape': np.nan, 'direction_accuracy': np.nan}
        
        results[name] = {
            'model': model,
            'train_r2': train_metrics['r2'],
            'train_rmse': train_metrics['rmse'],
            'test_r2': test_metrics['r2'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'direction_accuracy': test_metrics['direction_accuracy'],
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
    
    # Selection du meilleur modele base sur R2 test
    valid_models = {name: res for name, res in results.items() if not np.isnan(res['test_r2'])}
    if valid_models:
        best_model_name = max(valid_models.keys(), key=lambda k: valid_models[k]['test_r2'])
    else:
        # Fallback sur R2 train si pas de test valide
        best_model_name = max(results.keys(), key=lambda k: results[k]['train_r2'])
    
    best_model_info = results[best_model_name]
    best_model = best_model_info['model']
    
    print(f"🏆 Meilleur modèle sélectionné: {best_model_name} (R²={best_model_info['test_r2']:.3f})")
    
    # Predictions futures 2025-2027
    future_data = []
    last_row = df_agg.iloc[-1]
    
    for i, year in enumerate([2025, 2026, 2027]):
        if i == 0:
            prev_prix = df_agg.iloc[-1]['Prixm2Moyen']
        else:
            prev_prix = future_data[i-1]['prix_pred']
        
        annee_idx = year - df_agg['annee'].min()
        
        future_features = np.array([[
            annee_idx,
            last_row['SurfaceMoy'],
            last_row['nb_mutations'],
            last_row['PropMaisons'],
            prev_prix,
            0,
            prev_prix
        ]])
        
        future_features_scaled = scaler.transform(future_features)
        prix_pred = best_model.predict(future_features_scaled)[0]
        
        future_data.append({'annee': year, 'prix_pred': prix_pred})
    
    # Calcul des statistiques de base
    prix_2024 = df_agg.iloc[-1]['Prixm2Moyen']
    variation_pct = ((future_data[-1]['prix_pred'] - prix_2024) / prix_2024) * 100
    
    return {
        'departement': code_departement,
        'best_model_name': best_model_name,
        'results': results,
        'annees': annees,
        'y': y,
        'annees_train': annees_train,
        'y_train': y_train,
        'annees_test': annees_test,
        'y_test': y_test,
        'best_model_info': best_model_info,
        'prix_2024': prix_2024,
        'prix_2025': future_data[0]['prix_pred'],
        'prix_2026': future_data[1]['prix_pred'],
        'prix_2027': future_data[2]['prix_pred'],
        'variation_pct': variation_pct,
        'future_data': future_data,
        'zone_name': f"Département {code_departement}",
        'nb_mutations': int(df_dept['nb_mutations'].sum()),
        'prix_moyen_m2': float(df_dept['Prixm2Moyen'].mean()),
        'surface_moyenne': float(df_dept['SurfaceMoy'].mean()) if 'SurfaceMoy' in df_dept.columns else 0.0,
        'evolution': {int(year): float(price) for year, price in zip(annees, y)}
    }

def predire_france_complete(df_clean):
    """Prediction ML pour la France entiere (agregation nationale)"""
    print("🔍 Analyse de la France entière...\n")
    
    # Agregation nationale par annee
    df_france = df_clean.groupby('annee').agg({
        'Prixm2Moyen': 'mean',
        'SurfaceMoy': 'mean', 
        'nb_mutations': 'sum',
        'NbMaisons': 'sum',
        'NbApparts': 'sum'
    }).reset_index()
    
    if len(df_france) < 5:
        return None
    
    print("⚙️ Feature engineering pour la France...")
    print("🧠 Entraînement de 5 modèles ML pour prédire 2025-2027...")
    
    # Feature engineering pour la France
    df_france['Annee_idx'] = df_france['annee'] - df_france['annee'].min()
    df_france['PropMaisons'] = df_france['NbMaisons'] / (df_france['NbMaisons'] + df_france['NbApparts'] + 1e-6)
    df_france['Prix_MA3'] = df_france['Prixm2Moyen'].rolling(window=3, min_periods=1).mean()
    df_france['Prix_Growth'] = df_france['Prixm2Moyen'].pct_change().fillna(0) * 100
    df_france['Prix_Lag1'] = df_france['Prixm2Moyen'].shift(1).fillna(df_france['Prixm2Moyen'].iloc[0])
    df_france = df_france.fillna(0)
    
    # Features et cible
    feature_columns = ['Annee_idx', 'SurfaceMoy', 'nb_mutations', 'PropMaisons', 
                       'Prix_MA3', 'Prix_Growth', 'Prix_Lag1']
    
    X = df_france[feature_columns].values
    y = df_france['Prixm2Moyen'].values
    annees = df_france['annee'].values
    
    # Split temporel
    n_years = len(df_france)
    if n_years < 6:
        train_size = max(int(n_years * 0.75), n_years - 2)
        test_size = n_years - train_size
    else:
        test_size = max(int(n_years * 0.2), 2)
        train_size = n_years - test_size
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    annees_train = annees[:train_size]
    annees_test = annees[train_size:]
    
    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.array([])
    
    # Modeles ML
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1, max_iter=10000),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    }
    
    y_train_mean = np.mean(y_train)
    results = {}
    
    # Entrainement des modeles
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        
        y_train_pred = model.predict(X_train_scaled)
        train_metrics = calculer_metriques_avancees(y_train, y_train_pred, y_train_mean)
        
        if len(X_test_scaled) > 0:
            y_test_pred = model.predict(X_test_scaled)
            test_metrics = calculer_metriques_avancees(y_test, y_test_pred, y_train_mean)
        else:
            y_test_pred = np.array([])
            test_metrics = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'direction_accuracy': np.nan}
        
        results[name] = {
            'model': model,
            'train_r2': train_metrics['r2'],
            'train_rmse': train_metrics['rmse'],
            'test_r2': test_metrics['r2'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'direction_accuracy': test_metrics['direction_accuracy'],
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
    
    # Selection du meilleur modele
    valid_models = {name: res for name, res in results.items() if not np.isnan(res['test_r2'])}
    if valid_models:
        best_model_name = max(valid_models.keys(), key=lambda k: valid_models[k]['test_r2'])
    else:
        best_model_name = max(results.keys(), key=lambda k: results[k]['train_r2'])
    
    best_model_info = results[best_model_name]
    best_model = best_model_info['model']
    
    print(f"🏆 Meilleur modèle sélectionné: {best_model_name} (R²={best_model_info['test_r2']:.3f})")
    
    # Predictions futures
    future_data = []
    last_row = df_france.iloc[-1]
    
    for i, year in enumerate([2025, 2026, 2027]):
        if i == 0:
            prev_prix = df_france.iloc[-1]['Prixm2Moyen']
        else:
            prev_prix = future_data[i-1]['prix_pred']
        
        annee_idx = year - df_france['annee'].min()
        
        future_features = np.array([[
            annee_idx,
            last_row['SurfaceMoy'],
            last_row['nb_mutations'],
            last_row['PropMaisons'],
            prev_prix,
            0,
            prev_prix
        ]])
        
        future_features_scaled = scaler.transform(future_features)
        prix_pred = best_model.predict(future_features_scaled)[0]
        
        future_data.append({'annee': year, 'prix_pred': prix_pred})
    
    # Calculs finaux
    prix_2024 = df_france.iloc[-1]['Prixm2Moyen']
    variation_pct = ((future_data[-1]['prix_pred'] - prix_2024) / prix_2024) * 100
    
    return {
        'departement': 'FRANCE',
        'best_model_name': best_model_name,
        'results': results,
        'annees': annees,
        'y': y,
        'annees_train': annees_train,
        'y_train': y_train,
        'annees_test': annees_test,
        'y_test': y_test,
        'best_model_info': best_model_info,
        'prix_2024': prix_2024,
        'prix_2025': future_data[0]['prix_pred'],
        'prix_2026': future_data[1]['prix_pred'],
        'prix_2027': future_data[2]['prix_pred'],
        'variation_pct': variation_pct,
        'future_data': future_data,
        'zone_name': "France entière",
        'nb_mutations': int(df_clean['nb_mutations'].sum()),
        'prix_moyen_m2': float(df_clean['Prixm2Moyen'].mean()),
        'surface_moyenne': float(df_clean['SurfaceMoy'].mean()) if 'SurfaceMoy' in df_clean.columns else 0.0,
        'evolution': {int(year): float(price) for year, price in zip(annees, y)}
    }

def analyze_data(user_input):
    """Fonction principale d'analyse avec prediction ML avancee"""
    try:
        # Chargement des donnees
        df_clean = charger_donnees()
        if df_clean is None:
            return None
        
        # Analyse selon l'input
        if user_input.upper() == "FRANCE":
            # Analyse de toute la France
            result = predire_france_complete(df_clean)
            return result
        
        # Extraire le code departement pour les autres cas
        if len(user_input) >= 2:
            code_dept = user_input[:2]
        else:
            print(f"Code invalide: {user_input}")
            return None
        
        # Verifier que le departement existe
        if code_dept not in df_clean['Departement'].values:
            print(f"❌ Département {code_dept} non trouvé")
            depts_dispo = sorted(df_clean['Departement'].unique())
            print(f"Départements disponibles : {', '.join(depts_dispo[:10])}...")
            return None
        
        print(f"🔍 Analyse du département {code_dept}...\n")
        
        # Prediction ML pour le departement
        result = predire_departement_complet(df_clean, code_dept)
        
        return result
        
    except Exception as e:
        print(f"Erreur lors de l'analyse: {e}")
        import traceback
        traceback.print_exc()
        return None
