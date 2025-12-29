"""
ANALYSEUR IMMOBILIER AVANCÉ - PROJET INFO 3
Combine les données historiques de 200 ans + données DVF récentes (2014-2024)
Utilise les données historiques pour améliorer les prédictions ML
"""
VERBOSE = False

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def charger_donnees_historiques():
    """Charge les données historiques de 200 ans"""
    vprint("[INFO] Chargement des donnees historiques (200 ans)...")         # n'affiche pas dans le terminal mais ne detruit pas le code
    
    try:
        # Essayer de charger le fichier CSV d'abord
        df_historique = pd.read_csv('data/valeur_immobilier_france.csv')
        
        # Nettoyer et préparer les données historiques
        df_historique.columns = df_historique.columns.str.strip()
        
        # Identifier les colonnes année et prix
        possible_year_cols = ['annee', 'Annee', 'Year', 'year', 'Date', 'date']
        possible_price_cols = ['prix', 'Prix', 'price', 'Price', 'valeur', 'Valeur', 'value', 'Value']
        
        year_col = None
        price_col = None
        
        for col in df_historique.columns:
            if any(year_name.lower() in col.lower() for year_name in possible_year_cols):
                year_col = col
                break
        
        for col in df_historique.columns:
            if any(price_name.lower() in col.lower() for price_name in possible_price_cols):
                price_col = col
                break
        
        if year_col and price_col:
            df_hist_clean = df_historique[[year_col, price_col]].copy()
            df_hist_clean.columns = ['annee', 'prix_historique']
            df_hist_clean = df_hist_clean.dropna()
            
            # Convertir en numérique
            df_hist_clean['annee'] = pd.to_numeric(df_hist_clean['annee'], errors='coerce')
            df_hist_clean['prix_historique'] = pd.to_numeric(df_hist_clean['prix_historique'], errors='coerce')
            df_hist_clean = df_hist_clean.dropna()
            
            # Filtrer à partir de 1940 pour avoir des données plus fiables et cohérentes
            df_hist_clean = df_hist_clean[
                (df_hist_clean['annee'] >= 1940) & 
                (df_hist_clean['annee'] <= 2020) &
                (df_hist_clean['prix_historique'] > 0)
            ]
            
            vprint(f"  [OK] Donnees historiques : {len(df_hist_clean)} annees ({df_hist_clean['annee'].min():.0f}-{df_hist_clean['annee'].max():.0f})")
            return df_hist_clean
        else:
            print(f"  [WARNING] Colonnes non trouvees. Colonnes disponibles: {df_historique.columns.tolist()}")
            return None
            
    except Exception as e:
        print(f"  [ERROR] Erreur chargement donnees historiques: {e}")
        return None

def charger_donnees_dvf():
    """Charge les données DVF récentes"""
    vprint("[INFO] Chargement des donnees DVF recentes (2014-2024)...")
    
    try:
        df_dvf = pd.read_csv('data/dvf.csv', sep=',', skipinitialspace=True)
        df_dvf.columns = df_dvf.columns.str.strip()
        
        # Nettoyage DVF
        df_dvf_clean = df_dvf.dropna(subset=['annee', 'Prixm2Moyen', 'nb_mutations']).copy()
        df_dvf_clean = df_dvf_clean[(df_dvf_clean['Prixm2Moyen'] > 100) & (df_dvf_clean['Prixm2Moyen'] < 20000)]
        df_dvf_clean = df_dvf_clean[df_dvf_clean['nb_mutations'] > 0]
        
        # Agrégation France par année (séries moyennes)
        df_france = df_dvf_clean.groupby('annee').agg({
            'Prixm2Moyen': 'mean',
            'SurfaceMoy': 'mean',
            'nb_mutations': 'sum'
        }).reset_index()
        
        df_france.columns = ['annee', 'prix_dvf', 'surface_moy', 'nb_mutations']
        
        vprint(f"  [OK] Donnees DVF : {len(df_france)} annees ({df_france['annee'].min():.0f}-{df_france['annee'].max():.0f})")
        vprint(f"  [OK] Prix moyen DVF : {df_france['prix_dvf'].mean():.2f} €/m2")
        
        # Retourner à la fois l'agrégation France et le dataframe nettoyé (utile pour agrégations locales)
        return df_france, df_dvf_clean
        
    except Exception as e:
        print(f"  [ERROR] Erreur chargement donnees DVF: {e}")
        return None

def fusionner_donnees(df_historique, df_dvf):
    """Fusionne et normalise les données historiques et DVF"""
    vprint("[INFO] Fusion et normalisation des donnees...")
    
    if df_historique is None or df_dvf is None:
        return None
    
    # Normaliser les données historiques pour qu'elles correspondent à l'échelle DVF
    # On va utiliser les années de chevauchement pour calibrer
    
    # Trouver les années communes
    annees_communes = set(df_historique['annee'].astype(int)).intersection(set(df_dvf['annee'].astype(int)))
    
    if len(annees_communes) > 0:
        vprint(f"  [OK] Annees communes trouvees : {sorted(annees_communes)}")
        
        # Calculer le facteur de normalisation
        prix_hist_communes = df_historique[df_historique['annee'].isin(annees_communes)]['prix_historique'].mean()
        prix_dvf_communes = df_dvf[df_dvf['annee'].isin(annees_communes)]['prix_dvf'].mean()
        
        facteur_normalisation = prix_dvf_communes / prix_hist_communes
        vprint(f"  [OK] Facteur de normalisation : {facteur_normalisation:.2f}")
        
        # Normaliser les prix historiques
        df_historique_norm = df_historique.copy()
        df_historique_norm['prix_normalise'] = df_historique_norm['prix_historique'] * facteur_normalisation
    else:
        print("  [WARNING] Pas d'annees communes, normalisation par moyenne generale")
        facteur_normalisation = df_dvf['prix_dvf'].mean() / df_historique['prix_historique'].mean()
        df_historique_norm = df_historique.copy()
        df_historique_norm['prix_normalise'] = df_historique_norm['prix_historique'] * facteur_normalisation
    
    # Créer le dataset fusionné
    df_fusionne = df_historique_norm[['annee', 'prix_normalise']].copy()
    df_fusionne.columns = ['annee', 'prix']
    df_fusionne['source'] = 'historique'
    
    df_dvf_simple = df_dvf[['annee', 'prix_dvf']].copy()
    df_dvf_simple.columns = ['annee', 'prix']
    df_dvf_simple['source'] = 'dvf'
    
    # Concaténer
    df_complet = pd.concat([df_fusionne, df_dvf_simple], ignore_index=True)
    df_complet = df_complet.sort_values('annee').reset_index(drop=True)
    
    vprint(f"  [OK] Dataset fusionne : {len(df_complet)} points de donnees")
    vprint(f"  [OK] Periode couverte : {df_complet['annee'].min():.0f}-{df_complet['annee'].max():.0f}")
    
    return df_complet, df_dvf


def agregation_par_code_postal(df_dvf_full, code_postal=None, departement_code=None):
    """Retourne un DataFrame agrégé par année pour un code postal ou un département.

    - Si `code_postal` est fourni, filtre sur `Code_postal`.
    - Si `departement_code` est fourni (2 digits), filtre sur `nom_departement` ou `INSEE_COM`.
    Retourne un DataFrame avec colonnes `annee`, `prix_dvf`, `surface_moy`, `nb_mutations`.
    """
    df = df_dvf_full.copy()
    # Normaliser le code postal si nécessaire
    if code_postal is not None:
        code_postal = str(code_postal).strip().zfill(5)
        df_sel = df[df['Code_postal'] == code_postal]
    elif departement_code is not None:
        dep = str(departement_code).zfill(2)
        # INSEE_COM contient le code commune en début; extraire 2 premières chars
        df['code_departement'] = df['INSEE_COM'].astype(str).str[:2]
        df_sel = df[df['code_departement'] == dep]
        df = df.drop(columns=['code_departement'], errors='ignore')
    else:
        return None

    if df_sel.empty:
        return pd.DataFrame()

    df_agg = df_sel.groupby('annee').agg({
        'Prixm2Moyen': 'mean',
        'SurfaceMoy': 'mean',
        'nb_mutations': 'sum'
    }).reset_index()
    df_agg.columns = ['annee', 'prix_dvf', 'surface_moy', 'nb_mutations']
    return df_agg

def creer_features_avancees(df_complet):
    """Cree des features avancees (utilisees dans la version hybride stable)."""
    vprint("[INFO] Creation de features avancees avec donnees historiques...")
    df_features = df_complet.copy()
    df_features['annee_idx'] = df_features['annee'] - df_features['annee'].min()
    df_features['decennie'] = (df_features['annee'] // 10) * 10
    df_features['siecle'] = (df_features['annee'] // 100) * 100
    df_features['prix_ma5'] = df_features['prix'].rolling(window=5, min_periods=1).mean()
    df_features['prix_ma10'] = df_features['prix'].rolling(window=10, min_periods=1).mean()
    df_features['prix_ma20'] = df_features['prix'].rolling(window=20, min_periods=1).mean()
    df_features['prix_growth'] = df_features['prix'].pct_change().fillna(0) * 100
    df_features['prix_volatility'] = df_features['prix_growth'].rolling(window=10, min_periods=1).std().fillna(0)
    df_features['cycle_court'] = np.sin(2 * np.pi * df_features['annee_idx'] / 10)
    df_features['cycle_moyen'] = np.sin(2 * np.pi * df_features['annee_idx'] / 25)
    df_features['cycle_long'] = np.sin(2 * np.pi * df_features['annee_idx'] / 50)
    df_features['tendance_lineaire'] = df_features['annee_idx']
    df_features['tendance_quadratique'] = df_features['annee_idx'] ** 2
    crises = [1870, 1914, 1929, 1939, 1973, 2008]
    for crise in crises:
        df_features[f'dist_crise_{crise}'] = np.abs(df_features['annee'] - crise)
    vprint(f"  [OK] Features creees : {len([col for col in df_features.columns if col not in ['annee', 'prix', 'source']])}")
    return df_features

def creer_features_pour_annee(df_features, annee_cible, feature_columns):
    """Crée les features pour une année donnée en utilisant l'historique disponible"""
    # Filtrer les données jusqu'à l'année précédente
    historical_data = df_features[df_features['annee'] < annee_cible].copy()
    
    if len(historical_data) < 5:  # Pas assez d'historique
        return None
    
    # Créer une ligne avec l'année cible pour générer les features
    temp_row = pd.DataFrame({'annee': [annee_cible], 'prix': [0], 'source': ['prediction']})
    temp_df = pd.concat([historical_data, temp_row], ignore_index=True)
    
    # Régénérer les features avec cette nouvelle ligne
    temp_df = creer_features_avancees(temp_df)
    
    # Retourner les features de la dernière ligne (année cible)
    if not temp_df.empty:
        features_row = temp_df.iloc[-1]
        return features_row[feature_columns].values
    return None

def entrainer_modeles(df_features):
    """Entraîne les modèles ML avec tuning léger et baseline.
    Retourne le dictionnaire des résultats, le nom du meilleur modèle (selon R² validation robuste),
    et les masques train/test pour la suite.
    """
    vprint("[INFO] Entrainement des modeles ML avances (avec tuning)...")
    feature_columns = [col for col in df_features.columns if col not in ['annee', 'prix', 'source']]
    X = df_features[feature_columns].values
    y = df_features['prix'].values

    # Validation robuste puis entraînement final
    train_mask_robust = df_features['annee'] <= 2010
    test_mask_robust = (df_features['annee'] > 2010) & (df_features['annee'] <= 2020)
    train_mask = df_features['annee'] <= 2020
    test_years = [2021, 2022, 2023, 2024]
    test_mask = df_features['annee'].isin(test_years)

    X_train_robust = X[train_mask_robust]
    y_train_robust = y[train_mask_robust]
    X_test_robust = X[test_mask_robust]
    y_test_robust = y[test_mask_robust]

    X_train_final = X[train_mask]
    y_train_final = y[train_mask]

    test_data = df_features[test_mask].copy() if test_mask.sum() > 0 else pd.DataFrame()

    train_data = df_features[train_mask]
    hist_data = train_data[train_data['source'] == 'historique']
    dvf_data = train_data[train_data['source'] == 'dvf']

    vprint(f"  [OK] ENTRAINEMENT sur {len(X_train_final)} points total:")
    if len(hist_data) > 0:
        vprint(f"    [INFO] {len(hist_data)} points historiques ({hist_data['annee'].min():.0f}-{hist_data['annee'].max():.0f})")
    else:
        vprint("    [INFO] 0 points historiques")
    if len(dvf_data) > 0:
        vprint(f"    [INFO] {len(dvf_data)} points DVF ({dvf_data['annee'].min():.0f}-{dvf_data['annee'].max():.0f})")
    else:
        vprint("    [INFO] 0 points DVF")

    decennies = {}
    for _, row in train_data.iterrows():
        decennie = int(row['annee'] // 10) * 10
        decennies[decennie] = decennies.get(decennie, 0) + 1

    vprint("    [INFO] Couverture par decennie:")
    for dec in sorted(decennies.keys()):
        vprint(f"      {dec}s : {decennies[dec]} points")

    vprint(f"  [INFO] VALIDATION sur {len(test_data)} annees recentes:")
    if len(test_data) > 0:
        for year in sorted(test_data['annee'].values):
            prix_reel = test_data[test_data['annee'] == year]['prix'].iloc[0]
            vprint(f"      {year:.0f} : {prix_reel:.0f} €/m² (valeur reelle)")

    # Scalers
    scaler_robust = StandardScaler()
    X_train_robust_scaled = scaler_robust.fit_transform(X_train_robust)
    X_test_robust_scaled = scaler_robust.transform(X_test_robust)

    scaler_final = StandardScaler()
    X_train_final_scaled = scaler_final.fit_transform(X_train_final)

    results = {}
    
    # Ridge grid
    vprint("    Recherche grille Ridge (tuning alpha)...")
    alphas_ridge = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_ridge = (None, None, -np.inf)
    for a in alphas_ridge:
        m = Ridge(alpha=a)
        m.fit(X_train_robust_scaled, y_train_robust)
        if len(y_test_robust) > 0:
            val_r2 = r2_score(y_test_robust, m.predict(X_test_robust_scaled))
        else:
            val_r2 = -np.inf
        if val_r2 > best_ridge[2]:
            best_ridge = (m, a, val_r2)
    ridge_final = Ridge(alpha=best_ridge[1]).fit(X_train_final_scaled, y_train_final)
    results[f'Ridge_alpha_{best_ridge[1]}'] = {
        'model': ridge_final,
        'scaler': scaler_final,
        'train_r2': r2_score(y_train_final, ridge_final.predict(X_train_final_scaled)),
        'robust_r2': best_ridge[2],
        'y_train_pred': ridge_final.predict(X_train_final_scaled),
        'y_test_pred': np.array([]),
        'feature_columns': feature_columns
    }

    # Lasso grid
    vprint("    Recherche grille Lasso (tuning alpha)...")
    alphas_lasso = [0.0001, 0.001, 0.01, 0.1]
    best_lasso = (None, None, -np.inf)
    for a in alphas_lasso:
        m = Lasso(alpha=a, max_iter=10000)
        m.fit(X_train_robust_scaled, y_train_robust)
        if len(y_test_robust) > 0:
            val_r2 = r2_score(y_test_robust, m.predict(X_test_robust_scaled))
        else:
            val_r2 = -np.inf
        if val_r2 > best_lasso[2]:
            best_lasso = (m, a, val_r2)
    lasso_final = Lasso(alpha=best_lasso[1], max_iter=10000).fit(X_train_final_scaled, y_train_final)
    results[f'Lasso_alpha_{best_lasso[1]}'] = {
        'model': lasso_final,
        'scaler': scaler_final,
        'train_r2': r2_score(y_train_final, lasso_final.predict(X_train_final_scaled)),
        'robust_r2': best_lasso[2],
        'y_train_pred': lasso_final.predict(X_train_final_scaled),
        'y_test_pred': np.array([]),
        'feature_columns': feature_columns
    }

    # Calcul des métriques de test temporel (2021-2024) pour chaque modèle
    for name, info in list(results.items()):
        model = info['model']
        scaler = info['scaler']
        # prédictions pour les années tests (utiliser creer_features_pour_annee)
        y_test_real = []
        y_test_pred = []
        for year in test_years:
            if year in df_features['annee'].values:
                real_row = df_features[df_features['annee'] == year].iloc[0]
                y_test_real.append(real_row['prix'])
                feat = creer_features_pour_annee(df_features, year, feature_columns)
                if feat is not None:
                    if scaler is not None:
                        feat_scaled = scaler.transform([feat])
                        pred = model.predict(feat_scaled)[0]
                    else:
                        pred = model.predict([feat])[0]
                    y_test_pred.append(pred)
        if len(y_test_real) > 0 and len(y_test_pred) > 0:
            y_test_real = np.array(y_test_real)
            y_test_pred = np.array(y_test_pred)
            valid_mask = ~np.isnan(y_test_pred)
            if valid_mask.sum() > 0:
                y_test_real_valid = y_test_real[valid_mask]
                y_test_pred_valid = y_test_pred[valid_mask]
                test_r2 = r2_score(y_test_real_valid, y_test_pred_valid)
                test_mae = mean_absolute_error(y_test_real_valid, y_test_pred_valid)
            else:
                test_r2 = test_mae = np.nan
        else:
            test_r2 = test_mae = np.nan

        # enrich results
        info['test_r2'] = test_r2
        info['test_mae'] = test_mae
        info['y_test_pred'] = np.array(y_test_pred)

    # Choisir le meilleur modèle selon R² robuste (validation historique 2011-2020)
    best_model_name = max(results.keys(), key=lambda k: (results[k]['robust_r2'] if not np.isnan(results[k]['robust_r2']) else -np.inf))

    vprint(f"  [RESULT] Meilleur modele : {best_model_name} (R2 validation: {results[best_model_name]['robust_r2']:.3f})")

    return results, best_model_name, train_mask, test_mask

def predire_futur(results, best_model_name, df_features):
    """Prédictions futures 2025-2027 basées sur l'historique complet"""
    vprint("[INFO] Predictions futures avec modele avance...")
    
    best_model_info = results[best_model_name]
    model = best_model_info['model']
    scaler = best_model_info['scaler']
    feature_columns = best_model_info['feature_columns']
    
    # Données de base pour les prédictions
    last_row = df_features.iloc[-1]
    annee_min = df_features['annee'].min()
    
    predictions_futures = []
    
    for year in [2025, 2026, 2027]:
        future_features = {}
        annee_idx = year - annee_min

        # Build features dynamically based on feature_columns
        for col in feature_columns:
            if col == 'annee_idx':
                future_features[col] = annee_idx
            elif col in ['decennie', 'siecle']:
                if col == 'decennie':
                    future_features[col] = (year // 10) * 10
                else:
                    future_features[col] = (year // 100) * 100
            elif col in ['prix_ma5', 'prix_ma10', 'prix_ma20', 'prix_ma50']:
                # use last known moving average if available
                future_features[col] = float(last_row[col]) if col in last_row.index else float(last_row.get('prix', 0))
            elif col in ['prix_growth', 'prix_growth_5']:
                future_features[col] = 0.0
            elif col == 'prix_volatility':
                future_features[col] = float(last_row[col]) if col in last_row.index else 0.0
            elif col in ['cycle_court', 'cycle_moyen', 'cycle_long']:
                if col == 'cycle_court':
                    future_features[col] = np.sin(2 * np.pi * annee_idx / 10)
                elif col == 'cycle_moyen':
                    future_features[col] = np.sin(2 * np.pi * annee_idx / 25)
                else:
                    future_features[col] = np.sin(2 * np.pi * annee_idx / 50)
            elif col in ['tendance_lineaire', 'tendance_quadratique']:
                future_features[col] = annee_idx if col == 'tendance_lineaire' else annee_idx ** 2
            elif col.startswith('dist_crise_'):
                # compute distance to crisis year embedded in column name
                try:
                    crise_year = int(col.split('_')[-1])
                    future_features[col] = abs(year - crise_year)
                except Exception:
                    future_features[col] = float(last_row.get(col, 0))
            else:
                # fallback: try to reuse last known value or zero
                future_features[col] = float(last_row[col]) if col in last_row.index else 0.0

        # Create feature vector matching order
        X_future = np.array([[future_features.get(col, 0.0) for col in feature_columns]])
        X_future_scaled = scaler.transform(X_future)
        prix_pred = model.predict(X_future_scaled)[0]
        predictions_futures.append({'annee': year, 'prix_pred': float(prix_pred)})
    
    return predictions_futures


def _compute_dvf_metrics_for_models(results_dict, df_features, df_dvf_agg):
    """Calcule R2/MAE des modèles spécifiquement sur la série DVF agrégée.

    Modifie `results_dict` en place en ajoutant `dvf_r2` et `dvf_mae` pour chaque modèle si possible.
    """
    from sklearn.metrics import r2_score, mean_absolute_error
    if df_dvf_agg is None or df_dvf_agg.empty:
        return

    feature_columns = None
    for name, info in results_dict.items():
        model = info.get('model')
        scaler = info.get('scaler')
        feature_columns = info.get('feature_columns')
        if model is None or feature_columns is None:
            info['dvf_r2'] = np.nan
            info['dvf_mae'] = np.nan
            continue

        y_real = []
        y_pred = []
        for _, row in df_dvf_agg.iterrows():
            year = int(row['annee'])
            real = float(row['prix_dvf'])
            feat = creer_features_pour_annee(df_features, year, feature_columns)
            if feat is None:
                continue
            if scaler is not None:
                try:
                    pred = model.predict(scaler.transform([feat]))[0]
                except Exception:
                    pred = model.predict([feat])[0]
            else:
                pred = model.predict([feat])[0]
            y_real.append(real)
            y_pred.append(pred)

        if len(y_real) > 0 and len(y_pred) > 0:
            try:
                info['dvf_r2'] = float(r2_score(np.array(y_real), np.array(y_pred)))
                info['dvf_mae'] = float(mean_absolute_error(np.array(y_real), np.array(y_pred)))
            except Exception:
                info['dvf_r2'] = np.nan
                info['dvf_mae'] = np.nan
        else:
            info['dvf_r2'] = np.nan
            info['dvf_mae'] = np.nan



def sauvegarder_modele(best_model_info, best_model_name, path='models'):
    """Sauvegarde le modèle, le scaler et les métadonnées dans le dossier `models/`"""
    import os
    import pickle

    os.makedirs(path, exist_ok=True)
    model = best_model_info.get('model')
    scaler = best_model_info.get('scaler')
    feature_columns = best_model_info.get('feature_columns')

    meta = {
        'model_name': best_model_name,
        'feature_columns': feature_columns
    }

    try:
        with open(os.path.join(path, f'{best_model_name}_model.pkl'), 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler, 'meta': meta}, f)
        vprint(f"[OK] Modele sauvegarde : {os.path.join(path, best_model_name + '_model.pkl')}")
    except Exception as e:
        print(f"[ERROR] Echec sauvegarde modele : {e}")

def analyze_data(user_input):
    """Fonction principale d'analyse"""
    try:
        vprint("[INFO] ANALYSEUR AVANCE - Donnees historiques 200 ans + DVF recentes")
        print("=" * 70)
        
        # 1. Charger les deux sources
        df_historique = charger_donnees_historiques()
        dvf_result = charger_donnees_dvf()
        if dvf_result is None:
            print("[ERROR] Impossible de charger les donnees DVF")
            return None
        df_dvf_france, df_dvf_full = dvf_result

        if df_historique is None:
            print("[ERROR] Impossible de charger les donnees historiques")
            return None
        
        # 2. Fusionner
        # Pour FRANCE, on fusionne avec l'agrégation nationale
        result = fusionner_donnees(df_historique, df_dvf_france)
        if result is None:
            return None
        df_complet, df_dvf_original = result
        
        # 3. Features avancées
        df_features = creer_features_avancees(df_complet)
        
        # 4. Entraînement ML
        results, best_model_name, train_mask, test_mask = entrainer_modeles(df_features)
        
        # 5. Prédictions futures
        predictions_futures = predire_futur(results, best_model_name, df_features)
        
        # 6. Préparer les résultats
        best_info = results[best_model_name]

        # Sauvegarder automatiquement le meilleur modèle (optionnel)
        try:
            sauvegarder_modele(best_info, best_model_name)
        except Exception:
            pass
        
        # Prix de référence 2024
        prix_2024 = df_features[df_features['annee'] == 2024]['prix'].iloc[-1] if len(df_features[df_features['annee'] == 2024]) > 0 else df_features.iloc[-1]['prix']
        
        variation_pct = ((predictions_futures[-1]['prix_pred'] - prix_2024) / prix_2024) * 100
        
        return {
            'type': 'hybride',
            'zone_name': 'France (Analyse complète 200 ans)',
            'nb_mutations': 'N/A (données historiques)',
            'prix_moyen_m2': float(df_complet[df_complet['source'] == 'dvf']['prix'].mean()),
            'surface_moyenne': 'N/A',
            
            # Données pour graphiques
            'df_historique': df_historique,
            'df_dvf': df_dvf_original,
            'df_complet': df_complet,
            'annees_train': df_features[train_mask]['annee'].values,
            'y_train': df_features[train_mask]['prix'].values,
            'annees_test': df_features[test_mask]['annee'].values if test_mask.sum() > 0 else np.array([]),
            'y_test': df_features[test_mask]['prix'].values if test_mask.sum() > 0 else np.array([]),
            
            # Modèle
            'best_model_name': best_model_name,
            'best_model_info': best_info,
            'results': results,
            
            # Prédictions
            'prix_2024': prix_2024,
            'prix_2025': predictions_futures[0]['prix_pred'],
            'prix_2026': predictions_futures[1]['prix_pred'],
            'prix_2027': predictions_futures[2]['prix_pred'],
            'variation_pct': variation_pct,
            'future_data': predictions_futures,
            
            # Évolution pour affichage
            'evolution': {int(row['annee']): float(row['prix']) for _, row in df_complet.iterrows()}
        }
        
    except Exception as e:
        print(f"[ERROR] Erreur dans l'analyse: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_postal(code_or_postal):
    """Analyse pour un `code_postal` (5 digits) ou un `departement` (2 digits) passé en string/int.

    Retourne un dictionnaire similaire à `analyze_data` mais basé sur l'agrégation DVF locale.
    Si le niveau local n'a pas assez de points (moins de 5 années), retourne None.
    """
    try:
        # Charger historiques et DVF complets
        df_historique = charger_donnees_historiques()
        dvf_result = charger_donnees_dvf()
        if dvf_result is None:
            return None
        df_dvf_france, df_dvf_full = dvf_result

        # Détecter si input est code postal (5 chiffres) ou département (2 chiffres)
        s = str(code_or_postal).strip()
        df_local = None
        if len(s) == 5 and s.isdigit():
            df_local = agregation_par_code_postal(df_dvf_full, code_postal=s)
            zone_name = f'Code postal {s}'
        elif len(s) <= 3 and s.isdigit():
            dep = s.zfill(2)
            df_local = agregation_par_code_postal(df_dvf_full, departement_code=dep)
            zone_name = f'Département {dep}'
        else:
            print(f"Entrée locale non reconnue: {s}")
            return None

        # Si pas assez de données locales, tenter un repli automatique sur le département
        if df_local is None or df_local.empty or len(df_local) < 5:
            # Si l'entrée était un code postal, extraire les 2 premiers chiffres
            if len(s) == 5 and s.isdigit():
                dep_try = s[:2]
                vprint(f"[INFO] Pas assez de donnees pour {s}, tentative sur departement {dep_try}...")
                df_local = agregation_par_code_postal(df_dvf_full, departement_code=dep_try)
                zone_name = f'Departement {dep_try}'

        if df_local is None or df_local.empty or len(df_local) < 5:
            print(f"[WARNING] Pas assez de donnees locales pour {code_or_postal}")
            return None

        # Fusionner l'historique national (converti) avec la série locale DVF
        # On conserve l'historique national pour les trends et on ajoute la série locale DVF
        df_complet, _ = fusionner_donnees(df_historique, df_dvf_france)

        # Remplacer les points DVF nationaux par les points locaux pour les années communes
        df_local_simple = df_local[['annee', 'prix_dvf']].copy()
        df_local_simple.columns = ['annee', 'prix']
        df_local_simple['source'] = 'dvf_local'

        # Construire df_complet_local: garder historique normalisé + dvf_local
        df_historique_norm = df_complet[df_complet['source'] == 'historique'][['annee', 'prix']].copy()
        df_local_f = df_local_simple.copy()
        df_local_f = df_local_f.sort_values('annee')

        df_comb = pd.concat([df_historique_norm, df_local_f.rename(columns={'prix': 'prix'})], ignore_index=True)
        df_comb = df_comb.sort_values('annee').reset_index(drop=True)

        # Créer features et entraîner comme pour France
        df_features = creer_features_avancees(df_comb)
        results, best_model_name, train_mask, test_mask = entrainer_modeles(df_features)
        predictions_futures = predire_futur(results, best_model_name, df_features)

        best_info = results[best_model_name]

        prix_2024 = df_comb[df_comb['annee'] == 2024]['prix'].iloc[-1] if len(df_comb[df_comb['annee'] == 2024]) > 0 else df_comb.iloc[-1]['prix']
        variation_pct = ((predictions_futures[-1]['prix_pred'] - prix_2024) / prix_2024) * 100

        return {
            'type': 'local',
            'zone_name': zone_name,
            'prix_moyen_m2': float(df_local['prix_dvf'].mean()),
            'surface_moyenne': float(df_local['surface_moy'].mean()),
            'df_historique': df_historique,
            'df_dvf': df_local,
            'df_complet': df_comb,
            'annees_train': df_features[train_mask]['annee'].values,
            'y_train': df_features[train_mask]['prix'].values,
            'annees_test': df_features[test_mask]['annee'].values if test_mask.sum() > 0 else np.array([]),
            'y_test': df_features[test_mask]['prix'].values if test_mask.sum() > 0 else np.array([]),
            'best_model_name': best_model_name,
            'best_model_info': best_info,
            'results': results,
            'prix_2024': prix_2024,
            'prix_2025': predictions_futures[0]['prix_pred'],
            'prix_2026': predictions_futures[1]['prix_pred'],
            'prix_2027': predictions_futures[2]['prix_pred'],
            'variation_pct': variation_pct,
            'future_data': predictions_futures,
            'evolution': {int(row['annee']): float(row['prix']) for _, row in df_comb.iterrows()}
        }
    except Exception as e:
        print(f"Erreur analyse locale: {e}")
        import traceback
        traceback.print_exc()
        return None