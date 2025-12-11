"""
=====================================================================================
PRÉDICTION DU PRIX AU M² PAR DÉPARTEMENT - ANALYSE INTERACTIVE
Modèle ML avancé avec Train/Test/Validation
=====================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import warnings
warnings.filterwarnings('ignore')


def charger_donnees():
    """Charge et prépare les données DVF avec feature engineering"""
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
    """Crée les features avancées pour un département"""
    df_agg = df_dept.groupby('annee').agg({
        'Prixm2Moyen': 'mean',
        'SurfaceMoy': 'mean',
        'nb_mutations': 'sum',
        'NbMaisons': 'sum',
        'NbApparts': 'sum',
        'PrixMoyen': 'mean'
    }).reset_index()
    
    # Feature engineering
    df_agg['Annee_idx'] = df_agg['annee'] - df_agg['annee'].min()
    df_agg['PropMaisons'] = df_agg['NbMaisons'] / (df_agg['NbMaisons'] + df_agg['NbApparts'] + 1e-6)
    df_agg['Prix_MA3'] = df_agg['Prixm2Moyen'].rolling(window=3, min_periods=1).mean()
    df_agg['Prix_Growth'] = df_agg['Prixm2Moyen'].pct_change().fillna(0) * 100
    df_agg['Prix_Volatility'] = df_agg['Prixm2Moyen'].rolling(window=3, min_periods=1).std().fillna(0)
    df_agg['Prix_Lag1'] = df_agg['Prixm2Moyen'].shift(1).fillna(df_agg['Prixm2Moyen'].iloc[0])
    df_agg['Mutations_Growth'] = df_agg['nb_mutations'].pct_change().fillna(0) * 100
    
    return df_agg.fillna(0)


def detecter_outliers_temporels(y, threshold=3):
    """Détecte les années avec des valeurs aberrantes (chocs de marché)"""
    mean_price = np.mean(y)
    std_price = np.std(y)
    z_scores = np.abs((y - mean_price) / std_price)
    outliers_idx = np.where(z_scores > threshold)[0]
    return outliers_idx, z_scores


def calculer_metriques_avancees(y_true, y_pred, y_train_mean):
    """Calcule des métriques avancées pour évaluer la qualité du modèle"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Direction Accuracy (prédit-on correctement la hausse/baisse ?)
    if len(y_true) > 1:
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        direction_accuracy = np.mean(true_direction == pred_direction) * 100
    else:
        direction_accuracy = np.nan
    
    # Bias (le modèle sur-prédit ou sous-prédit ?)
    bias = np.mean(y_pred - y_true)
    
    # Amélioration vs modèle naïf (prédire la moyenne)
    naive_rmse = np.sqrt(mean_squared_error(y_true, np.full_like(y_true, y_train_mean)))
    improvement_vs_naive = ((naive_rmse - rmse) / naive_rmse) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'direction_accuracy': direction_accuracy,
        'bias': bias,
        'improvement_vs_naive': improvement_vs_naive,
        'naive_rmse': naive_rmse
    }


def cross_validation_temporelle(X, y, model, n_splits=3):
    """Validation croisée temporelle pour séries chronologiques"""
    if len(X) < n_splits + 2:
        return {'cv_rmse_mean': np.nan, 'cv_rmse_std': np.nan, 'cv_r2_mean': np.nan}
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_rmse_scores = []
    cv_r2_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        
        scaler_cv = StandardScaler()
        X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
        X_val_cv_scaled = scaler_cv.transform(X_val_cv)
        
        model.fit(X_train_cv_scaled, y_train_cv)
        y_pred_cv = model.predict(X_val_cv_scaled)
        
        cv_rmse_scores.append(np.sqrt(mean_squared_error(y_val_cv, y_pred_cv)))
        cv_r2_scores.append(r2_score(y_val_cv, y_pred_cv))
    
    return {
        'cv_rmse_mean': np.mean(cv_rmse_scores),
        'cv_rmse_std': np.std(cv_rmse_scores),
        'cv_r2_mean': np.mean(cv_r2_scores),
        'cv_r2_std': np.std(cv_r2_scores)
    }


def tester_stabilite_predictions(model, scaler, X_test, y_test, n_bootstrap=50):
    """Test de stabilité par bootstrap pour mesurer la variance des prédictions"""
    if len(X_test) < 2:
        return {'pred_std': np.nan, 'pred_confidence_95': np.nan}
    
    predictions = []
    for _ in range(n_bootstrap):
        # Bootstrap sampling avec remplacement
        indices = np.random.choice(len(X_test), size=len(X_test), replace=True)
        X_boot = X_test[indices]
        X_boot_scaled = scaler.transform(X_boot)
        pred_boot = model.predict(X_boot_scaled)
        predictions.append(pred_boot.mean())
    
    pred_std = np.std(predictions)
    pred_confidence_95 = 1.96 * pred_std  # Intervalle de confiance à 95%
    
    return {
        'pred_std': pred_std,
        'pred_confidence_95': pred_confidence_95
    }


def detecter_overfitting(train_r2, val_r2, test_r2, train_rmse, test_rmse, threshold_r2=0.15, threshold_rmse_ratio=1.5):
    """Détecte le niveau d'overfitting avec plusieurs critères"""
    overfitting_signals = []
    overfitting_score = 0
    
    # Critère 1 : Écart important entre train et test R²
    if not np.isnan(test_r2) and (train_r2 - test_r2) > threshold_r2:
        overfitting_signals.append(f"R² gap: {train_r2 - test_r2:.3f}")
        overfitting_score += 1
    
    # Critère 2 : RMSE test >> RMSE train
    if not np.isnan(test_rmse) and (test_rmse / train_rmse) > threshold_rmse_ratio:
        overfitting_signals.append(f"RMSE ratio: {test_rmse / train_rmse:.2f}x")
        overfitting_score += 1
    
    # Critère 3 : R² négatif sur test
    if not np.isnan(test_r2) and test_r2 < -0.5:
        overfitting_signals.append(f"R² négatif sévère: {test_r2:.3f}")
        overfitting_score += 2
    
    # Critère 4 : Validation et test divergent fortement
    if not np.isnan(val_r2) and not np.isnan(test_r2):
        if abs(val_r2 - test_r2) > 0.3:
            overfitting_signals.append(f"Val/Test divergence: {abs(val_r2 - test_r2):.3f}")
            overfitting_score += 1
    
    overfitting_level = "SÉVÈRE" if overfitting_score >= 3 else "MODÉRÉ" if overfitting_score >= 2 else "LÉGER" if overfitting_score >= 1 else "AUCUN"
    
    return {
        'overfitting_level': overfitting_level,
        'overfitting_score': overfitting_score,
        'overfitting_signals': overfitting_signals
    }


def analyser_tendance(y, annees):
    """Analyse la tendance de la série temporelle"""
    # Coefficient de variation (volatilité relative)
    cv = (np.std(y) / np.mean(y)) * 100
    
    # Croissance moyenne annuelle
    if len(y) > 1:
        total_growth = ((y[-1] / y[0]) - 1) * 100
        cagr = ((y[-1] / y[0]) ** (1 / (len(y) - 1)) - 1) * 100
    else:
        total_growth, cagr = 0, 0
    
    # Monotonie (la série est-elle régulièrement croissante/décroissante ?)
    if len(y) > 2:
        diffs = np.diff(y)
        monotonie_score = abs(np.sum(diffs > 0) - np.sum(diffs < 0)) / len(diffs) * 100
    else:
        monotonie_score = 0
    
    return {
        'coefficient_variation': cv,
        'total_growth_pct': total_growth,
        'cagr_pct': cagr,
        'monotonie_score': monotonie_score,
        'volatilite': 'HAUTE' if cv > 20 else 'MOYENNE' if cv > 10 else 'FAIBLE'
    }


def predire_departement(df_clean, code_departement):
    """
    Prédiction ML avancée pour un département avec Train/Test/Validation
    """
    # Filtrer le département
    df_dept = df_clean[df_clean['Departement'] == code_departement]
    
    if len(df_dept) < 5:
        return None
    
    # Créer les features
    df_agg = creer_features_departement(df_dept)
    
    if len(df_agg) < 5:
        return None
    
    # Features et cible
    feature_columns = ['Annee_idx', 'SurfaceMoy', 'nb_mutations', 'PropMaisons', 
                       'Prix_MA3', 'Prix_Growth', 'Prix_Volatility', 'Prix_Lag1', 'Mutations_Growth']
    
    X = df_agg[feature_columns].values
    y = df_agg['Prixm2Moyen'].values
    annees = df_agg['annee'].values
    
    # Split temporel amélioré (minimum 2 ans par set)
    n_years = len(df_agg)
    
    # Stratégie adaptative selon la taille du dataset
    if n_years < 6:
        # Très petit dataset : pas de validation, seulement train/test
        train_size = max(int(n_years * 0.75), n_years - 2)
        val_size = 0
        test_size = n_years - train_size
    elif n_years < 10:
        # Petit dataset : minimum 2 ans pour validation et test
        test_size = 2
        val_size = 2
        train_size = n_years - val_size - test_size
    else:
        # Dataset normal : 70% train, 15% validation, 15% test (minimum 2 ans chacun)
        test_size = max(int(n_years * 0.15), 2)
        val_size = max(int(n_years * 0.15), 2)
        train_size = n_years - val_size - test_size
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    annees_train = annees[:train_size]
    annees_val = annees[train_size:train_size+val_size]
    annees_test = annees[train_size+val_size:]
    
    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if len(X_val) > 0 else np.array([])
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.array([])
    
    # Analyser la tendance des données
    tendance_info = analyser_tendance(y, annees)
    
    # Détecter les outliers
    outliers_idx, z_scores = detecter_outliers_temporels(y)
    
    # Entraîner plusieurs modèles avec hyperparamètres optimisés selon la tendance
    # Si haute volatilité, favoriser la régularisation forte
    alpha_ridge = 10.0 if tendance_info['volatilite'] == 'HAUTE' else 1.0
    alpha_lasso = 1.0 if tendance_info['volatilite'] == 'HAUTE' else 0.1
    
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=alpha_ridge),
        'Lasso': Lasso(alpha=alpha_lasso, max_iter=10000),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    }
    
    y_train_mean = np.mean(y_train)
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        
        y_train_pred = model.predict(X_train_scaled)
        train_metrics = calculer_metriques_avancees(y_train, y_train_pred, y_train_mean)
        
        if len(X_val_scaled) > 0:
            y_val_pred = model.predict(X_val_scaled)
            val_metrics = calculer_metriques_avancees(y_val, y_val_pred, y_train_mean)
        else:
            y_val_pred = np.array([])
            val_metrics = {k: np.nan for k in ['rmse', 'mae', 'r2', 'mape', 'direction_accuracy', 'bias', 'improvement_vs_naive', 'naive_rmse']}
        
        if len(X_test_scaled) > 0:
            y_test_pred = model.predict(X_test_scaled)
            test_metrics = calculer_metriques_avancees(y_test, y_test_pred, y_train_mean)
            
            # Tests de stabilité
            stabilite = tester_stabilite_predictions(model, scaler, X_test, y_test)
        else:
            y_test_pred = np.array([])
            test_metrics = {k: np.nan for k in ['rmse', 'mae', 'r2', 'mape', 'direction_accuracy', 'bias', 'improvement_vs_naive', 'naive_rmse']}
            stabilite = {'pred_std': np.nan, 'pred_confidence_95': np.nan}
        
        # Validation croisée temporelle
        cv_results = cross_validation_temporelle(X_train, y_train, model, n_splits=min(3, len(X_train) // 2))
        
        # Détection d'overfitting
        overfitting_info = detecter_overfitting(
            train_metrics['r2'], 
            val_metrics['r2'], 
            test_metrics['r2'],
            train_metrics['rmse'],
            test_metrics['rmse']
        )
        
        results[name] = {
            'model': model,
            'train_rmse': train_metrics['rmse'],
            'train_r2': train_metrics['r2'],
            'train_mae': train_metrics['mae'],
            'train_mape': train_metrics['mape'],
            'val_rmse': val_metrics['rmse'],
            'val_r2': val_metrics['r2'],
            'val_mae': val_metrics['mae'],
            'test_rmse': test_metrics['rmse'],
            'test_r2': test_metrics['r2'],
            'test_mae': test_metrics['mae'],
            'test_mape': test_metrics['mape'],
            'test_direction_accuracy': test_metrics['direction_accuracy'],
            'test_bias': test_metrics['bias'],
            'improvement_vs_naive': test_metrics['improvement_vs_naive'],
            'y_train_pred': y_train_pred,
            'y_val_pred': y_val_pred,
            'y_test_pred': y_test_pred,
            'cv_rmse_mean': cv_results['cv_rmse_mean'],
            'cv_rmse_std': cv_results['cv_rmse_std'],
            'cv_r2_mean': cv_results['cv_r2_mean'],
            'stabilite': stabilite,
            'overfitting': overfitting_info
        }
    
    # Sélection intelligente du meilleur modèle avec score composite
    # Critères : RMSE test + Overfitting + Stabilité + Direction Accuracy
    model_scores = {}
    
    for name, res in results.items():
        score = 0
        
        # Critère 1 : RMSE test (poids 40%)
        if not np.isnan(res['test_rmse']):
            # Normaliser RMSE (plus petit = meilleur)
            rmse_scores = [results[k]['test_rmse'] for k in results.keys() if not np.isnan(results[k]['test_rmse'])]
            if len(rmse_scores) > 0:
                rmse_normalized = 1 - (res['test_rmse'] - min(rmse_scores)) / (max(rmse_scores) - min(rmse_scores) + 1e-10)
                score += rmse_normalized * 40
        
        # Critère 2 : Pas d'overfitting (poids 30%)
        overfitting_penalty = res['overfitting']['overfitting_score'] * 7.5
        score -= overfitting_penalty
        
        # Critère 3 : Direction Accuracy (poids 15%)
        if not np.isnan(res['test_direction_accuracy']):
            score += (res['test_direction_accuracy'] / 100) * 15
        
        # Critère 4 : Amélioration vs modèle naïf (poids 10%)
        if not np.isnan(res['improvement_vs_naive']):
            score += max(0, min(res['improvement_vs_naive'], 100) / 100) * 10
        
        # Critère 5 : Stabilité des prédictions (poids 5%)
        if not np.isnan(res['stabilite']['pred_std']):
            # Moins de variance = mieux
            std_scores = [results[k]['stabilite']['pred_std'] for k in results.keys() if not np.isnan(results[k]['stabilite']['pred_std'])]
            if len(std_scores) > 0 and max(std_scores) > 0:
                stability_score = 1 - (res['stabilite']['pred_std'] / max(std_scores))
                score += stability_score * 5
        
        model_scores[name] = score
    
    # Sélectionner le modèle avec le meilleur score composite
    if model_scores:
        best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k])
    else:
        # Fallback sur RMSE test si le scoring échoue
        if all(not np.isnan(results[k]['test_rmse']) for k in results.keys()):
            best_model_name = min(results.keys(), key=lambda k: results[k]['test_rmse'])
        else:
            best_model_name = min(results.keys(), key=lambda k: results[k]['train_rmse'])
    
    best_model_info = results[best_model_name]
    
    # Prédictions futures (2025-2027)
    future_data = []
    last_row = df_agg.iloc[-1]
    
    for i, year in enumerate([2025, 2026, 2027]):
        if i == 0:
            prev_prix = df_agg.iloc[-1]['Prixm2Moyen']
            prev_prix_lag1 = df_agg.iloc[-1]['Prixm2Moyen']
            prev_prix_lag2 = df_agg.iloc[-2]['Prixm2Moyen'] if len(df_agg) > 1 else prev_prix
        else:
            prev_prix = future_data[i-1]['prix_pred']
            prev_prix_lag1 = prev_prix
            prev_prix_lag2 = df_agg.iloc[-1]['Prixm2Moyen'] if i == 1 else future_data[i-2]['prix_pred']
        
        annee_idx = year - df_agg['annee'].min()
        
        future_features = np.array([[
            annee_idx,
            last_row['SurfaceMoy'],
            last_row['nb_mutations'],
            last_row['PropMaisons'],
            prev_prix,
            0,
            df_agg['Prix_Volatility'].iloc[-1],
            prev_prix_lag1,
            0
        ]])
        
        future_features_scaled = scaler.transform(future_features)
        prix_pred = best_model_info['model'].predict(future_features_scaled)[0]
        
        future_data.append({'annee': year, 'prix_pred': prix_pred})
    
    prix_2024 = df_agg.iloc[-1]['Prixm2Moyen']
    variation_pct = ((future_data[-1]['prix_pred'] - prix_2024) / prix_2024) * 100
    
    # Calculer intervalle de confiance pour les prédictions futures
    confidence_interval = best_model_info['stabilite']['pred_confidence_95']
    
    return {
        'departement': code_departement,
        'best_model_name': best_model_name,
        'model_scores': model_scores,
        'results': results,
        'annees': annees,
        'y': y,
        'annees_train': annees_train,
        'y_train': y_train,
        'annees_val': annees_val,
        'y_val': y_val,
        'annees_test': annees_test,
        'y_test': y_test,
        'best_model_info': best_model_info,
        'prix_2024': prix_2024,
        'prix_2025': future_data[0]['prix_pred'],
        'prix_2026': future_data[1]['prix_pred'],
        'prix_2027': future_data[2]['prix_pred'],
        'variation_pct': variation_pct,
        'future_data': future_data,
        'nb_annees': len(df_agg),
        'tendance_info': tendance_info,
        'outliers_idx': outliers_idx,
        'confidence_interval': confidence_interval
    }


def afficher_resultats(resultat):
    """Affiche les résultats de la prédiction"""
    if resultat is None:
        print("❌ Données insuffisantes (minimum 5 ans requis)\n")
        return
    
    dept = resultat['departement']
    best = resultat['best_model_info']
    
    print("="*100)
    print(f"📊 PRÉDICTION POUR LE DÉPARTEMENT {dept}")
    print("="*100)
    
    print(f"\n🧠 MEILLEUR MODÈLE : {resultat['best_model_name']}")
    print(f"  • Score composite       : {resultat['model_scores'][resultat['best_model_name']]:.2f}/100")
    print(f"  • RMSE sur train        : {best['train_rmse']:.2f} €/m²")
    print(f"  • R² sur train          : {best['train_r2']:.4f}")
    
    if not np.isnan(best['test_rmse']):
        print(f"  • RMSE sur test         : {best['test_rmse']:.2f} €/m²")
        print(f"  • MAE sur test          : {best['test_mae']:.2f} €/m²")
        print(f"  • R² sur test           : {best['test_r2']:.4f} {'✅' if best['test_r2'] >= 0.85 else '⚠️'}")
        
        if not np.isnan(best['test_direction_accuracy']):
            dir_emoji = "🎯" if best['test_direction_accuracy'] >= 70 else "⚠️"
            print(f"  • Direction Accuracy    : {best['test_direction_accuracy']:.1f}% {dir_emoji}")
        
        if not np.isnan(best['improvement_vs_naive']):
            print(f"  • Amélioration vs Naïf  : {best['improvement_vs_naive']:.1f}%")
        
        print(f"\n  🔬 DIAGNOSTIC D'OVERFITTING :")
        print(f"     Niveau : {best['overfitting']['overfitting_level']}")
        if best['overfitting']['overfitting_signals']:
            for signal in best['overfitting']['overfitting_signals']:
                print(f"     ⚠️  {signal}")
    
    if not np.isnan(best['cv_rmse_mean']):
        print(f"\n  📊 VALIDATION CROISÉE TEMPORELLE :")
        print(f"     RMSE moyen : {best['cv_rmse_mean']:.2f} ± {best['cv_rmse_std']:.2f} €/m²")
        print(f"     R² moyen   : {best['cv_r2_mean']:.4f}")
    
    print(f"\n📊 COMPARAISON DES 5 MODÈLES :")
    print(f"{'Modèle':<20} {'Score':<10} {'Test RMSE':<12} {'Test R²':<12} {'Overfitting':<15}")
    print("-"*70)
    for name, res in resultat['results'].items():
        score_str = f"{resultat['model_scores'].get(name, 0):.1f}" if resultat['model_scores'] else "N/A"
        rmse_str = f"{res['test_rmse']:.2f}" if not np.isnan(res['test_rmse']) else "N/A"
        r2_str = f"{res['test_r2']:.4f}" if not np.isnan(res['test_r2']) else "N/A"
        ovf_str = res['overfitting']['overfitting_level']
        marker = "🏆" if name == resultat['best_model_name'] else "  "
        print(f"{marker} {name:<18} {score_str:<10} {rmse_str:<12} {r2_str:<12} {ovf_str:<15}")
    
    print(f"\n💰 PRÉDICTIONS DU PRIX AU M² (avec intervalle de confiance à 95%) :")
    ci = resultat['confidence_interval']
    print(f"  • 2024 (référence)      : {resultat['prix_2024']:,.0f} €/m²")
    
    if not np.isnan(ci):
        print(f"  • 2025                  : {resultat['prix_2025']:,.0f} €/m² [±{ci:.0f}]")
        print(f"  • 2026                  : {resultat['prix_2026']:,.0f} €/m² [±{ci:.0f}]")
        print(f"  • 2027                  : {resultat['prix_2027']:,.0f} €/m² [±{ci:.0f}]")
    else:
        print(f"  • 2025                  : {resultat['prix_2025']:,.0f} €/m²")
        print(f"  • 2026                  : {resultat['prix_2026']:,.0f} €/m²")
        print(f"  • 2027                  : {resultat['prix_2027']:,.0f} €/m²")
    
    variation = resultat['variation_pct']
    symbole = "📈" if variation > 2 else "📉" if variation < -2 else "→"
    print(f"\n{symbole} TENDANCE 2024-2027 : {variation:+.2f}%")
    
    # Afficher l'analyse de tendance
    tendance = resultat['tendance_info']
    print(f"\n📈 ANALYSE DE LA SÉRIE TEMPORELLE :")
    print(f"  • Volatilité            : {tendance['volatilite']} (CV={tendance['coefficient_variation']:.1f}%)")
    print(f"  • Croissance annuelle   : {tendance['cagr_pct']:.2f}% CAGR")
    print(f"  • Croissance totale     : {tendance['total_growth_pct']:.1f}%")
    
    if len(resultat['outliers_idx']) > 0:
        print(f"  • ⚠️ Années aberrantes   : {len(resultat['outliers_idx'])} détectée(s)")
    
    print("="*100 + "\n")


def visualiser_prediction(resultat):
    """Visualisation avancée avec comparaison des modèles"""
    if resultat is None:
        return
    
    plt.figure(figsize=(16, 10))
    
    # Graphique 1 : Historique + Prédictions
    plt.subplot(2, 2, 1)
    

    # Données réelles (lignes solides avec marqueurs)
    plt.plot(resultat['annees_train'], resultat['y_train'], 'o-', color='#2E86AB', linewidth=2.5, markersize=8, label='Train (données réelles)', alpha=0.9)
    if len(resultat['annees_val']) > 0:
        plt.plot(resultat['annees_val'], resultat['y_val'], 's-', color='#A23B72', linewidth=2.5, markersize=8, label='Validation (données réelles)', alpha=0.9)
    if len(resultat['annees_test']) > 0:
        plt.plot(resultat['annees_test'], resultat['y_test'], '^-', color='#F18F01', linewidth=2.5, markersize=8, label='Test (données réelles)', alpha=0.9)
    
    # Prédictions du modèle (lignes pointillées, mêmes couleurs)
    best = resultat['best_model_info']
    plt.plot(resultat['annees_train'], best['y_train_pred'], '--', color='#2E86AB', linewidth=2, label='Train (prédictions)', alpha=0.7)
    if len(best['y_val_pred']) > 0:
        plt.plot(resultat['annees_val'], best['y_val_pred'], '--', color='#A23B72', linewidth=2, label='Validation (prédictions)', alpha=0.7)
    if len(best['y_test_pred']) > 0:
        plt.plot(resultat['annees_test'], best['y_test_pred'], '--', color='#F18F01', linewidth=2, label='Test (prédictions)', alpha=0.7)
    
    # Prédictions futures (ligne pointillée rouge avec marqueurs diamants)
    future_years = [d['annee'] for d in resultat['future_data']]
    future_prices = [d['prix_pred'] for d in resultat['future_data']]
    plt.plot(future_years, future_prices, 'D--', color='#C73E1D', linewidth=2.5, markersize=10, label='Futur 2025-2027', alpha=0.9)
    
    plt.axvline(x=2024.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.title(f"Département {resultat['departement']} - {resultat['best_model_name']}", fontsize=14, fontweight='bold')
    plt.xlabel('Année', fontsize=12)
    plt.ylabel('Prix au m² (€)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc='best')
    
    # Graphique 2 : Résidus
    if len(resultat['annees_test']) > 0:
        plt.subplot(2, 2, 2)
        residuals = resultat['y_test'] - best['y_test_pred']
        plt.bar(resultat['annees_test'], residuals, color=['green' if r > 0 else 'red' for r in residuals], alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.title('Résidus sur Test Set', fontsize=14, fontweight='bold')
        plt.xlabel('Année', fontsize=12)
        plt.ylabel('Résidu (€/m²)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
    
    # Graphique 3 : Comparaison RMSE
    plt.subplot(2, 2, 3)
    model_names = list(resultat['results'].keys())
    test_rmses = [resultat['results'][name]['test_rmse'] for name in model_names]
    colors = ['green' if name == resultat['best_model_name'] else 'lightblue' for name in model_names]
    plt.barh(model_names, test_rmses, color=colors, alpha=0.8)
    plt.xlabel('RMSE sur Test (€/m²)', fontsize=12)
    plt.title('Comparaison des Modèles (RMSE)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Graphique 4 : Comparaison R² (CORRIGÉ - sans limite xlim)
    plt.subplot(2, 2, 4)
    test_r2s = [resultat['results'][name]['test_r2'] for name in model_names]
    colors = ['green' if name == resultat['best_model_name'] else 'lightcoral' for name in model_names]
    plt.barh(model_names, test_r2s, color=colors, alpha=0.8)
    plt.xlabel('R² sur Test', fontsize=12)
    plt.title('Comparaison des Modèles (R²)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    # ⚠️ SUPPRESSION de plt.xlim(0, 1) pour afficher les R² négatifs
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)  # Ligne de référence à 0
    
    plt.tight_layout()
    plt.show()


def main():
    """Programme principal"""
    print("\n" + "="*100)
    print("🤖 PRÉDICTION DU PRIX AU M² PAR DÉPARTEMENT - MODÈLE ML AVANCÉ (2025-2027)")
    print("="*100 + "\n")
    
    # Charger les données
    df_clean = charger_donnees()
    
    # Demander le code postal
    code_postal = input("📍 Entrez un code postal (ex: 75001, 69001, 33000) : ").strip()
    
    # Extraire le département (2 premiers chiffres)
    if len(code_postal) < 2:
        print("❌ Code postal invalide (minimum 2 chiffres)\n")
        return
    
    code_dept = code_postal[:2]
    print(f"\n🔍 Analyse du département {code_dept}...\n")
    
    # Vérifier que le département existe
    if code_dept not in df_clean['Departement'].values:
        print(f"❌ Département {code_dept} non trouvé dans la base\n")
        depts_dispo = sorted(df_clean['Departement'].unique())
        print(f"Départements disponibles ({len(depts_dispo)}) : {', '.join(depts_dispo[:20])}...")
        return
    
    print("⚙️ Feature engineering en cours...")
    print("🧠 Entraînement de 5 modèles ML (LinearRegression, Ridge, Lasso, RandomForest, GradientBoosting)...")
    print("🔀 Split temporel : Train / Validation / Test\n")
    
    # Effectuer la prédiction avec modèle ML avancé
    resultat = predire_departement(df_clean, code_dept)
    
    # Afficher les résultats
    afficher_resultats(resultat)
    
    # Visualiser
    if resultat is not None:
        visualiser_prediction(resultat)
        
        print("\n" + "="*100)
        print("📋 RÉSUMÉ DE L'ANALYSE")
        print("="*100)
        print(f"\n🎯 Département : {code_dept}")
        print(f"📊 Années de données : {resultat['nb_annees']}")
        print(f"🧠 Meilleur modèle : {resultat['best_model_name']}")
        print(f"📈 Tendance 2024-2027 : {resultat['variation_pct']:+.2f}%")
        print("\n⚠️ LIMITES :")
        print("  • Prédictions basées sur la continuité des tendances historiques")
        print("  • Ne prend pas en compte les chocs économiques majeurs")
        print("  • Utiliser comme ordre de grandeur, pas comme certitude absolue")
        print("  • Un R² négatif signifie que le modèle performe moins bien qu'une simple moyenne")
        print("    (souvent dû à un test set trop petit ou à un overfitting sévère)")
        print("\n" + "="*100)
        print("✅ ANALYSE TERMINÉE")
        print("="*100 + "\n")


if __name__ == "__main__":
    main()
