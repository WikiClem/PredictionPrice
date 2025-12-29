"""
=====================================================================================
PRÉDICTION DU PRIX AU M² PAR DÉPARTEMENT - ANALYSE INTERACTIVE
Modèle ML avancé avec Train/Test/Validation
=====================================================================================
"""

import pandas as pd  # Manipulation de données tabulaires (DataFrames)
import numpy as np  # Calculs numériques et matrices
import matplotlib.pyplot as plt  # Création de graphiques
from sklearn.model_selection import train_test_split  # Division train/test
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # Modèles de régression
from sklearn.preprocessing import StandardScaler  # Normalisation des données
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Métriques d'évaluation
from sklearn.model_selection import TimeSeriesSplit, cross_val_score  # Validation temporelle
import warnings  # Gestion des messages d'avertissement
warnings.filterwarnings('ignore')  # Supprime les warnings pour un affichage plus propre


def charger_donnees():  # Fonction principale de chargement des données
    """Charge et prépare les données DVF avec feature engineering"""
    print("📂 Chargement et nettoyage des données...\n")
    
    # 🔍 ÉTAPE 1 : LECTURE DU FICHIER CSV
    # pd.read_csv() = fonction pandas pour lire un fichier CSV
    # Exemple : si le fichier contient "INSEE_COM,annee,Prixm2Moyen\n75001,2020,8500\n"
    # Résultat : DataFrame avec colonnes [INSEE_COM, annee, Prixm2Moyen]
    df = pd.read_csv(r'C:\Users\Clement\OneDrive\Bureau\data imo\code\dvf.csv',  # Lecture du fichier CSV
                     sep=',',              # Séparateur = virgule (CSV standard)
                     skipinitialspace=True)  # Ignore les espaces après les virgules
    
    # 🧹 ÉTAPE 2 : NETTOYAGE DES NOMS DE COLONNES
    # .str.strip() = enlève les espaces en début/fin de chaque nom de colonne
    # Exemple : [' INSEE_COM ', 'annee '] → ['INSEE_COM', 'annee']
    df.columns = df.columns.str.strip()  # Supprime espaces début/fin des noms colonnes
    
    # 🏷️ ÉTAPE 3 : CRÉATION DE LA COLONNE DÉPARTEMENT
    # INSEE_COM = code commune (ex: 75001, 69123, 13055)
    # .astype(str) = convertit en texte (au cas où ce serait des nombres)
    # .str[:2] = prend les 2 premiers caractères
    # Exemples : 75001 → '75', 69123 → '69', 13055 → '13'
    df['Departement'] = df['INSEE_COM'].astype(str).str[:2]  # Extrait département des 2 premiers chiffres
    
    # 🚿 ÉTAPE 4 : NETTOYAGE DES DONNÉES (CRUCIAL POUR LA QUALITÉ DU MODÈLE)
    
    # 4.1 : SUPPRESSION DES LIGNES AVEC VALEURS MANQUANTES
    # .dropna() = supprime les lignes où au moins une des colonnes spécifiées est NaN
    # subset = liste des colonnes obligatoires
    # Exemple : ligne avec annee=2020, Prixm2Moyen=NaN → SUPPRIMÉE
    # .copy() = crée une copie indépendante (évite les warnings pandas)
    df_clean = df.dropna(subset=['annee', 'Prixm2Moyen', 'nb_mutations']).copy()
    
    # 4.2 : FILTRAGE DES PRIX ABERRANTS (OUTLIERS)
    # Logique : prix < 100€/m² = probablement erreur de saisie
    #          prix > 20000€/m² = probablement erreur de saisie ou exception
    # & = ET logique en pandas
    # Exemples : 50€/m² → SUPPRIMÉ (trop faible), 25000€/m² → SUPPRIMÉ (trop élevé)
    df_clean = df_clean[(df_clean['Prixm2Moyen'] > 100) & (df_clean['Prixm2Moyen'] < 20000)]
    
    # 4.3 : SUPPRESSION DES ANNÉES SANS ACTIVITÉ
    # nb_mutations = 0 = aucune vente cette année-là = pas d'info utile
    # Exemple : ligne avec nb_mutations=0 → SUPPRIMÉE
    df_clean = df_clean[df_clean['nb_mutations'] > 0]
    
    print(f"  ✓ Lignes nettoyées : {len(df_clean):,}")  # Affiche nombre de lignes après nettoyage avec séparateurs milliers
    print(f"  ✓ Départements disponibles : {df_clean['Departement'].nunique()}\n")  # Compte départements uniques disponibles
    
    return df_clean  # Retourne DataFrame nettoyé prêt pour analyse


def creer_features_departement(df_dept):  # Fonction de transformation des données brutes en variables ML
    """🏗️ COURS : FEATURE ENGINEERING POUR L'IMMOBILIER
    
    Cette fonction transforme les données brutes en variables utilisables par le ML.
    Feature Engineering = art de créer de nouvelles variables à partir des données existantes
    pour améliorer la performance des modèles de Machine Learning.
    """
    
    # 📊 ÉTAPE 1 : AGRÉGATION PAR ANNÉE
    # Problème : on a plusieurs lignes par département/année (différentes communes)
    # Solution : grouper par année et calculer des statistiques
    # 
    # .groupby('annee') = groupe toutes les lignes avec la même année
    # .agg({}) = applique des fonctions d'agrégation à chaque groupe
    # 
    # Exemple de transformation :
    # AVANT :
    #   annee | commune | Prixm2Moyen | nb_mutations
    #   2020  | 75001   | 8500        | 120
    #   2020  | 75002   | 9000        | 85
    #   2020  | 75003   | 8800        | 95
    # 
    # APRÈS :
    #   annee | Prixm2Moyen | nb_mutations
    #   2020  | 8766.7      | 300
    #
    df_agg = df_dept.groupby('annee').agg({  # Groupe par année et calcule stats
        'Prixm2Moyen': 'mean',    # Prix moyen au m² cette année-là
        'SurfaceMoy': 'mean',     # Surface moyenne des logements
        'nb_mutations': 'sum',    # TOTAL des ventes dans le département
        'NbMaisons': 'sum',       # TOTAL maisons vendues
        'NbApparts': 'sum',       # TOTAL appartements vendus
        'PrixMoyen': 'mean'       # Prix moyen (pas au m²)
    }).reset_index()  # reset_index() = transforme l'index (annee) en colonne normale
    
    if len(df_agg) < 3:  # Vérifie si assez d'années (minimum 3 pour ML décent)
        return None  # Retourne None si pas assez de données
    
    # 🧠 ÉTAPE 2 : FEATURE ENGINEERING AVANCÉ
    # Création de nouvelles variables plus intelligentes que les données brutes
    
    # 2.1 : NORMALISATION DE L'ANNÉE (Min-Max Scaling)
    # 🎯 OBJECTIF : transformer les années en valeurs entre 0 et 1
    # Pourquoi ? Les années (2014, 2015, 2016...) sont de gros nombres
    # Le ML préfère des valeurs petites et normalisées
    #
    # FORMULE : (valeur - minimum) / (maximum - minimum)
    # + 1e-6 = évite la division par 0 si toutes les années sont identiques
    #
    # 📚 EXEMPLE CONCRET :
    # Si on a les années [2020, 2021, 2022, 2023, 2024]
    # min = 2020, max = 2024
    # 2020 → (2020-2020)/(2024-2020) = 0/4 = 0.0
    # 2021 → (2021-2020)/(2024-2020) = 1/4 = 0.25
    # 2022 → (2022-2020)/(2024-2020) = 2/4 = 0.5
    # 2023 → (2023-2020)/(2024-2020) = 3/4 = 0.75
    # 2024 → (2024-2020)/(2024-2020) = 4/4 = 1.0
    df_agg['Annee_idx'] = (df_agg['annee'] - df_agg['annee'].min()) / (df_agg['annee'].max() - df_agg['annee'].min() + 1e-6)  # Normalise années entre 0 et 1
    
    # 2.2 : PROPORTION DE MAISONS (Feature Ratio)
    # 🎯 OBJECTIF : quelle part du marché représentent les maisons ?
    # Logique économique : maisons = terrain inclus = plus cher au m²
    #                     appartements = juste construction = moins cher au m²
    #
    # FORMULE : NbMaisons / (NbMaisons + NbApparts)
    # + 1e-6 = évite division par 0 si aucune vente
    #
    # 📚 EXEMPLE CONCRET :
    # Si NbMaisons = 300 et NbApparts = 700
    # PropMaisons = 300/(300+700) = 300/1000 = 0.3 = 30% de maisons
    # Si PropMaisons = 0.8 → marché dominé par les maisons → prix/m² plus élevé
    # Si PropMaisons = 0.2 → marché dominé par les appartements → prix/m² plus faible
    df_agg['PropMaisons'] = df_agg['NbMaisons'] / (df_agg['NbMaisons'] + df_agg['NbApparts'] + 1e-6)  # Calcule ratio maisons/(maisons+apparts)
    
    # 📈 ÉTAPE 3 : FEATURES TEMPORELLES (Time Series Features)
    # Ces variables capturent l'évolution dans le temps
    
    # 3.1 : MOYENNE MOBILE SUR 3 ANS (Moving Average)
    # 🎯 OBJECTIF : lisser les fluctuations et capturer la tendance
    # .rolling(window=3) = fenêtre glissante de 3 valeurs
    # min_periods=1 = calcule même avec moins de 3 valeurs au début
    #
    # ⚠️ ATTENTION : Cette feature est PROBLÉMATIQUE (comme tu l'as remarqué !)
    # On utilise le prix passé pour prédire le prix futur = quasi-circulaire
    #
    # 📚 EXEMPLE CONCRET :
    # Prix par année : [1000, 1100, 1200, 1050, 1300]
    # Prix_MA3 :      [1000, 1050, 1100, 1117, 1183]
    # Calcul pour 2023 (1117) = (1100+1200+1050)/3 = 3350/3 = 1117
    df_agg['Prix_MA3'] = df_agg['Prixm2Moyen'].rolling(window=min(3, len(df_agg)), min_periods=1).mean()  # Moyenne mobile 3 ans pour lisser tendance
    
    # 3.2 : ÉCART À LA TENDANCE (Trend Deviation)
    # 🎯 OBJECTIF : mesurer si le prix actuel est au-dessus/en-dessous de la tendance
    # FORMULE : (Prix_Actuel / Prix_Tendance) - 1
    #
    # ⚠️ ATTENTION : Encore PLUS problématique que Prix_MA3 !
    # On divise le prix par sa propre moyenne = très circulaire
    #
    # 📚 EXEMPLE CONCRET :
    # Si Prix_Actuel = 1200 et Prix_MA3 = 1100
    # Prix_Trend = (1200/1100) - 1 = 1.091 - 1 = 0.091 = +9.1% au-dessus de la tendance
    # Si Prix_Trend > 0 → prix au-dessus de la moyenne récente
    # Si Prix_Trend < 0 → prix en-dessous de la moyenne récente
    df_agg['Prix_Trend'] = (df_agg['Prixm2Moyen'] / df_agg['Prix_MA3']) - 1  # Écart relatif par rapport à moyenne mobile
    
    # 3.3 : PRIX DE L'ANNÉE PRÉCÉDENTE (Lag Feature)
    # 🎯 OBJECTIF : utiliser le prix de l'année N-1 pour prédire l'année N
    # .shift(1) = décale toutes les valeurs d'une position vers le bas
    #
    # 📚 EXEMPLE CONCRET :
    # Prix original : [1000, 1100, 1200, 1050, 1300]
    # Prix_Lag1 :    [NaN,  1000, 1100, 1200, 1050]
    # Pour prédire 2023, on utilise le prix de 2022
    df_agg['Prix_Lag1'] = df_agg['Prixm2Moyen'].shift(1)  # Prix année précédente (décalage temporel)
    
    # 📊 ÉTAPE 4 : TRANSFORMATION LOGARITHMIQUE
    # 🎯 OBJECTIF : gérer les valeurs extrêmes dans le nombre de mutations
    #
    # PROBLÈME : nb_mutations peut varier énormément
    # Exemple : Paris = 50000 mutations, Lozère = 100 mutations
    # Le ML a du mal avec ces écarts de 500x
    #
    # SOLUTION : Transformation logarithmique
    # np.log1p(x) = ln(x + 1) (le +1 évite ln(0) = -∞)
    #
    # 📚 COURS MATHS : POURQUOI LE LOGARITHME ?
    # Le log compresse les grandes valeurs et étend les petites
    # log(100) = 4.6, log(1000) = 6.9, log(10000) = 9.2
    # Écart original : 10000/100 = 100x
    # Écart après log : 9.2/4.6 = 2x seulement !
    #
    # 📚 EXEMPLE CONCRET :
    # nb_mutations = [100, 500, 1000, 5000]
    # Mutations_Log = [4.6, 6.2, 6.9, 8.5]
    # Les écarts sont maintenant raisonnables pour le ML
    df_agg['Mutations_Log'] = np.log1p(df_agg['nb_mutations'])  # Transformation log pour normaliser grandes valeurs
    
    # 🔧 ÉTAPE 5 : GESTION DES VALEURS MANQUANTES (CRUCIAL !)
    # Problème : .shift(1) crée des NaN, certains calculs peuvent échouer
    #
    # STRATÉGIE INTELLIGENTE (pas juste fillna(0)) :
    # 1. .interpolate() = remplit les NaN par interpolation linéaire
    # 2. .bfill() = "backward fill" = remplit avec la valeur suivante
    # 3. .ffill() = "forward fill" = remplit avec la valeur précédente
    #
    # 📚 EXEMPLE CONCRET :
    # Valeurs : [100, NaN, NaN, 200]
    # Après interpolation : [100, 133.3, 166.7, 200]
    # Beaucoup plus intelligent que de mettre 0 !
    df_agg = df_agg.interpolate(method='linear').bfill().ffill()  # Remplit NaN intelligemment par interpolation
    
    return df_agg  # Retourne DataFrame avec features engineerées


def detecter_outliers_temporels(y, threshold=3):  # Fonction pour identifier années aberrantes (crises, bulles)
    """Détecte les années avec des valeurs aberrantes (chocs de marché)"""
    mean_price = np.mean(y)  # Calcule prix moyen sur toute la période
    std_price = np.std(y)   # Calcule écart-type des prix
    z_scores = np.abs((y - mean_price) / std_price)  # Z-score = distance en écarts-types de la moyenne
    outliers_idx = np.where(z_scores > threshold)[0]  # Indices des années avec |z-score| > 3
    return outliers_idx, z_scores  # Retourne positions aberrantes et tous les z-scores


def calculer_metriques_avancees(y_true, y_pred, y_train_mean):  # Calcule métriques complètes d'évaluation ML
    """Calcule des métriques avancées pour évaluer la qualité du modèle"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Racine de l'erreur quadratique moyenne
    mae = mean_absolute_error(y_true, y_pred)  # Erreur absolue moyenne
    r2 = r2_score(y_true, y_pred)  # Coefficient de détermination (% variance expliquée)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Erreur moyenne en pourcentage
    
    # Direction Accuracy (prédit-on correctement la hausse/baisse ?)
    if len(y_true) > 1:  # Si plus d'une observation
        true_direction = np.diff(y_true) > 0  # Vraie direction : hausse=True, baisse=False
        pred_direction = np.diff(y_pred) > 0  # Direction prédite : hausse=True, baisse=False
        direction_accuracy = np.mean(true_direction == pred_direction) * 100  # % directions correctes
    else:
        direction_accuracy = np.nan  # Pas de direction calculable avec 1 seule valeur
    
    # Bias (le modèle sur-prédit ou sous-prédit ?)
    bias = np.mean(y_pred - y_true)  # Biais moyen (>0 = sur-prédiction, <0 = sous-prédiction)
    
    # Amélioration vs modèle naïf (prédire la moyenne)
    naive_rmse = np.sqrt(mean_squared_error(y_true, np.full_like(y_true, y_train_mean)))  # RMSE si on prédisait toujours la moyenne
    improvement_vs_naive = ((naive_rmse - rmse) / naive_rmse) * 100  # % d'amélioration vs prédiction naïve
    
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


def cross_validation_temporelle(X, y, model, n_splits=3):  # Validation croisée respectant ordre temporel
    """Validation croisée temporelle pour séries chronologiques"""
    if len(X) < n_splits + 2:  # Vérifie si assez de données pour diviser
        return {'cv_rmse_mean': np.nan, 'cv_rmse_std': np.nan, 'cv_r2_mean': np.nan}  # Retourne NaN si pas assez
    
    tscv = TimeSeriesSplit(n_splits=n_splits)  # Crée splitter temporel (pas aléatoire)
    cv_rmse_scores = []  # Liste pour stocker RMSE de chaque fold
    cv_r2_scores = []   # Liste pour stocker R² de chaque fold
    
    for train_idx, val_idx in tscv.split(X):  # Boucle sur chaque fold temporel
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]  # Sépare train/validation pour ce fold
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]  # Sépare targets train/validation
        
        scaler_cv = StandardScaler()  # Nouveau scaler pour ce fold
        X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)  # Standardise train de ce fold
        X_val_cv_scaled = scaler_cv.transform(X_val_cv)  # Applique même transformation au validation
        
        model.fit(X_train_cv_scaled, y_train_cv)  # Entraîne modèle sur train de ce fold
        y_pred_cv = model.predict(X_val_cv_scaled)  # Prédit sur validation de ce fold
        
        cv_rmse_scores.append(np.sqrt(mean_squared_error(y_val_cv, y_pred_cv)))  # Ajoute RMSE à la liste
        cv_r2_scores.append(r2_score(y_val_cv, y_pred_cv))  # Ajoute R² à la liste
    
    return {  # Retourne statistiques agrégées de la validation croisée
        'cv_rmse_mean': np.mean(cv_rmse_scores),  # RMSE moyen sur tous les folds
        'cv_rmse_std': np.std(cv_rmse_scores),   # Écart-type RMSE (mesure stabilité)
        'cv_r2_mean': np.mean(cv_r2_scores),     # R² moyen sur tous les folds
        'cv_r2_std': np.std(cv_r2_scores)        # Écart-type R² (mesure stabilité)
    }


def tester_stabilite_predictions(model, scaler, X_test, y_test, n_bootstrap=50):  # Test robustesse par resampling
    """Test de stabilité par bootstrap pour mesurer la variance des prédictions"""
    if len(X_test) < 2:  # Pas assez de données pour bootstrap
        return {'pred_std': np.nan, 'pred_confidence_95': np.nan}  # Retourne NaN si impossible
    
    predictions = []  # Liste pour stocker prédictions de chaque échantillon bootstrap
    for _ in range(n_bootstrap):  # Répète n_bootstrap fois
        # Bootstrap sampling avec remplacement
        indices = np.random.choice(len(X_test), size=len(X_test), replace=True)  # Tire indices aléatoirement avec remise
        X_boot = X_test[indices]  # Crée échantillon bootstrap
        X_boot_scaled = scaler.transform(X_boot)  # Standardise échantillon bootstrap
        pred_boot = model.predict(X_boot_scaled)  # Prédit sur échantillon bootstrap
        predictions.append(pred_boot.mean())  # Stocke prédiction moyenne de cet échantillon
    
    pred_std = np.std(predictions)  # Écart-type des prédictions bootstrap (mesure variabilité)
    pred_confidence_95 = 1.96 * pred_std  # Intervalle de confiance à 95% (1.96 = quantile normal)
    
    return {  # Retourne métriques de stabilité
        'pred_std': pred_std,  # Écart-type des prédictions
        'pred_confidence_95': pred_confidence_95  # Marge d'erreur à 95%
    }


def detecter_overfitting(train_r2, val_r2, test_r2, train_rmse, test_rmse, threshold_r2=0.15, threshold_rmse_ratio=1.5):  # Détecte surapprentissage
    """Détecte le niveau d'overfitting avec plusieurs critères"""
    overfitting_signals = []  # Liste des signaux d'overfitting détectés
    overfitting_score = 0     # Score numérique du niveau d'overfitting
    
    # Critère 1 : Écart important entre train et test R²
    if not np.isnan(test_r2) and (train_r2 - test_r2) > threshold_r2:  # Si écart R² > seuil
        overfitting_signals.append(f"R² gap: {train_r2 - test_r2:.3f}")  # Ajoute signal à la liste
        overfitting_score += 1  # Incrémente score overfitting
    
    # Critère 2 : RMSE test >> RMSE train
    if not np.isnan(test_rmse) and (test_rmse / train_rmse) > threshold_rmse_ratio:  # Si RMSE test beaucoup plus élevé
        overfitting_signals.append(f"RMSE ratio: {test_rmse / train_rmse:.2f}x")  # Ajoute ratio RMSE
        overfitting_score += 1  # Incrémente score
    
    # Critère 3 : R² négatif sur test
    if not np.isnan(test_r2) and test_r2 < -0.5:  # R² fortement négatif = très mauvais
        overfitting_signals.append(f"R² négatif sévère: {test_r2:.3f}")  # Signal critique
        overfitting_score += 2  # Double pénalité pour R² très négatif
    
    # Critère 4 : Validation et test divergent fortement
    if not np.isnan(val_r2) and not np.isnan(test_r2):  # Si on a validation ET test
        if abs(val_r2 - test_r2) > 0.3:  # Écart validation/test > 30%
            overfitting_signals.append(f"Val/Test divergence: {abs(val_r2 - test_r2):.3f}")  # Signal instabilité
            overfitting_score += 1  # Incrémente score
    
    # Classification du niveau d'overfitting selon score
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
    
    # 🧪 ÉTAPE 6 : SÉLECTION ET VALIDATION DES FEATURES
    # Features et cible avec sélection robuste
    # 🎯 OBJECTIF : choisir quelles variables utiliser pour prédire le prix
    #
    # 📚 LISTE DES FEATURES CANDIDATES :
    feature_columns = [
        'Annee_idx',      # Position temporelle normalisée (0 → 1)
        'SurfaceMoy',     # Surface moyenne des logements en m²
        'PropMaisons',    # Proportion de maisons (0 → 1)
        'Prix_MA3',       # Moyenne mobile 3 ans (⚠️ PROBLÉMATIQUE)
        'Prix_Trend',     # Écart à la tendance (⚠️ TRÈS PROBLÉMATIQUE)
        'Mutations_Log'   # Log du nombre de mutations
    ]
    
    # 🔍 VÉRIFICATION : s'assurer que les colonnes existent vraiment
    # Parfois des colonnes peuvent manquer selon les données
    # [col for col in liste if condition] = list comprehension Python
    # Équivaut à : for col in feature_columns: if col in df_agg.columns: available_features.append(col)
    available_features = [col for col in feature_columns if col in df_agg.columns]
    
    # 🚺 GARDE-FOU : minimum 3 features pour entraîner un modèle décent
    # Si moins de 3 features, le modèle sera trop simple et peu fiable
    if len(available_features) < 3:
        return None
    # 📊 ÉTAPE 7 : PRÉPARATION DES MATRICES X (FEATURES) ET y (TARGET)
    #
    # 🎯 OBJECTIF : transformer le DataFrame en matrices numpy pour le ML
    #
    # 📚 COURS ML : NOTATION STANDARD
    # X = matrice des features (variables explicatives)
    #     Chaque ligne = une observation (année)
    #     Chaque colonne = une feature (Annee_idx, SurfaceMoy, etc.)
    #     Dimension : [n_annees, n_features]
    #
    # y = vecteur target (variable à prédire)
    #     Chaque élément = le prix à prédire pour une année
    #     Dimension : [n_annees]
    #
    # .values = convertit DataFrame pandas → array numpy (plus rapide pour ML)
    #
    # 📚 EXEMPLE CONCRET :
    # Si df_agg contient :
    #   annee | Annee_idx | SurfaceMoy | PropMaisons | Prixm2Moyen
    #   2020  | 0.0       | 85         | 0.3         | 1500
    #   2021  | 0.25      | 87         | 0.35        | 1600
    #   2022  | 0.5       | 89         | 0.4         | 1700
    #
    # X = [[0.0,  85, 0.3 ],      y = [1500,
    #      [0.25, 87, 0.35],           1600,
    #      [0.5,  89, 0.4 ]]           1700]
    #
    X = df_agg[available_features].values  # Matrice features (2D)
    y = df_agg['Prixm2Moyen'].values      # Vecteur target (1D)
    annees = df_agg['annee'].values        # Vecteur années (pour l'affichage)
    
    # 🧹 ÉTAPE 8 : VALIDATION ET NETTOYAGE FINAL
    # Validation des données
    # 🎯 OBJECTIF : vérifier qu'il n'y a pas de NaN ou valeurs invalides
    #
    # np.any(np.isnan(X)) = vérifie s'il y a AU MOINS un NaN dans X
    # NaN = "Not a Number" = valeur indéfinie (ex: 0/0, log(-1))
    # Si NaN présent → le ML va planter → on arrête tout
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        return None
        
    # 🎯 ÉTAPE 9 : SPLIT TEMPOREL (TIME SERIES SPLIT)
    # 📚 COURS : POURQUOI PAS train_test_split() CLASSIQUE ?
    #
    # PROBLÈME avec split aléatoire :
    # train_test_split() mélange aléatoirement les données
    # On pourrait entraîner sur 2023 et tester sur 2020 = FUITE TEMPORELLE !
    # Le modèle connaîtrait le futur pour prédire le passé = triche
    #
    # SOLUTION : Split temporel chronologique
    # Train = années les plus anciennes
    # Validation = années moyennes  
    # Test = années les plus récentes
    #
    # 📚 EXEMPLE CONCRET :
    # Années : [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    # Train : [2014, 2015, 2016, 2017, 2018, 2019]  (60%)
    # Valid : [2020, 2021]                           (20%)
    # Test  : [2022, 2023, 2024]                     (20%)
    #
    # Split temporel plus conservateur pour éviter l'overfitting
    n_years = len(df_agg)
    
    # 🧠 STRATÉGIE ADAPTATIVE DE SPLIT (selon la quantité de données)
    # 🎯 OBJECTIF : adapter le ratio Train/Validation/Test selon le dataset
    #
    # 📚 COURS : DILEMME BIAIS-VARIANCE EN PETIT DATASET
    # - Plus de données en train = modèle mieux entraîné (moins de biais)
    # - Plus de données en test = évaluation plus fiable (moins de variance)
    # Avec peu de données, il faut choisir !
    #
    if n_years < 6:
        # 🚨 TRÈS PETIT DATASET : 50-50 train/test pour éviter l'overfitting
        # Pas de validation = pas assez de données
        # Priorité : éviter l'overfitting plutôt que d'optimiser
        # 
        # 📚 EXEMPLE : 5 années [2020, 2021, 2022, 2023, 2024]
        # train_size = max(int(5 * 0.5), 2) = max(2, 2) = 2
        # Train: [2020, 2021], Test: [2022, 2023, 2024]
        train_size = max(int(n_years * 0.5), 2)
        val_size = 0
        test_size = n_years - train_size
        
    elif n_years < 10:
        # 🟠 PETIT DATASET : 60% train, 40% test
        # 1 année en validation si possible
        # 
        # 📚 EXEMPLE : 8 années [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        # test_size = max(int(8 * 0.4), 3) = max(3, 3) = 3
        # val_size = 1 if 8 > 6 else 0 = 1
        # train_size = 8 - 1 - 3 = 4
        # Train: [2017, 2018, 2019, 2020], Val: [2021], Test: [2022, 2023, 2024]
        test_size = max(int(n_years * 0.4), 3)
        val_size = 1 if n_years > 6 else 0
        train_size = n_years - val_size - test_size
        
    else:
        # 🟢 DATASET NORMAL : 60% train, 20% validation, 20% test
        # Split classique équilibré
        # 
        # 📚 EXEMPLE : 15 années [2010-2024]
        # test_size = max(int(15 * 0.2), 2) = max(3, 2) = 3
        # val_size = max(int(15 * 0.2), 2) = max(3, 2) = 3
        # train_size = 15 - 3 - 3 = 9
        # Train: [2010-2018], Val: [2019-2021], Test: [2022-2024]
        test_size = max(int(n_years * 0.2), 2)
        val_size = max(int(n_years * 0.2), 2)
        train_size = n_years - val_size - test_size
    
    # ✂️ ÉTAPE 10 : CRÉATION DES ENSEMBLES TRAIN/VALIDATION/TEST
    # 🎯 OBJECTIF : découper chronologiquement les données
    #
    # 📚 COURS PYTHON : SLICING D'ARRAYS
    # array[:n] = les n premiers éléments
    # array[n:m] = les éléments de l'index n à m-1
    # array[n:] = les éléments à partir de l'index n
    #
    # 📚 EXEMPLE CONCRET :
    # Si train_size=4, val_size=1, test_size=3
    # X_train = X[:4] = lignes 0,1,2,3 (4 années les plus anciennes)
    # X_val = X[4:5] = ligne 4 (année du milieu)
    # X_test = X[5:] = lignes 5,6,7 (3 années les plus récentes)
    #
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    annees_train = annees[:train_size]
    annees_val = annees[train_size:train_size+val_size]
    annees_test = annees[train_size+val_size:]
    
    # 📊 ÉTAPE 11 : STANDARDISATION (FEATURE SCALING)
    # 🎯 OBJECTIF : normaliser toutes les features sur la même échelle
    #
    # 📚 COURS ML : POURQUOI STANDARDISER ?
    # PROBLÈME : features ont des échelles différentes
    # - Annee_idx : [0, 1]
    # - SurfaceMoy : [50, 150]
    # - Mutations_Log : [4, 10]
    #
    # Sans standardisation, le ML privilégie les features avec grandes valeurs
    # SurfaceMoy=100 aura 100x plus d'impact qu'Annee_idx=0.5 !
    #
    # SOLUTION : StandardScaler
    # Transforme chaque feature : (valeur - moyenne) / écart-type
    # Résultat : toutes les features ont moyenne=0 et écart-type=1
    #
    # 📚 EXEMPLE CONCRET :
    # AVANT : SurfaceMoy = [80, 85, 90, 95] (moyenne=87.5, std=6.45)
    # APRÈS : SurfaceMoy = [-1.16, -0.39, 0.39, 1.16] (moyenne=0, std=1)
    #
    # ⚠️ IMPORTANT : on fit seulement sur TRAIN, puis on transforme train/val/test
    # Sinon = fuite de données (data leakage) !
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit + Transform sur Train
    X_val_scaled = scaler.transform(X_val) if len(X_val) > 0 else np.array([])    # Transform seulement
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.array([]) # Transform seulement
    
    # 📈 ÉTAPE 12 : ANALYSER LA TENDANCE DES DONNÉES (pour adapter les hyperparamètres)
    # Analyser la tendance des données
    # 🎯 OBJECTIF : comprendre la complexité du signal pour adapter la régularisation
    tendance_info = analyser_tendance(y, annees)
    
    # Détecter les outliers
    # 🎯 OBJECTIF : identifier les années avec des prix aberrants (crises, bulles...)
    outliers_idx, z_scores = detecter_outliers_temporels(y)
    
    # 🤖 ÉTAPE 13 : DÉFINITION DES MODÈLES ML AVEC RÉGULARISATION FORTE
    # 🎯 OBJECTIF : créer 3 modèles avec différents niveaux de régularisation
    #
    # 📚 COURS : RÉGULARISATION POUR ÉVITER L'OVERFITTING
    #
    # PROBLÈME : avec peu de données, le modèle peut "apprendre par cœur"
    # = mémoriser parfaitement les données d'entraînement mais être nul sur nouveaux data
    #
    # SOLUTION : Régularisation = ajouter une "pénalité" aux coefficients trop gros
    #
    # Hyperparamètres adaptatifs selon la taille du dataset et la volatilité
    # 📚 LOGIQUE D'ADAPTATION :
    # - Dataset petit (< 8 ans) → alpha plus élevé (éviter overfitting)
    # - Volatilité haute → alpha plus élevé (stabiliser le modèle)
    # - Situation normale → alpha modéré
    base_ridge_alpha = 100.0 if n_years < 8 else 50.0 if tendance_info['volatilite'] == 'HAUTE' else 10.0
    base_lasso_alpha = 10.0 if n_years < 8 else 5.0 if tendance_info['volatilite'] == 'HAUTE' else 1.0
    
    # 📚 COURS MATHS : LES 3 MODÈLES DE RÉGULARISATION
    #
    # 🔵 1. RIDGE CONSERVATIVE (alpha élevé)
    # FORMULE : min[ Σ(y - ŷ)² + α×Σ(βⱼ²) ]
    #           ↓        ↓        ↓
    #         Erreur  +  Pénalité  Ridge
    #
    # α = base_ridge_alpha × 2 = DOUBLE pénalité
    # EFFET : coefficients très "shrinkés" vers 0
    # AVANTAGE : très robuste, jamais d'overfitting sévère
    # INCONVÉNIENT : peut être trop rigide, sous-performance possible
    #
    # 📚 EXEMPLE CONCRET :
    # Sans Ridge : β = [2000, -500, 800] (coefficients gros)
    # Ridge Conservative : β = [200, -50, 80] (coefficients réduits 10x)
    #
    models = {
        'Ridge_Conservative': Ridge(alpha=base_ridge_alpha * 2, random_state=42),
        
        # 🟡 2. RIDGE MODERATE (alpha moyen)
        # FORMULE : min[ Σ(y - ŷ)² + α×Σ(βⱼ²) ]
        # α = base_ridge_alpha = pénalité équilibrée
        # EFFET : bon compromis biais-variance
        # AVANTAGE : souvent le meilleur choix en pratique
        # INCONVÉNIENT : peut encore overfitter si données très complexes
        #
        # 📚 EXEMPLE CONCRET :
        # Ridge Moderate : β = [400, -100, 160] (coefficients réduits 5x)
        #
        'Ridge_Moderate': Ridge(alpha=base_ridge_alpha, random_state=42),
        
        # 🔴 3. LASSO ROBUST (sélection de features)
        # FORMULE : min[ Σ(y - ŷ)² + α×Σ|βⱼ| ]
        #           ↓        ↓        ↓
        #         Erreur  +  Pénalité  L1
        #
        # DIFFÉRENCE avec Ridge : |βⱼ| au lieu de βⱼ²
        # EFFET MAGIQUE : met certains coefficients exactement à 0
        # = SÉLECTION AUTOMATIQUE des features importantes
        # AVANTAGE : modèle plus simple et interprétable
        # INCONVÉNIENT : peut éliminer des features utiles
        #
        # 📚 EXEMPLE CONCRET :
        # Lasso : β = [0, -100, 160] (première feature éliminée !)
        #         Le modèle décide automatiquement de ne pas utiliser Annee_idx
        #
        'Lasso_Robust': Lasso(alpha=base_lasso_alpha, max_iter=10000, random_state=42)
    }
    
    # 🎯 ÉTAPE 14 : ENTRAÎNEMENT ET ÉVALUATION DES 3 MODÈLES
    # 📚 COURS : PROCESSUS DE VALIDATION CROISÉE
    #
    # OBJECTIF : tester chaque modèle sur Train/Validation/Test
    # et calculer des métriques pour comparer leur performance
    #
    # y_train_mean = moyenne des prix d'entraînement (pour métrique "naive")
    y_train_mean = np.mean(y_train)
    
    results = {}  # Dictionnaire pour stocker les résultats de chaque modèle
    
    # 🔁 BOUCLE PRINCIPALE : tester chaque modèle
    for name, model in models.items():
        # 📚 ÉTAPE 14.1 : ENTRAÎNEMENT SUR TRAIN SET
        # .fit() = algorithme d'optimisation qui trouve les meilleurs coefficients β
        # Pour Ridge : résout min[ Σ(y - ŷ)² + α×Σ(βⱼ²) ]
        # Pour Lasso : résout min[ Σ(y - ŷ)² + α×Σ|βⱼ| ]
        #
        # 📚 MATHÉMATIQUES INTERNES :
        # L'algorithme utilise la descente de gradient ou des formules analytiques
        # pour trouver les β qui minimisent la fonction objectif
        #
        model.fit(X_train_scaled, y_train)
        
        # 📚 ÉTAPE 14.2 : PRÉDICTIONS SUR TRAIN (pour détecter l'overfitting)
        # .predict() = applique l'équation ŷ = β₀ + β₁×x₁ + β₂×x₂ + ... + βₙ×xₙ
        # avec les coefficients β trouvés lors du .fit()
        y_train_pred = model.predict(X_train_scaled)
        
        # calculer_metriques_avancees() = calcule R², RMSE, MAE, etc.
        # (voir fonction détaillée plus bas)
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
    
    # Sélection du modèle avec focus anti-overfitting
    model_scores = {}
    
    for name, res in results.items():
        score = 0
        
        # Critère 1 : R² test positif (poids 50%) - priorité absolue
        if not np.isnan(res['test_r2']):
            if res['test_r2'] > 0:
                score += res['test_r2'] * 50
            else:
                score -= abs(res['test_r2']) * 10  # Pénalité forte pour R² négatif
        
        # Critère 2 : Pas d'overfitting sévère (poids 30%)
        if res['overfitting']['overfitting_level'] == 'AUCUN':
            score += 30
        elif res['overfitting']['overfitting_level'] == 'LÉGER':
            score += 15
        elif res['overfitting']['overfitting_level'] == 'MODÉRÉ':
            score += 5
        # SÉVÈRE = 0 points
        
        # Critère 3 : RMSE raisonnable (poids 20%)
        if not np.isnan(res['test_rmse']):
            # Pénaliser les RMSE extrêmes
            if res['test_rmse'] < 1000:  # RMSE acceptable
                rmse_scores = [results[k]['test_rmse'] for k in results.keys() if not np.isnan(results[k]['test_rmse']) and results[k]['test_rmse'] < 1000]
                if len(rmse_scores) > 0:
                    rmse_normalized = 1 - (res['test_rmse'] - min(rmse_scores)) / (max(rmse_scores) - min(rmse_scores) + 1e-10)
                    score += rmse_normalized * 20
        
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
            last_row['PropMaisons'],
            prev_prix,
            0,  # Prix_Trend
            last_row['Mutations_Log']
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


def afficher_resultats(resultat):  # Affiche résultats formatés de la prédiction
    """Affiche les résultats de la prédiction"""
    if resultat is None:  # Si aucun résultat (pas assez de données)
        print("❌ Données insuffisantes (minimum 5 ans requis)\n")  # Message d'erreur
        return  # Quitte la fonction
    
    dept = resultat['departement']  # Code département (ex: '75')
    best = resultat['best_model_info']  # Infos du meilleur modèle
    
    print("="*100)
    print(f"📊 PRÉDICTION POUR LE DÉPARTEMENT {dept}")  # Affichage simplifié sans nom complet
    print("="*100)
    
    print(f"\n🧠 MEILLEUR MODÈLE : {resultat['best_model_name']}")
    print(f"  • Score composite       : {resultat['model_scores'][resultat['best_model_name']]:.2f}/100")
    print(f"  • R² sur train          : {best['train_r2']:.4f}")
    
    if not np.isnan(best['test_r2']):
        r2_emoji = "✅" if best['test_r2'] >= 0.5 else "⚠️" if best['test_r2'] >= 0 else "❌"
        print(f"  • R² sur test           : {best['test_r2']:.4f} {r2_emoji}")
        
        if not np.isnan(best['test_direction_accuracy']):
            dir_emoji = "🎯" if best['test_direction_accuracy'] >= 70 else "⚠️"
            print(f"  • Direction Accuracy    : {best['test_direction_accuracy']:.1f}% {dir_emoji}")
        
        if not np.isnan(best['improvement_vs_naive']):
            improv_emoji = "📈" if best['improvement_vs_naive'] > 0 else "📉"
            print(f"  • Amélioration vs Naïf  : {best['improvement_vs_naive']:.1f}% {improv_emoji}")
        
        print(f"\n  🔬 DIAGNOSTIC D'OVERFITTING :")
        print(f"     Niveau : {best['overfitting']['overfitting_level']}")
        if best['overfitting']['overfitting_signals']:
            for signal in best['overfitting']['overfitting_signals']:
                print(f"     ⚠️  {signal}")
    
    if not np.isnan(best['cv_rmse_mean']):
        print(f"\n  📊 VALIDATION CROISÉE TEMPORELLE :")
        print(f"     RMSE moyen : {best['cv_rmse_mean']:.2f} ± {best['cv_rmse_std']:.2f} €/m²")
        print(f"     R² moyen   : {best['cv_r2_mean']:.4f}")
    
    print(f"\n📊 COMPARAISON DES 3 MODÈLES :")
    print(f"{'Modèle':<20} {'Score':<10} {'R² Test':<12} {'Overfitting':<15} {'Diagnostic':<20}")
    print("-"*80)
    for name, res in resultat['results'].items():
        score_str = f"{resultat['model_scores'].get(name, 0):.1f}" if resultat['model_scores'] else "N/A"
        r2_str = f"{res['test_r2']:.4f}" if not np.isnan(res['test_r2']) else "N/A"
        ovf_str = res['overfitting']['overfitting_level'][:8]  # Raccourci
        
        # Diagnostic simplifié
        if not np.isnan(res['test_r2']):
            if res['test_r2'] >= 0.5:
                diagnostic = "BON"
            elif res['test_r2'] >= 0:
                diagnostic = "MOYEN"
            else:
                diagnostic = "MAUVAIS"
        else:
            diagnostic = "N/A"
            
        marker = "🏆" if name == resultat['best_model_name'] else "  "
        print(f"{marker} {name:<18} {score_str:<10} {r2_str:<12} {ovf_str:<15} {diagnostic:<20}")
    
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


def visualiser_prediction(resultat):  # Crée graphiques matplotlib de la prédiction
    """Visualisation avancée avec focus sur les métriques importantes"""
    if resultat is None:  # Pas de résultat à afficher
        return  # Quitte sans rien faire
    
    dept = resultat['departement']  # Code département
    
    plt.figure(figsize=(16, 12))  # Crée figure matplotlib 16x12 pouces
    
    # Graphique 1 : Historique + Prédictions (plus grand)
    plt.subplot(3, 2, (1, 2))  # Occupe 2 colonnes sur grille 3x2
    
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
    
    # Prédictions futures
    future_years = [d['annee'] for d in resultat['future_data']]
    future_prices = [d['prix_pred'] for d in resultat['future_data']]
    plt.plot(future_years, future_prices, 'D--', color='#C73E1D', linewidth=2.5, markersize=10, label='Futur 2025-2027', alpha=0.9)
    
    plt.axvline(x=2024.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)  # Ligne verticale séparant historique/futur
    plt.title(f"Département {dept} | Modèle: {resultat['best_model_name']} | R²={best['test_r2']:.3f}", fontsize=14, fontweight='bold')  # Titre simplifié
    plt.xlabel('Année', fontsize=12)  # Label axe X
    plt.ylabel('Prix au m² (€)', fontsize=12)  # Label axe Y
    plt.grid(True, alpha=0.3)  # Grille transparente
    plt.legend(fontsize=9, loc='best')  # Légende auto-positionnée
    
    # Graphique 2 : Diagnostic des erreurs
    if len(resultat['annees_test']) > 0:
        plt.subplot(3, 2, 3)
        residuals = resultat['y_test'] - best['y_test_pred']
        colors = ['green' if abs(r) < np.std(residuals) else 'red' for r in residuals]
        plt.bar(resultat['annees_test'], residuals, color=colors, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.axhline(y=np.std(residuals), color='orange', linestyle='--', alpha=0.7, label='±1 σ')
        plt.axhline(y=-np.std(residuals), color='orange', linestyle='--', alpha=0.7)
        plt.title('Résidus et Erreurs Significatives', fontsize=12, fontweight='bold')
        plt.xlabel('Année', fontsize=10)
        plt.ylabel('Résidu (€/m²)', fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3, axis='y')
    
    # Graphique 3 : Comparaison R² avec seuils de qualité
    plt.subplot(3, 2, 4)
    model_names = list(resultat['results'].keys())
    test_r2s = [resultat['results'][name]['test_r2'] for name in model_names]
    
    # Couleurs selon la qualité du R²
    colors = []
    for name, r2 in zip(model_names, test_r2s):
        if name == resultat['best_model_name']:
            colors.append('darkgreen' if r2 >= 0.5 else 'orange' if r2 >= 0 else 'darkred')
        else:
            colors.append('lightgreen' if r2 >= 0.5 else 'lightblue' if r2 >= 0 else 'lightcoral')
    
    plt.barh(model_names, test_r2s, color=colors, alpha=0.8)
    plt.xlabel('R² sur Test', fontsize=10)
    plt.title('Performance R² (Vert=Bon, Orange=Moyen, Rouge=Mauvais)', fontsize=10, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.8, label='Seuil minimum')
    plt.axvline(x=0.5, color='green', linestyle='--', linewidth=1, alpha=0.8, label='Seuil bon')
    plt.legend(fontsize=8)
    
    # Graphique 4 : Case avec R² exact du meilleur modèle
    plt.subplot(3, 2, 5)
    best_res = resultat['results'][resultat['best_model_name']]
    r2_value = best_res['test_r2']
    
    # Créer une "case" avec le R² exact
    plt.text(0.5, 0.7, f"R² EXACT", fontsize=16, fontweight='bold', ha='center', transform=plt.gca().transAxes)
    plt.text(0.5, 0.5, f"{r2_value:.6f}", fontsize=20, fontweight='bold', ha='center', transform=plt.gca().transAxes)
    
    # Colorier selon la qualité
    if r2_value >= 0.5:
        color_text = "Excellente"
        color_bg = 'green'
    elif r2_value >= 0.2:
        color_text = "Bonne"
        color_bg = 'orange' 
    elif r2_value >= 0:
        color_text = "Acceptable"
        color_bg = 'yellow'
    else:
        color_text = "Mauvaise"
        color_bg = 'red'
    
    plt.text(0.5, 0.3, f"Qualité: {color_text}", fontsize=12, ha='center', transform=plt.gca().transAxes)
    plt.text(0.5, 0.15, f"Modèle: {resultat['best_model_name']}", fontsize=10, ha='center', transform=plt.gca().transAxes)
    
    # Supprimer les axes pour faire une "case"
    plt.gca().set_xlim(0, 1)
    plt.gca().set_ylim(0, 1)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, facecolor=color_bg, alpha=0.2, edgecolor='black', linewidth=2))
    plt.title('Résultat R² du Meilleur Modèle', fontsize=12, fontweight='bold')
    
    # Graphique 5 : Supprimé (remplacé par du texte explicatif)
    plt.subplot(3, 2, 6)  # Position 6 sur grille 3x2
    plt.text(0.5, 0.6, "INFORMATION", fontsize=14, fontweight='bold', ha='center', transform=plt.gca().transAxes)  # Titre centré
    plt.text(0.5, 0.4, f"Département analysé:", fontsize=10, ha='center', transform=plt.gca().transAxes)  # Sous-titre
    plt.text(0.5, 0.3, f"Code {dept}", fontsize=12, fontweight='bold', ha='center', transform=plt.gca().transAxes)  # Code département
    plt.text(0.5, 0.1, f"Années de données: {resultat['nb_annees']}", fontsize=10, ha='center', transform=plt.gca().transAxes)  # Nb années
    
    # Supprimer les axes
    plt.gca().set_xlim(0, 1)
    plt.gca().set_ylim(0, 1)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().add_patch(plt.Rectangle((0.1, 0.05), 0.8, 0.8, facecolor='lightblue', alpha=0.3, edgecolor='black', linewidth=1))
    
    plt.tight_layout()
    plt.show()


def main():
    """💻 PROGRAMME PRINCIPAL - INTERFACE UTILISATEUR
    
    Cette fonction orchestre tout le processus :
    1. Charge les données
    2. Demande à l'utilisateur un code postal
    3. Extrait le département
    4. Lance la prédiction ML
    5. Affiche les résultats et graphiques
    """
    print("\n" + "="*100)
    print("🤖 PRÉDICTION DU PRIX AU M² PAR DÉPARTEMENT - MODÈLES LINÉAIRES (2025-2027)")
    print("="*100 + "\n")
    
    # 📚 ÉTAPE 1 : CHARGEMENT DES DONNÉES
    # Charger les données
    df_clean = charger_donnees()
    
    # 📚 ÉTAPE 2 : INTERFACE UTILISATEUR
    # input() = fonction Python pour demander une saisie à l'utilisateur
    # .strip() = enlève les espaces en début/fin (au cas où l'utilisateur tape " 75001 ")
    # Demander le code postal
    code_postal = input("📍 Entrez un code postal (ex: 75001, 69001, 33000) : ").strip()
    
    # 📚 ÉTAPE 3 : VALIDATION DE LA SAISIE
    # Vérification basique : le code postal doit faire au moins 2 caractères
    # (pour extraire le département)
    # Extraire le département (2 premiers chiffres)
    if len(code_postal) < 2:
        print("❌ Code postal invalide (minimum 2 chiffres)\n")
        return  # Quitte la fonction = arrête le programme
    
    # 📚 ÉTAPE 4 : EXTRACTION DU DÉPARTEMENT
    # Logique métier : code postal 75001 → département 75 (Paris)
    #                  code postal 69123 → département 69 (Rhône)
    # [:2] = slice des 2 premiers caractères
    code_dept = code_postal[:2]
    print(f"\n🔍 Analyse du département {code_dept}...\n")
    
    # 📚 ÉTAPE 5 : VÉRIFICATION DE L'EXISTENCE DU DÉPARTEMENT
    # .values = convertit la Series pandas en array numpy
    # 'in' = opérateur Python pour vérifier l'appartenance
    # Vérifier que le département existe
    if code_dept not in df_clean['Departement'].values:
        print(f"❌ Département {code_dept} non trouvé dans la base\n")
        
        # 📚 AIDE À L'UTILISATEUR : afficher les départements disponibles
        # .unique() = valeurs uniques (sans doublons)
        # sorted() = tri par ordre alphabétique/numérique
        # [:20] = affiche seulement les 20 premiers (pour ne pas surcharger)
        depts_dispo = sorted(df_clean['Departement'].unique())
        print(f"Départements disponibles ({len(depts_dispo)}) : {', '.join(depts_dispo[:20])}...")
        return  # Arrête le programme
    
    # 📚 ÉTAPE 6 : AFFICHAGE DU PROCESSUS EN COURS
    # Messages informatifs pour que l'utilisateur comprenne ce qui se passe
    print("⚙️ Feature engineering en cours...")
    print("🧠 Entraînement de 3 modèles régularisés (Ridge Conservative/Moderate, Lasso Robust)...")
    print("🔀 Split temporel : Train / Validation / Test\n")
    
    # 📚 ÉTAPE 7 : EXÉCUTION DE LA PRÉDICTION ML
    # Effectuer la prédiction avec modèle ML avancé
    # predire_departement() = fonction principale qui fait tout le travail ML
    # Retourne un dictionnaire avec tous les résultats ou None si échec
    resultat = predire_departement(df_clean, code_dept)
    
    # 📚 ÉTAPE 8 : AFFICHAGE DES RÉSULTATS
    # Afficher les résultats
    # afficher_resultats() = fonction qui formate joliment les résultats
    afficher_resultats(resultat)
    
    # 📚 ÉTAPE 9 : VISUALISATION GRAPHIQUE
    # Visualiser seulement si on a des résultats valides
    if resultat is not None:
        # visualiser_prediction() = crée les graphiques matplotlib
        visualiser_prediction(resultat)
        
        # 📚 ÉTAPE 10 : RÉSUMÉ FINAL ET PÉDAGOGIE
        # Affichage d'un résumé compact et d'avertissements pédagogiques
        #
        print("\n" + "="*100)
        print("📋 RÉSUMÉ DE L'ANALYSE")
        print("="*100)
        
        # Informations clés extraites du résultat
        print(f"\n🎯 Département : {code_dept}")
        print(f"📊 Années de données : {resultat['nb_annees']}")
        print(f"🧠 Meilleur modèle : {resultat['best_model_name']}")
        print(f"📈 Tendance 2024-2027 : {resultat['variation_pct']:+.2f}%")
        
        # 🎓 SECTION PÉDAGOGIQUE IMPORTANTE
        print("\n⚠️ LIMITES ET MISE EN GARDE :")
        print("  • Prédictions basées sur la continuité des tendances historiques")
        print("  • Ne prend pas en compte les chocs économiques majeurs")
        print("    (guerre, pandémie, crise financière, changement de politique...)")
        print("  • Utiliser comme ordre de grandeur, pas comme certitude absolue")
        print("  • Un R² négatif signifie que le modèle performe moins bien qu'une simple moyenne")
        print("    (souvent dû à un test set trop petit ou à un overfitting sévère)")
        
        # 📚 EXPLICATION DU R² POUR L'UTILISATEUR
        print("\n📚 COMMENT INTERPRÉTER LE R² :")
        print("  • R² = 0.8 → Le modèle explique 80% de la variance = EXCELLENT")
        print("  • R² = 0.5 → Le modèle explique 50% de la variance = BON")
        print("  • R² = 0.2 → Le modèle explique 20% de la variance = MOYEN")
        print("  • R² = 0.0 → Le modèle = prédiction naïve (moyenne) = MAUVAIS")
        print("  • R² < 0   → Le modèle est pire qu'une simple moyenne = CATASTROPHIQUE")
        
        print("\n" + "="*100)
        print("✅ ANALYSE TERMINÉE")
        print("="*100 + "\n")


# 📚 POINT D'ENTRÉE DU PROGRAMME
# Cette condition vérifie si le fichier est exécuté directement (pas importé)
# __name__ == "__main__" = True seulement si on lance "python code_essaye.py"
# Si on fait "import code_essaye", cette condition sera False
#
# 🎯 OBJECTIF : permettre d'utiliser ce fichier comme :
# 1. Programme exécutable : python code_essaye.py
# 2. Module importable : from code_essaye import predire_departement
#
if __name__ == "__main__":
    main()  # Lance le programme principal
