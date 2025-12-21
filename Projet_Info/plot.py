"""
VISUALISATION AVANCÉE - PROJET INFO 3
Graphiques combinant données historiques de 200 ans + données DVF récentes
Affichage sur 2 graphiques optimisés
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import vprint

def visualiser_analyse(results):
    """Affiche les résultats de l'analyse avec 2 graphiques optimisés"""
    
    if results is None:
        print("[ERROR] Aucun resultat a afficher")
        return
        return
    
    vprint("[INFO] Generation des graphiques avances...")
    
    # Configuration matplotlib (compact view requested)
    plt.style.use('seaborn-v0_8')
    plt.rcParams['font.size'] = 9

    # Taille de figure (retour à la valeur par défaut pour lisibilité)
    fig_w, fig_h = (10, 6)
    top_ratio = 3
    # Paramètres de rendu (ajustés pour local vs national)
    if results.get('type', '') == 'local':
        marker_size = 6
        future_marker_size = 5
    else:
        marker_size = 3
        future_marker_size = 4

    # Utiliser GridSpec pour contrôler mieux la disposition : ax1 large en haut, ax2 plus étroit en bas-gauche
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(nrows=2, ncols=3, height_ratios=[top_ratio, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    # ax2 occupe la rangée du bas mais uniquement les deux premières colonnes (gauche)
    ax2 = fig.add_subplot(gs[1, 0:2])
    
    # === GRAPHIQUE 1: ÉVOLUTION COMPLÈTE ===
    
    # Données historiques (200 ans)
    df_historique = results['df_historique']
    if df_historique is not None:
        # Sous-échantillonner la série historique pour alléger l'affichage
        try:
            hist_plot = df_historique.sort_values('annee').reset_index(drop=True).iloc[::3]
        except Exception:
            hist_plot = df_historique
        ax1.plot(hist_plot['annee'], hist_plot['prix_historique'], color='lightcoral', linewidth=1.2, alpha=0.8, label=f'Données historiques ({df_historique["annee"].min():.0f}-{df_historique["annee"].max():.0f})')
    
    # Données DVF récentes
    df_dvf = results['df_dvf']
    ax1.plot(df_dvf['annee'], df_dvf['prix_dvf'], color='steelblue', linewidth=1.2, marker='o', markersize=marker_size, alpha=0.95, label=f'Données DVF ({df_dvf["annee"].min():.0f}-{df_dvf["annee"].max():.0f})')
    
    # Prédictions train/test (afficher entraînement comme ligne pour réduire le bruit)
    # Montrer la courbe d'entraînement uniquement pour l'analyse nationale
    show_train = results.get('type', '') != 'local'
    if show_train and len(results['annees_train']) > 0:
        y_train_pred = results['best_model_info']['y_train_pred']
        # trier les années avant de tracer
        train_years = np.array(results['annees_train'])
        try:
            order = np.argsort(train_years)
            train_years_sorted = train_years[order]
            y_train_sorted = np.array(y_train_pred)[order]
        except Exception:
            train_years_sorted = train_years
            y_train_sorted = np.array(y_train_pred)
        ax1.plot(train_years_sorted, y_train_sorted, color='orange', alpha=0.7, linewidth=1.0, label=f'Entraînement (R²={results["best_model_info"]["train_r2"]:.3f})')
    
    if len(results['annees_test']) > 0:
        y_test_pred = results['best_model_info']['y_test_pred']
        # Afficher la validation DVF (si disponible) — plus représentative pour la série DVF
        dvf_r2 = results['best_model_info'].get('dvf_r2', np.nan)
        dvf_mae = results['best_model_info'].get('dvf_mae', np.nan)
        if not np.isnan(dvf_r2):
            ax1.scatter([], [], color='red', s=24, marker='^', label=f'DVF R²={dvf_r2:.3f} MAE={dvf_mae:.0f}')
        
        # Ligne de référence test
        ax1.plot(results['annees_test'], results['y_test'], 
                color='red', linewidth=2, linestyle='--', alpha=0.7,
                    label='Valeurs reelles test (DVF)')
    
    # Prédictions futures
    future_years = [pred['annee'] for pred in results['future_data']]
    future_prices = [pred['prix_pred'] for pred in results['future_data']]
    
    # Connexion 2024 -> futures
    last_year = df_dvf['annee'].iloc[-1]
    last_price = df_dvf['prix_dvf'].iloc[-1]
    
    ax1.plot([last_year] + future_years, [last_price] + future_prices, color='darkgreen', linewidth=1.2, marker='s', markersize=future_marker_size, label=f'Prédictions futures (2025-2027)')
    
    # Annotations futures : placer l'annotation principale plus bas pour ne pas chevaucher la bordure
    x_annot = last_year + 0.5
    # Décalage vertical réduit pour petits prix et augmenté pour grands prix
    for i, (year, price) in enumerate(zip(future_years, future_prices)):
        if i < len(future_years) - 1:
            ax1.text(year, price + (0.01 * last_price + 3), f'{price:.0f}', ha='center', va='bottom', fontsize=8, color='darkgreen')
        else:
            # Pour la dernière étiquette (2027), la placer plus bas (va='top') et décaler verticalement négativement
            dy = -(0.06 * last_price + 15)
            ax1.annotate(f'{year}: {price:.0f} €/m²', xy=(year, price), xytext=(x_annot, price + dy), textcoords='data', ha='left', va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.95), arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1))
    
    # Configuration graphique 1
    ax1.set_title(f'ANALYSE AVANCEE FRANCE - Modele {results["best_model_name"]}\n'
                 f'Donnees de 200 ans + DVF recentes | Evolution {results["variation_pct"]:+.1f}% (2024-2027)', 
                 fontsize=14, fontweight='bold', pad=10)
    
    ax1.set_xlabel('Année', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Prix immobilier (€/m²)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Zones de couleur pour les périodes
    if df_historique is not None:
        ax1.axvspan(df_historique['annee'].min(), 2013, alpha=0.1, color='coral', label='_Période historique')
    ax1.axvspan(2014, 2024, alpha=0.1, color='blue', label='_Période DVF')
    ax1.axvspan(2025, 2027, alpha=0.1, color='green', label='_Prédictions')
    
    # Définir l'axe x : pour les analyses locales, zoomer sur les dernières années
    try:
        global_min_year = int(df_historique['annee'].min()) if df_historique is not None else int(df_dvf['annee'].min())
    except Exception:
        global_min_year = int(df_dvf['annee'].min())
    global_max_year = int(df_dvf['annee'].max())

    if results.get('type', '') == 'local':
        # Afficher uniquement les N dernières années (ex : dernières 8 années disponibles)
        N = 8
        calculated_xmin = max(global_min_year, global_max_year - (N - 1))
        # S'assurer que 2015 reste visible si les données couvrent 2015
        desired_min = 2015
        xmin = max(global_min_year, min(calculated_xmin, desired_min))
        xmax = 2027
        ax1.set_xlim(xmin, xmax + 1)
    else:
        # Mode national : afficher toute la période historique + prédictions
        xmin = global_min_year
        xmax = 2027
        ax1.set_xlim(xmin - 1, xmax + 1)
    
    # Eviter que la légende n'écrase le graphique
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.95)
    
    # === GRAPHIQUE 2: COMPARAISON DES MODÈLES ===
    
    model_names = []
    train_r2_scores = []
    validation_r2_scores = []  # Utiliser validation au lieu de test
    
    for name, info in results['results'].items():
        model_names.append(name.replace('_forte', '').replace('_moyenne', '').replace('_simple', ''))
        train_r2_scores.append(info['train_r2'])
        # Utiliser le R² robuste de validation au lieu du R² de test
        if 'robust_r2' in info:
            validation_r2_scores.append(info['robust_r2'])
        else:
            validation_r2_scores.append(info['test_r2'] if not np.isnan(info['test_r2']) else -1)
    
    x_pos = np.arange(len(model_names))
    # réduire la largeur des barres pour que le plot R2 soit plus compact
    width = 0.28
    
    # Ajuster l'échelle pour inclure les R² négatifs
    min_r2 = min(min(train_r2_scores), min(validation_r2_scores))
    max_r2 = max(max(train_r2_scores), max(validation_r2_scores))
    
    bars1 = ax2.bar(x_pos - width/2, train_r2_scores, width, 
                   label='R² Entraînement (1940-2020)', color='lightgreen', alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, validation_r2_scores, width,
                   label='R² Validation (2011-2020)', color='salmon', alpha=0.8)
    
    # Ligne de référence à R²=0 (performance minimum acceptable) — affichée une seule fois plus bas
    
    # Annotations des valeurs (afficher pour toutes les barres, gérer NaN -> 'N/A')
    y_min, y_max = -5, 1.5
    # offset relatif en fraction de l'échelle pour éviter chevauchement
    rel_offset = 0.03 * (y_max - y_min)
    for i, (train_r2, val_r2) in enumerate(zip(train_r2_scores, validation_r2_scores)):
        # Formatter la valeur ou afficher N/A
        train_label = f'{train_r2:.2f}' if not np.isnan(train_r2) else 'N/A'
        val_label = f'{val_r2:.2f}' if not np.isnan(val_r2) else 'N/A'

        # Calculer position y pour le label en le clampant dans les bornes
        # On place les labels au-dessus de la barre si possible, sinon en dessous
        def label_y(pos_value, above=True):
            if np.isnan(pos_value):
                # position par défaut pour N/A:  y_max - petit offset
                base = y_max - rel_offset if above else y_min + rel_offset
            else:
                base = pos_value + rel_offset if above else pos_value - rel_offset
            # clamp
            return max(y_min + 0.01, min(y_max - 0.01, base))

        train_y = label_y(train_r2, above=True)
        val_y = label_y(val_r2, above=True)

        # Si les deux labels se chevauchent (même i), on décale l'un vers le bas
        if abs(train_y - val_y) < (rel_offset * 0.6):
            val_y = label_y(val_r2, above=False)

        ax2.text(i - width/2, train_y, train_label,
                ha='center', va='bottom' if (not np.isnan(train_r2) and train_r2 >= 0) or np.isnan(train_r2) else 'top',
                fontsize=9, fontweight='bold')
        ax2.text(i + width/2, val_y, val_label,
                ha='center', va='bottom' if (not np.isnan(val_r2) and val_r2 >= 0) or np.isnan(val_r2) else 'top',
                fontsize=9, fontweight='bold', color='darkred')
    
    # Marquer le meilleur modèle
    best_model_clean = results['best_model_name'].replace('_forte', '').replace('_moyenne', '').replace('_simple', '')
    try:
        best_idx = model_names.index(best_model_clean)
        # Ne pas recolorer la barre d'entraînement (garder le vert pour cohérence)
        # Au lieu de cela, surligner le label du modèle (xtick) en jaune/orange
        # Récupérer les étiquettes actuelles et appliquer un bbox sur celle du meilleur
        xticks = ax2.get_xticklabels()
        if 0 <= best_idx < len(xticks):
            label = xticks[best_idx]
            label.set_fontweight('bold')
            label.set_color('black')
            label.set_bbox(dict(facecolor='gold', edgecolor='orange', boxstyle='round,pad=0.2', alpha=0.9))
    except ValueError:
        print(f"[WARNING] Modele '{best_model_clean}' non trouve dans {model_names}")
    
    ax2.set_title('Performance des Modeles ML Avances', fontsize=12, fontweight='bold', pad=8)
    ax2.set_xlabel('Modèles', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Score R²', fontsize=10, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=35, ha='right', fontsize=9)
    # Déplacer la légende des R2 à droite du sous-plot (en utilisant bbox_to_anchor)
    ax2.legend(fontsize=9, bbox_to_anchor=(1.05, 0.5), loc='center left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Forcer les limites y du R² selon la demande (borne affichée -5 à 1.5)
    ax2.set_ylim(-5, 1.5)

    # Ligne de référence (R²=0) et seuil d'excellence
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
    # Placer un label latéral aligné sur la ligne R²=0 (à droite du sous-plot)
    try:
        ax2.text(1.01, 0, 'R² = 0 (baseline)', transform=ax2.get_yaxis_transform(), ha='left', va='center', fontweight='bold')
    except Exception:
        pass
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Seuil excellence (0.8)')
    
    # Pour tenir compte du placement de la légende à droite et du layout GridSpec
    # Augmenter les marges blanches en haut et en bas (plus d'espace autour de la fenêtre)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.12, hspace=0.35)
    plt.show()
    
    # Résumé textuel (affiché seulement si verbose)
    vprint("\n" + "="*70)
    vprint("RESUME ANALYSE AVANCEE")
    vprint("="*70)
    vprint(f"Zone analysee : {results['zone_name']}")
    vprint(f"Meilleur modele : {results['best_model_name']}")
    vprint("\nEXPLICATION DES METRIQUES :")
    vprint("  - R2 Train = Comment le modele APPREND sur les donnees 1940-2020")
    vprint("  - R2 Validation = Comment il PREDIT 2011-2020 (apres apprentissage sur 1940-2010)")
    vprint("  - R2 > 0 = Mieux que la moyenne | R2 < 0 = Pire que la moyenne")
    vprint("  - MAE = Erreur moyenne absolue en EUR/m2")

    vprint(f"\nRESULTATS DU MODELE {results['best_model_name']} :")
    train_r2 = results['best_model_info']['train_r2']
    vprint(f"R2 Entrainement : {train_r2:.3f} ({'Bon apprentissage' if train_r2 > 0.8 else 'Apprentissage moyen' if train_r2 > 0.5 else 'Apprentissage faible'})")

    if 'robust_r2' in results['best_model_info']:
        val_r2 = results['best_model_info']['robust_r2']
        mae = results['best_model_info'].get('robust_mae', results['best_model_info'].get('test_mae', np.nan))
        vprint(f"R2 Validation : {val_r2:.3f} ({'Predictions correctes' if val_r2 > 0 else 'PROBLEME: Pire que la moyenne !'})")
        if not np.isnan(mae):
            vprint(f"Erreur moyenne : {mae:.0f} EUR/m2")
        else:
            vprint("Erreur moyenne : N/A")

    if not np.isnan(results['best_model_info']['test_r2']):
        pred_r2 = results['best_model_info']['test_r2']
        vprint(f"R2 Predictions futures : {pred_r2:.3f} ({'Fiables' if pred_r2 > 0.5 else 'Risques' if pred_r2 > 0 else 'TRES RISQUES'})")
        if 'mape' in results['best_model_info'] and not np.isnan(results['best_model_info']['mape']):
            vprint(f"Erreur pourcentage : {results['best_model_info']['mape']:.1f}%")

    # Diagnostic global
    if 'robust_r2' in results['best_model_info']:
        r2_val = results['best_model_info']['robust_r2']
        if r2_val > 0.7:
            verdict = "MODELE FIABLE"
        elif r2_val > 0.3:
            verdict = "MODELE MOYEN"
        elif r2_val > 0:
            verdict = "MODELE FAIBLE"
        else:
            verdict = "MODELE PROBLEMATIQUE"
        vprint(f"\nVERDICT : {verdict}")
    vprint(f"Prix 2024 : {results['prix_2024']:.0f} EUR/m2")
    vprint(f"Prix 2025 : {results['prix_2025']:.0f} EUR/m2 ({((results['prix_2025']-results['prix_2024'])/results['prix_2024']*100):+.1f}%)")
    vprint(f"Prix 2026 : {results['prix_2026']:.0f} EUR/m2 ({((results['prix_2026']-results['prix_2024'])/results['prix_2024']*100):+.1f}%)")
    vprint(f"Prix 2027 : {results['prix_2027']:.0f} EUR/m2 ({((results['prix_2027']-results['prix_2024'])/results['prix_2024']*100):+.1f}%)")
    vprint(f"Evolution totale (2024-2027) : {results['variation_pct']:+.1f}%")

    if df_historique is not None:
        vprint(f"Donnees historiques : {len(df_historique)} points ({df_historique['annee'].min():.0f}-{df_historique['annee'].max():.0f})")
    vprint(f"Donnees DVF : {len(df_dvf)} points ({df_dvf['annee'].min():.0f}-{df_dvf['annee'].max():.0f})")
    vprint("="*70)

def show_results(results):
    """Interface principale pour afficher les résultats"""
    visualiser_analyse(results)