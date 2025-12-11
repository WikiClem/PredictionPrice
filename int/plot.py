"""
Module plot - Visualisation avancee avec predictions futures
Logique complete de code essaye.py avec graphiques ML
"""
import matplotlib.pyplot as plt
import numpy as np

def afficher_resultats_avances(resultat):
    """Affiche les resultats de la prediction ML avancee"""
    if resultat is None:
        print("❌ Données insuffisantes (minimum 5 ans requis)\n")
        return
    
    dept = resultat['departement']
    best = resultat['best_model_info']
    
    print("="*100)
    print(f"📊 PRÉDICTION ML AVANCÉE POUR LE DÉPARTEMENT {dept}")
    print("="*100)
    
    print(f"\n🧠 MEILLEUR MODÈLE : {resultat['best_model_name']}")
    print(f"  • R² sur train          : {best['train_r2']:.4f}")
    print(f"  • RMSE sur train        : {best['train_rmse']:.2f} €/m²")
    
    if not np.isnan(best['test_r2']):
        print(f"  • R² sur test           : {best['test_r2']:.4f} {'\u2705' if best['test_r2'] >= 0.7 else '\u26a0\ufe0f'}")
        print(f"  • RMSE sur test         : {best['test_rmse']:.2f} €/m²")
        
        if not np.isnan(best['direction_accuracy']):
            dir_emoji = "🎯" if best['direction_accuracy'] >= 70 else "⚠\ufe0f"
            print(f"  • Direction Accuracy    : {best['direction_accuracy']:.1f}% {dir_emoji}")
    
    print(f"\n📊 COMPARAISON DES 5 MODÈLES :")
    print(f"{'Modèle':<20} {'Test R²':<12} {'Test RMSE':<12} {'Direction':<12}")
    print("-"*60)
    for name, res in resultat['results'].items():
        r2_str = f"{res['test_r2']:.4f}" if not np.isnan(res['test_r2']) else "N/A"
        rmse_str = f"{res['test_rmse']:.2f}" if not np.isnan(res['test_rmse']) else "N/A"
        dir_str = f"{res['direction_accuracy']:.1f}%" if not np.isnan(res['direction_accuracy']) else "N/A"
        marker = "🏆" if name == resultat['best_model_name'] else "  "
        print(f"{marker} {name:<18} {r2_str:<12} {rmse_str:<12} {dir_str:<12}")
    
    print(f"\n💰 PRÉDICTIONS DU PRIX AU M² :")
    print(f"  • 2024 (référence)      : {resultat['prix_2024']:,.0f} €/m²")
    print(f"  • 2025                  : {resultat['prix_2025']:,.0f} €/m²")
    print(f"  • 2026                  : {resultat['prix_2026']:,.0f} €/m²")
    print(f"  • 2027                  : {resultat['prix_2027']:,.0f} €/m²")
    
    variation = resultat['variation_pct']
    symbole = "📈" if variation > 2 else "📉" if variation < -2 else "→"
    print(f"\n{symbole} TENDANCE 2024-2027 : {variation:+.2f}%")
    
    print("="*100 + "\n")

def creer_graphiques_complets(results):
    """Cree tous les graphiques comme dans code essaye.py"""
    try:
        if 'evolution' not in results:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Evolution des prix
        years = list(results['evolution'].keys())
        prices = list(results['evolution'].values())
        ax1.plot(years, prices, marker='o', linewidth=2, color='blue')
        ax1.set_title(f'Evolution des prix - {results["zone"]}', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Annee')
        ax1.set_ylabel('Prix moyen (euros/m2)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Repartition par type
        if 'repartition' in results:
            labels = list(results['repartition'].keys())
            sizes = list(results['repartition'].values())
            ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Repartition par type de bien', fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Donnees de repartition\nnon disponibles', 
                    horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
            ax2.set_title('Repartition par type de bien', fontsize=12, fontweight='bold')
        
        # 3. Performance des modeles
        if 'predictions' in results:
            models = list(results['predictions'].keys())
            r2_scores = [results['predictions'][model]['r2'] for model in models]
            bars = ax3.bar(models, r2_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            ax3.set_title('Performance des modeles (R2 Score)', fontsize=12, fontweight='bold')
            ax3.set_ylabel('R2 Score')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            
            # Ajout des valeurs sur les barres
            for bar, score in zip(bars, r2_scores):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, 'Donnees ML\nnon disponibles', 
                    horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
            ax3.set_title('Performance des modeles (R2 Score)', fontsize=12, fontweight='bold')
        
        # 4. Distribution des prix
        if 'prix_distribution' in results:
            ax4.hist(results['prix_distribution'], bins=50, alpha=0.7, color='green', edgecolor='black')
            ax4.set_title('Distribution des prix au m2', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Prix (euros/m2)')
            ax4.set_ylabel('Frequence')
        else:
            ax4.text(0.5, 0.5, 'Donnees de distribution\nnon disponibles', 
                    horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
            ax4.set_title('Distribution des prix au m2', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Erreur lors de la creation des graphiques: {e}")

def visualiser_prediction_avancee(resultat):
    """Visualisation simplifiee avec 2 graphiques principaux"""
    if resultat is None:
        return
    
    plt.figure(figsize=(16, 6))
    
    # Graphique 1 : Historique + Predictions futures (PRINCIPAL)
    plt.subplot(1, 2, 1)
    
    # Donnees reelles
    plt.plot(resultat['annees_train'], resultat['y_train'], 'o-', 
             color='#2E86AB', linewidth=3, markersize=8, 
             label='Train (données réelles)', alpha=0.9)
    
    if len(resultat['annees_test']) > 0:
        plt.plot(resultat['annees_test'], resultat['y_test'], '^-', 
                 color='#F18F01', linewidth=3, markersize=8, 
                 label='Test (données réelles)', alpha=0.9)
    
    # Predictions du meilleur modele
    best = resultat['best_model_info']
    plt.plot(resultat['annees_train'], best['y_train_pred'], '--', 
             color='#2E86AB', linewidth=2, label='Train (prédictions)', alpha=0.7)
    
    if len(best['y_test_pred']) > 0:
        plt.plot(resultat['annees_test'], best['y_test_pred'], '--', 
                 color='#F18F01', linewidth=2, label='Test (prédictions)', alpha=0.7)
    
    # Predictions futures (HIGHLIGHT)
    future_years = [d['annee'] for d in resultat['future_data']]
    future_prices = [d['prix_pred'] for d in resultat['future_data']]
    plt.plot(future_years, future_prices, 'D-', 
             color='#C73E1D', linewidth=4, markersize=12, 
             label='FUTUR 2025-2027', alpha=1.0)
    
    # Ligne de separation presente/futur
    plt.axvline(x=2024.5, color='red', linestyle=':', linewidth=2, alpha=0.8)
    plt.text(2024.6, max(resultat['y']) * 0.95, 'FUTUR', rotation=90, 
             fontsize=12, fontweight='bold', color='red')
    
    # R2 affiche sur le graphique
    r2_text = f"R² = {best['test_r2']:.3f}" if not np.isnan(best['test_r2']) else f"R² train = {best['train_r2']:.3f}"
    plt.text(0.02, 0.98, f"Meilleur modèle: {resultat['best_model_name']}\n{r2_text}", 
             transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
             verticalalignment='top')
    
    plt.title(f"Département {resultat['departement']} - PRÉDICTIONS 2025-2027", 
              fontsize=16, fontweight='bold')
    plt.xlabel('Année', fontsize=12)
    plt.ylabel('Prix au m² (€)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper left')
    
    # Graphique 2 : Comparaison R2 des modeles
    plt.subplot(1, 2, 2)
    model_names = list(resultat['results'].keys())
    test_r2s = [resultat['results'][name]['test_r2'] for name in model_names]
    colors = ['gold' if name == resultat['best_model_name'] else 'lightcoral' for name in model_names]
    
    bars = plt.barh(model_names, test_r2s, color=colors, alpha=0.8)
    plt.xlabel('R² sur Test', fontsize=12)
    plt.title('Performance des Modèles\n(R²)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # Afficher les valeurs R2 sur les barres
    for bar, r2 in zip(bars, test_r2s):
        if not np.isnan(r2):
            plt.text(r2 + 0.01 if r2 >= 0 else r2 - 0.01, bar.get_y() + bar.get_height()/2, 
                     f'{r2:.3f}', ha='left' if r2 >= 0 else 'right', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def show_results(results):
    """Fonction principale d'affichage avec ML avance et predictions futures"""
    try:
        if results is None:
            print("❌ Aucun résultat à afficher")
            return
        
        # Affichage textuel avance
        afficher_resultats_avances(results)
        
        # Visualisation avancee avec predictions futures
        visualiser_prediction_avancee(results)
        
    except Exception as e:
        print(f"Erreur lors de l'affichage: {e}")
        import traceback
        traceback.print_exc()