"""
MAIN - PROJET INFO 3
Interface utilisateur pour l'analyse combinant :
- Données historiques de 200 ans
- Données DVF récentes 2014-2024
Utilise l'historique pour améliorer les prédictions ML
"""

from utils import analyze_data, analyze_postal
from plot import show_results

def main():
    """Interface principale du système"""                          # visuel du terminal
    print("*" + "="*68 + "*")
    print("ANALYSEUR IMMOBILIER AVANCE")
    print("DONNEES 200 ANS + DVF RECENTES")
    print("*" + "="*68 + "*")
    print()
    print("Ce systeme combine :")
    print("   - Donnees historiques de 1936-2020")
    print("   - Donnees DVF recentes 2014-2024")
    print("   - Predictions futures 2025-2027")
    print()
    
    try:
        while True:
            print("-" * 70)
            print("COMMANDES DISPONIBLES :")
            print("   - 'FRANCE' ou un code postal : Analyse de la France ou d'un département")
            print("   - 'QUIT' : Quitter le programme")
            print("-" * 70)
            
            user_input = input("Votre choix : ").strip()
            print()
            
            # Gestion des commandes de sortie
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Au revoir ! Merci d'avoir utilise l'analyseur immobilier.")
                break
            
            # Analyse France
            if user_input.upper() == 'FRANCE' or user_input.lower() == 'france':
                print("Lancement de l'analyse France...")
                print()
                
                # Lancer l'analyse nationale
                results = analyze_data(user_input)
                
                if results:
                    print("Analyse terminée avec succès !")
                    print()
                    
                    # Afficher les graphiques
                    show_results(results)
                    
                else:
                    print("Échec de l'analyse.")
            
            else:
                # Si l'utilisateur fournit un code postal ou numéro de département
                s = user_input.strip()
                if s.isdigit() and (len(s) == 5 or len(s) <= 3):
                    print(f"Lancement de l'analyse locale pour : {s}...")
                    results = analyze_postal(s)
                    if results:
                        print("Analyse locale terminée !")
                        show_results(results)
                    else:
                        print("Échec de l'analyse locale. Pas assez de données ou erreur.")
                else:
                    print("Commande non reconnue.")
                    print("Utilisez 'FRANCE' ou un code postal (5 chiffres) ou 'QUIT'.")
            
            print()
            
    except KeyboardInterrupt:
        print("\\n  Programme interrompu par l'utilisateur. \\n  Au revoir !")
    except Exception as e:
        print(f"\\n  Erreur inattendue : {e}")
        print("  Veuillez relancer le programme.")

if __name__ == "__main__":
    main()