"""
Programme principal - Analyseur de prix immobilier
Architecture modulaire adaptee du code essaye.py fonctionnel
"""
# Import des modules personnalises
from utils import analyze_data
from plot import show_results

def main():
    """Programme principal - copie de la logique de code essaye.py"""
    print("ANALYSEUR DE PRIX IMMOBILIER")
    print("="*50)
    print("Entrez :")
    print("- Un code postal (5 chiffres) -> analyse du departement")
    print("- Un nom de departement -> analyse du departement")
    print("- Un nom de region -> analyse de la region")
    print("- 'FRANCE' -> analyse nationale")
    print("="*50)
    
    while True:
        try:
            user_input = input("Votre choix : ").strip()
            
            if user_input.upper() in ['EXIT', 'QUIT', 'Q', 'SORTIR']:
                print("Au revoir !")
                break
            
            if not user_input:
                print("Veuillez entrer une valeur valide.")
                continue
            
            # Analyse des donnees
            print(f"\nAnalyse en cours pour: {user_input}")
            results = analyze_data(user_input)
            
            if results:
                # Affichage des resultats
                show_results(results)
                print("\n" + "-"*50)
                print("Analyse terminee avec succes !")
                print("Tapez 'QUIT' pour quitter ou une nouvelle zone pour continuer.")
                print("-"*50)
            else:
                print(f"Aucun resultat trouve pour: {user_input}")
                print("Verifiez l'orthographe ou essayez:")
                print("- FRANCE")
                print("- 75 (pour Paris)")
                print("- 13 (pour les Bouches-du-Rhone)")
                
        except KeyboardInterrupt:
            print("\nProgramme interrompu par l'utilisateur.")
            print("Au revoir !")
            break
        except Exception as e:
            print(f"Erreur: {e}")
            print("Conseil: Verifiez que le fichier 'data/dvf.csv' existe")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()