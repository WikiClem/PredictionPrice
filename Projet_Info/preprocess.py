#%%
## data annee complete
import pandas as pd
import os


# Define the folder path containing the yearly data files
folder_path = r'.\data\data_annee'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Initialize an empty list to hold DataFrames
dfs = []

# Read each CSV file and append to the list
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, sep=',')  # Adjust sep if needed
    
    # Nettoyer les noms de colonnes (enlever guillemets et espaces)
    df.columns = df.columns.str.replace('"', '').str.strip()
    
    # Si la première colonne n'est pas INSEE_COM (ou est vide), la supprimer
    first_col = df.columns[0]
    if first_col != 'INSEE_COM' and (first_col == '' or 'Unnamed' in first_col):
        df = df.drop(columns=[first_col])
    
    # Uniformiser les noms de colonnes (majuscules/minuscules)
    df.columns = df.columns.str.replace('Annee', 'annee')
    df.columns = df.columns.str.replace('Nb_mutations', 'nb_mutations')
    df.columns = df.columns.str.replace('propmaison', 'PropMaison')
    df.columns = df.columns.str.replace('propappart', 'PropAppart')
    
    dfs.append(df)

# Concatenate all DataFrames
dvf_fusion = pd.concat(dfs, ignore_index=True)

# Save the merged DataFrame
dvf_fusion.to_csv(r'.\data\dvf_fusion.csv', sep=',', index=False, encoding='utf-8')


#%%
## code postal epuree
import pandas as pd


df = pd.read_csv(r'.\data\019HexaSmal.csv', sep=';', encoding='latin-1')

# Nettoyer les noms de colonnes (enlever # et espaces)
df.columns = df.columns.str.replace('#', '').str.strip()

# Convertir Code_postal en texte, nettoyer les espaces et ajouter 0 devant si besoin (format 5 chiffres)
df['Code_postal'] = df['Code_postal'].astype(str).str.strip().str.zfill(5)

# Sélectionner uniquement les colonnes Code_commune_INSEE et Code_postal
df_selectionne = df[['Code_commune_INSEE', 'Code_postal']].copy()

# Supprimer les doublons uniquement sur Code_commune_INSEE (garde la première occurrence)
df_unique = df_selectionne.drop_duplicates(subset=['Code_commune_INSEE'], keep='first')

# Nettoyer aussi Code_commune_INSEE pour enlever les espaces
df_unique['Code_commune_INSEE'] = df_unique['Code_commune_INSEE'].astype(str).str.strip()

# Sauvegarder le résultat EN FORÇANT le type string pour éviter la conversion en int
df_unique.to_csv(r'.\data\INSEE_code.csv', sep=';', index=False, encoding='utf-8')

#%%
## fusion de code_postal et data complete
import pandas as pd


dvf = pd.read_csv(r'.\data\dvf_fusion.csv', sep=',')

insee_code = pd.read_csv(r'.\data\INSEE_code.csv', sep=';', dtype={'Code_postal': str})

# Nettoyer les données et filtrer les codes INSEE vides
insee_code['Code_commune_INSEE'] = insee_code['Code_commune_INSEE'].str.strip()
dvf['INSEE_COM'] = dvf['INSEE_COM'].astype(str).str.strip()

# Filtrer les lignes avec des codes INSEE vides ou invalides
dvf = dvf[dvf['INSEE_COM'] != '']
dvf = dvf[dvf['INSEE_COM'].notna()]

insee_code['Code_postal'] = insee_code['Code_postal'].str.strip()

# Faire la jointure (left join pour garder toutes les lignes DVF)
dvf_avec_postal = dvf.merge(insee_code, left_on='INSEE_COM', right_on='Code_commune_INSEE', how='left')

dvf_avec_postal = dvf_avec_postal.drop(columns=['Code_commune_INSEE'])

# Afficher un résumé des codes postaux manquants
codes_postaux_manquants = dvf_avec_postal['Code_postal'].isna().sum()
print(f"Codes postaux manquants après jointure: {codes_postaux_manquants} sur {len(dvf_avec_postal)} lignes")

# Forcer Code_postal en string avant sauvegarde pour éviter les .0
dvf_avec_postal['Code_postal'] = dvf_avec_postal['Code_postal'].astype(str).str.replace('.0', '').str.zfill(5)

dvf_avec_postal.to_csv(r'.\data\dvf_avec_code_postal.csv', sep=',', index=False)
# %%
## rajout nom departement
import pandas as pd

dvf = pd.read_csv(r'.\data\dvf_avec_code_postal.csv', sep=',', dtype={'Code_postal': str})
departement = pd.read_csv(r'.\data\departements-france.csv', sep=',')

# Nettoyer les noms de colonnes (enlever les espaces en début/fin)
dvf.columns = dvf.columns.str.strip()

# NETTOYER TOUTES les colonnes texte (enlever espaces début/fin) SAUF Code_postal
for col in dvf.columns:
    if dvf[col].dtype == 'object' and col != 'Code_postal':
        dvf[col] = dvf[col].astype(str).str.strip()

# Traitement spécial pour Code_postal : garder le format exact avec zéros de début
if 'Code_postal' in dvf.columns:
    dvf['Code_postal'] = dvf['Code_postal'].astype(str).str.strip().str.zfill(5)

# Extraire les 2 premiers chiffres du code INSEE_COM pour obtenir le code département
dvf['code_departement'] = dvf['INSEE_COM'].astype(str).str[:2]

# Nettoyer les données départements
departement['code_departement'] = departement['code_departement'].astype(str).str.strip()

# Faire la jointure pour ajouter TOUTES les colonnes du département
dvf_avec_departement = dvf.merge(departement, on='code_departement', how='left')

# Convertir code_region en entier à 2 chiffres (gérer les NaN)
dvf_avec_departement['code_region'] = pd.to_numeric(dvf_avec_departement['code_region'], errors='coerce').astype('Int64')

# Supprimer la colonne temporaire code_departement
dvf_avec_departement = dvf_avec_departement.drop(columns=['code_departement'])

# Sauvegarder le résultat final
dvf_avec_departement.to_csv(r'.\data\dvf.csv', sep=',', index=False)
# %%
## Conversion Excel valeur-immobilier en CSV (France uniquement)
import pandas as pd
import os

# Lire le fichier Excel valeur-immobilier
print("Traitement du fichier valeur-immobilier...")
excel_file = r'.\data\valeur-immobilier-1800-2020_cle2abd1f.xls'

# Lire les données à partir de la ligne 15 (où commencent les vraies données)
df_excel = pd.read_excel(excel_file, skiprows=15)

# Identifier les colonnes d'années (colonnes numériques entre 1800 et 2020)
year_columns = [col for col in df_excel.columns if isinstance(col, (int, float)) and 1800 <= col <= 2020]

# Filtrer les lignes concernant la France (contenant "France" dans la première colonne)
france_mask = df_excel.iloc[:, 0].astype(str).str.contains('France|france', na=False, case=False)
df_france = df_excel[france_mask].copy()

# Nettoyer et préparer les données
first_col_name = df_france.columns[0]  # Nom de la première colonne
df_france_clean = df_france[[first_col_name] + year_columns].copy()

# Séparer les séries et ne garder que l'indice des prix des logements
indice_prix_row = None
for index, row in df_france_clean.iterrows():
    serie_name = row[first_col_name]
    if 'Indice du prix des logements' in str(serie_name):
        indice_prix_row = row
        break

# Convertir en format long (une ligne par année) - seulement l'indice des prix
df_long_list = []
if indice_prix_row is not None:
    for year in year_columns:
        value = indice_prix_row[year]
        if pd.notna(value):
            df_long_list.append({
                'Annee': int(year),
                'Valeur': float(value)
            })
        else:
            # Pour les années sans données, mettre 0
            df_long_list.append({
                'Annee': int(year),
                'Valeur': 0.0
            })
else:
    print("[ERROR] Serie 'Indice du prix des logements' non trouvee!")
    for year in year_columns:
        df_long_list.append({
            'Annee': int(year),
            'Valeur': 0.0
        })

df_valeur_immobilier = pd.DataFrame(df_long_list)

# Sauvegarder le fichier CSV
df_valeur_immobilier.to_csv(r'.\data\valeur_immobilier_france.csv', sep=',', index=False, encoding='utf-8')
print(f"Fichier sauvegardé: {len(df_valeur_immobilier)} lignes de données France")

# %%
