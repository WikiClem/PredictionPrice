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

insee_code['Code_commune_INSEE'] = insee_code['Code_commune_INSEE'].str.strip()
dvf['INSEE_COM'] = dvf['INSEE_COM'].astype(str).str.strip()

insee_code['Code_postal'] = insee_code['Code_postal'].str.strip()

dvf_avec_postal = dvf.merge(insee_code, left_on='INSEE_COM', right_on='Code_commune_INSEE', how='left')

dvf_avec_postal = dvf_avec_postal.drop(columns=['Code_commune_INSEE'])


dvf_avec_postal.to_csv(r'.\data\dvf_avec_code_postal.csv', sep=',', index=False)
# %%
