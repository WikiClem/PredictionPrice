Présentation du projet

Ce projet vise à réaliser une prédiction des prix de l’immobilier à moyen terme, sur la période 2014–2024. L’objectif principal est d’apprendre et de comprendre le fonctionnement des modèles de machine learning appliqués à des données réelles.



Présentation des données

Nous utilisons en premier lieu une base de données annuelle contenant des indicateurs immobiliers par commune :

data_annee : https://www.data.gouv.fr/datasets/indicateurs-immobiliers-par-commune-et-par-annee-prix-et-volumes-sur-la-periode-2014-2024/     

019HexaSmal.csv : https://www.data.gouv.fr/datasets/base-officielle-des-codes-postaux/     

departement-france.csv : https://www.data.gouv.fr/datasets/departements-de-france/      

valeur-immobilier-1800-2020_cle2abd1f.xls : https://www.data.gouv.fr/datasets/valeurs-immobilieres-economiques-et-financieres-de-1800-a-2020/     



Deux modèles de régression sont utilisés dans ce projet : Lasso, Ridge

Ces modèles permettent de limiter le surapprentissage tout en identifiant les variables les plus pertinentes pour la prédiction des prix immobiliers.

Pour pouvoir utiliser les fonctionnalité, il faut télécharger tout les fichier de data_annee du gouvernement de chaque année, puis lance le process ... 

-------------------------------------------------------------------     

structure demandée :     

projet_info/     
├── data/     
│ ├── data_annee/     
│ │ ├── dvf2014.csv     
│ │ ├── ...     
│ ├── 019HexaSmal.csv     
│ ├── departement-france.csv     
│ ├── valeur-immobilier-1800-2020_cle2abd1f.xls     
├── preprocess.py     
├── utils.py     
├── plot.py     
├── main.py     
├── READ_ME.md     

-------------------------------------------------------------------     

Lancez preprocess     

Puis lancer main et écrire dans le terminal    
