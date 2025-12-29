Présentation du projet

Ce projet vise à réaliser une prédiction des prix de l’immobilier à moyen terme, sur la période 2014–2024. L’objectif principal est d’apprendre et de comprendre le fonctionnement des modèles de machine learning appliqués à des données réelles.

Présentation des données

Nous utilisons en premier lieu une base de données annuelle contenant des indicateurs immobiliers par commune :

data_annee
Source : https://www.data.gouv.fr/datasets/indicateurs-immobiliers-par-commune-et-par-annee-prix-et-volumes-sur-la-periode-2014-2024/

Cette base fournit les prix et volumes immobiliers par commune et par année.

Afin de pouvoir réaliser des prédictions à l’échelle départementale, nous utilisons également des jeux de données complémentaires permettant de faire le lien entre les codes INSEE, les codes postaux et les départements :
019HexaSmal.csv – Base officielle des codes postaux
https://www.data.gouv.fr/datasets/base-officielle-des-codes-postaux/

departement-france.csv – Liste des départements français
https://www.data.gouv.fr/datasets/departements-de-france/

Enrichissement des données historiques

019HexaSmal.csv – Base officielle des codes postaux
https://www.data.gouv.fr/datasets/base-officielle-des-codes-postaux/

departement-france.csv – Liste des départements français
https://www.data.gouv.fr/datasets/departements-de-france/

Enrichissement des données historiques

Modèles utilisés

Deux modèles de régression sont utilisés dans ce projet :

Lasso

Ridge

Ces modèles permettent de limiter le surapprentissage tout en identifiant les variables les plus pertinentes pour la prédiction des prix immobiliers.

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
├── requirement.txt     

-------------------------------------------------------------------     

Lancez preprocess     

Puis lancer main et écrire dans le terminal     


