C'est un projet pour prédire le prix de l'imobilier en France      
       

Doc : https://docs.google.com/document/d/1lc0KLD5IluRxGSs1nl_7jPiohgQPNFfB2Za29JhYynk/edit?usp=drive_link    
    

data_annee : https://www.data.gouv.fr/datasets/indicateurs-immobiliers-par-commune-et-par-annee-prix-et-volumes-sur-la-periode-2014-2024/     

019HexaSmal.csv : https://www.data.gouv.fr/datasets/base-officielle-des-codes-postaux/    

departement-france.csv : https://www.data.gouv.fr/datasets/departements-de-france/
      
structure demandée :

projet_info/     
├── data/                 
│   ├── data_annee/    
│   │   ├── dvf2014.csv       
│   │   ├── ...     
│   ├── 019HexaSmal.csv      
│   ├── departement-france.csv      
│   ├── valeur-immobilier-1800-2020_cle2abd1f.xls      
├── preprocess.py      
├── utils.py      
├── plot.py       
├── main.py      
├── requirement.txt      


voici la data base de 200 ans utilisée pour la verification du fonctionnement des prédiction : https://www.data.gouv.fr/datasets/valeurs-immobilieres-economiques-et-financieres-de-1800-a-2020/
