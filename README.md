# deep-learning: Multi-Architecture Medical Analysis


Ce projet implémente trois architectures distinctes de Deep Learning en "PyTorch" pour résoudre des problématiques médicales critiques : la classification des maladies cardiaques (données tabulaires), la détection de la pneumonie (imagerie médicale) et la prédiction des pulsations cardiaques (séries temporelles).



## Structure Globale du Projet

Le projet combine trois approches méthodologiques complémentaires :

1. Partie I (MLP) : Classification binaire des maladies cardiaques à partir de données cliniques tabulaires (`heart.csv`).
2. Partie II (CNN) : Détection de la pneumonie par analyse de radiographies thoraciques (`chest_xray`).
3. Partie III (LSTM) : Régression et prédiction à court terme du rythme cardiaque à partir de signaux séquentiels.



##  Installations et Configuration

- Prérequis
* Python 3.8+
* PyTorch (avec support CUDA si disponible)

- Cloner le projet
```bash

##  Organisation du code

├── data/
│   ├── heart.csv             # Dataset tabulaire
│   └── chest_xray/           # Images radiographiques (Train/Test)
├── models/
│   ├── best_mlp_model.pth    # Poids optimaux du MLP
│   ├── best_cnn_model.pth    # Poids optimaux du CNN
│   └── best_rnn_model.pth    # Poids optimaux du LSTM
├── src/
│   ├── preprocessing.py      # Pipelines de traitement des données
│   ├── architectures.py      # Classes des modèles PyTorch
│   └── train.py              # Boucles d'entraînement et checkpoints
├── README.md
└── requirements.txt









