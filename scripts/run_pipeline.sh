#!/usr/bin/env bash

set -euo pipefail

echo "======================================"
echo "🚀 Lancement pipeline spatial ESA-grade"
echo "======================================"

# Vérification de Python
if ! command -v python3 &> /dev/null
then
    echo "❌ Python3 non trouvé"
    exit 1
fi

# Affichage version
echo "🐍 Python version : $(python3 --version)"

# Option : environnement virtuel (si présent)
if [ -d "venv" ]; then
    echo "🔧 Activation de l'environnement virtuel"
    source venv/bin/activate
fi

# Timestamp début
START_TIME=$(date +%s)

echo "▶️ Exécution du pipeline..."

# Lancement du pipeline
python3 main.py

# Timestamp fin
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "--------------------------------------"
echo "✅ Pipeline terminé avec succès"
echo "⏱️ Durée : ${DURATION}s"
echo "--------------------------------------"