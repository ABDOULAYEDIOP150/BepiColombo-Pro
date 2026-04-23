# BepiColombo DBSC – Pipeline de validation réaliste

## Résumé

Pipeline de calibration de données instrumentales inspiré de la mission spatiale **BepiColombo (ESA / JAXA)**.

Ce projet implémente les étapes clés d’un traitement scientifique :

- Analyse fréquentielle (FFT)
- Filtrage du signal (Butterworth, Savitzky-Golay, Combo)
- Calibration (suppression de dérive + correction instrumentale)
- Validation physique des données
- Comparaison inter-instruments (DBSC vs MAG)

🎯 **Objectif :** reproduire un pipeline réaliste de calibration de données issues d’instruments spatiaux.

---

## 🎯 Alignement avec le poste

Ce projet répond directement aux problématiques du poste :

- Calibration de l’instrument DBSC  
- Validation de la qualité des données  
- Comparaison entre instruments  
- Automatisation du traitement  
- Structuration d’un pipeline scientifique en Python  

---

## 1. Vue d’ensemble du projet

Ce projet propose un pipeline complet de traitement et de validation de signaux simulés dans un contexte spatial et instrumental inspiré de **BepiColombo DBSC / MAG**.

L’objectif est de partir d’une **donnée brute instrumentale** contenant à la fois :
- une composante physique utile,
- du bruit haute fréquence,
- des données manquantes,
- une dérive instrumentale lente,
- des artefacts de mesure,

puis de construire un pipeline reproductible permettant de :
1. observer et comprendre la donnée brute,
2. analyser son contenu fréquentiel,
3. tester plusieurs méthodes de filtrage,
4. calibrer les signaux filtrés,
5. valider les résultats obtenus,
6. comparer plusieurs instruments,
7. retenir le meilleur modèle final.

Ce dépôt a une double vocation :
- **scientifique et ingénierie** : analyser et traiter un signal de capteur ;
- **documentaire** : fournir un cadre clair, lisible et maintenable pour des utilisateurs techniques et non techniques.

---

## 2. Objectif du pipeline

Dans un contexte industriel ou spatial, un capteur ne mesure presque jamais un signal pur.  
La mesure observée est généralement un **signal mélangé** qui combine :

- le phénomène physique réel que l’on souhaite étudier,
- un bruit aléatoire lié à la chaîne de mesure,
- une dérive lente liée à l’instrument,
- des valeurs manquantes,
- parfois des points aberrants,
- des biais d’offset ou de gain.

Le but du pipeline est donc de répondre à la question suivante :

> **Comment rapprocher un signal brut mesuré du signal physique réel à l’aide de méthodes robustes de filtrage, de calibration, de validation et de comparaison inter-instruments ?**

---

## 3. Organisation générale du pipeline

Le pipeline suit les étapes suivantes :

1. **Données brutes**  
2. **FFT – analyse fréquentielle**  
3. **Filtrage – benchmark de plusieurs méthodes**  
4. **Calibration – benchmark final des signaux filtrés**  
5. **Validation – contrôle des plages physiques**  
6. **Comparaison inter-instruments – DBSC vs MAG**

Chaque étape produit :
- des figures,
- des métriques,
- une interprétation,
- une justification du choix méthodologique.

---

# 4. Description détaillée des étapes

## 4.1. Données brutes

### But
Comprendre la structure du signal avant tout traitement.

### Ce que l’on observe
Les données brutes simulées contiennent :
- `temperature` : signal mesuré par l’instrument ;
- `physical_signal` : signal physique de référence simulé ;
- `instrument_temperature` : température instrumentale ;
- `calibration_flag` : indicateur de qualité ;
- des attributs de qualité tels que :
  - le pourcentage de données manquantes,
  - le nombre d’artefacts injectés.

### Pourquoi cette étape est importante
Avant de filtrer ou de calibrer, il faut d’abord **qualifier** la donnée :
- le signal brut suit-il globalement la dynamique physique attendue ?
- le bruit est-il faible ou marqué ?
- la dérive instrumentale est-elle visible ?
- le signal contient-il des anomalies ponctuelles ?
- quelle est la différence entre un jeu propre et un jeu réaliste ?

### Sorties typiques
- affichage de la structure `xarray.Dataset`,
- tableau de comparaison avec le dataset de référence,
- métriques de qualité,
- figures de décomposition :
  - signal brut,
  - signal physique,
  - dérive estimée,
  - bruit haute fréquence estimé.

### Interprétation
Cette étape montre que la mesure brute n’est pas directement exploitable.  
Elle doit être analysée comme un **signal composite**.

---

## 4.2. FFT – Analyse fréquentielle

### But
Étudier le contenu fréquentiel du signal brut avant filtrage.

### Méthode utilisée
On calcule une **FFT (Fast Fourier Transform)** du signal moyen spatial.

### Pourquoi cette étape est importante
La FFT permet de :
- identifier les fréquences dominantes du signal ;
- distinguer les composantes utiles du bruit ;
- proposer une fréquence de coupure initiale cohérente pour les filtres ;
- vérifier si le signal contient plusieurs composantes fréquentielles.

### Ce que l’on extrait
- le spectre fréquentiel,
- les pics dominants,
- la fréquence dominante maximale utile,
- une fréquence de coupure proposée pour le filtrage.

### Interprétation
Si plusieurs pics sont présents, le signal n’est pas monofréquentiel.  
S’il existe en plus un fond spectral diffus, cela indique la présence de bruit.

### Décision d’ingénierie
La FFT ne corrige rien.  
Elle sert à **guider le filtrage**, en particulier pour les approches de type passe-bas comme Butterworth.

---

## 4.3. Filtrage – Benchmark des méthodes

### But
Comparer plusieurs méthodes de filtrage et retenir celle qui reconstruit le mieux le signal physique.

### Méthodes utilisées dans le projet

#### A. Butterworth
Filtre passe-bas classique.

**Principe :**
- conserve les basses fréquences utiles ;
- atténue les hautes fréquences associées au bruit.

**Paramètres étudiés :**
- ordre du filtre ;
- fréquence de coupure.

**Choix des paramètres :**
- guidé par la FFT ;
- validé par exploration de plusieurs combinaisons.

**Avantages :**
- interprétation physique claire ;
- cohérent avec une analyse fréquentielle ;
- efficace pour réduire le bruit haute fréquence.

**Limites :**
- dépend du bon choix de la coupure ;
- peut lisser excessivement le signal si le réglage est trop agressif.

---

#### B. Savitzky–Golay
Filtrage par ajustement polynomial local.

**Principe :**
- ajustement local d’un polynôme sur une fenêtre glissante ;
- lissage du signal en préservant sa forme.

**Paramètres étudiés :**
- taille de fenêtre ;
- ordre polynomial.

**Avantages :**
- préserve bien la forme locale ;
- utile pour lisser sans casser totalement les oscillations.

**Limites :**
- moins directement lié au contenu fréquentiel ;
- sensible au choix de la fenêtre.

---

#### C. Combo
Méthode hybride :
1. Butterworth,
2. puis Savitzky–Golay.

**Principe :**
- Butterworth réduit le bruit fréquentiel ;
- Savitzky–Golay affine ensuite le lissage local.

**Avantages :**
- combine deux logiques complémentaires ;
- utile si une seule méthode ne suffit pas.

**Limites :**
- plus complexe ;
- plus de paramètres ;
- pas forcément meilleur si Butterworth seul est déjà optimal.

---

### Comment le benchmark est réalisé
Chaque méthode est évaluée à partir :
- de la **MAE**,
- de la **RMSE**,
- de la **corrélation** avec le signal réel,
- d’un **score global** synthétique.

### Résultat attendu
Le meilleur modèle de filtrage est celui qui :
- diminue les erreurs,
- augmente la corrélation,
- conserve la structure utile du signal,
- réduit le bruit sans déformer la dynamique.

---

## 4.4. Calibration – Benchmark final

### But
Corriger les effets instrumentaux résiduels après filtrage.

### Principe
On ne calibre pas directement le signal brut.  
On calibre les **signaux déjà filtrés**, car ils sont plus stables et plus interprétables.

### Méthode de calibration utilisée
La calibration repose sur :
- l’estimation d’une **dérive lente** au cours du temps ;
- sa suppression par ajustement polynomial ;
- l’application d’un **gain** et d’un **offset** instrumentaux.

### Paramètres de calibration
- degré du polynôme de dérive ;
- option de préservation de la moyenne ;
- paramètres de gain et d’offset définis par instrument.

### Pourquoi c’est utile
Même un signal bien filtré peut encore présenter :
- un biais moyen,
- une dérive lente,
- une mauvaise restitution absolue du niveau du signal.

La calibration corrige donc la partie **instrumentale** qui ne relève pas du simple bruit.

### Métriques étudiées
Pour chaque modèle filtré calibré, on calcule :
- MAE calibré → réel,
- RMSE calibré → réel,
- corrélation calibré → réel,
- gain de calibration par rapport au signal filtré,
- dérive avant calibration,
- dérive après calibration,
- dérive supprimée.

### Ce que signifie “meilleur modèle final”
Le meilleur modèle final n’est pas forcément le meilleur filtre brut.  
C’est celui qui, **après calibration**, donne le meilleur compromis entre :
- fidélité au signal réel,
- stabilité,
- réduction de la dérive.

---

## 4.5. Validation

### But
Vérifier que les données calibrées restent physiquement plausibles.

### Méthode utilisée
Validation sur une plage admissible de température :
- **minimum : -50 °C**
- **maximum : +100 °C**

### Indicateurs de validation
- minimum,
- maximum,
- moyenne,
- écart-type,
- conformité à la plage autorisée.

### Pourquoi cette étape est importante
Un signal peut être :
- corrélé au signal réel,
- bien filtré,
- bien calibré,

mais malgré cela sortir des plages physiques admissibles.  
La validation contrôle donc la **cohérence physique** du résultat final.


NB : A choisir sur la base d'une référence.


## 4.6. Comparaison inter-instruments – DBSC vs MAG

### But
Comparer les signaux calibrés issus de deux instruments différents.

### Métriques utilisées
- MAE,
- RMSE,
- corrélation.

### Pourquoi cette étape est utile
Dans un pipeline multi-instrument, il ne suffit pas qu’un signal soit bon individuellement.  
Il faut aussi vérifier que plusieurs instruments produisent des résultats **cohérents entre eux**.

### Interprétation
- corrélation élevée : instruments cohérents ;
- corrélation moyenne : cohérence acceptable mais à surveiller ;
- corrélation faible : calibration, bruit ou modèle à revoir.

---

# 5. Résumé des méthodes utilisées

## 5.1. Méthodes principales

| Étape | Méthode | Rôle |
|---|---|---|
| Données brutes | Analyse descriptive | Comprendre la structure du signal |
| FFT | Transformée de Fourier | Identifier les fréquences dominantes |
| Filtrage | Butterworth | Réduire le bruit haute fréquence selon une logique fréquentielle |
| Filtrage | Savitzky-Golay | Lisser localement le signal |
| Filtrage | Combo | Combiner coupure fréquentielle et lissage local |
| Calibration | Retrait de dérive polynomial + gain/offset | Corriger les effets instrumentaux |
| Validation | Contrôle de plage physique | Vérifier la plausibilité physique |
| Comparaison | Comparaison inter-instruments | Vérifier la cohérence globale |

---

## 5.2. Différences entre les méthodes de filtrage

### Butterworth
- logique fréquentielle ;
- cohérent avec la FFT ;
- bon choix quand la fréquence de coupure peut être justifiée.

### Savitzky-Golay
- logique locale et géométrique ;
- utile pour lisser sans perdre complètement la forme ;
- moins directement lié à une interprétation fréquentielle.

### Combo
- méthode hybride ;
- plus flexible ;
- utile lorsqu’un double traitement améliore la qualité finale.

---

# 6. Architecture du dépôt

Le dépôt est organisé de la manière suivante :

```text
BepiColombo-Pro/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── docker.yml
├── scripts/
│   └── run_pipeline.sh
├── src/
│   ├── __init__.py
│   ├── calibration.py
│   ├── comparison.py
│   ├── fft_analysis.py
│   ├── filtering.py
│   ├── logger.py
│   ├── pipeline.py
│   ├── simulation.py
│   └── validation.py
├── style/
│   └── theme.css
├── test/
│   ├── test_calibration.py
│   └── test_filtering.py
├── .dockerignore
├── Dockerfile
├── main.py
├── README.md
├── requirements.txt
└── streamlit_app.py
![CI](https://github.com/ABDOULAYEDIOP150/BepiColombo-Pro/actions/workflows/ci.yml/badge.svg)
![Docker](https://github.com/ABDOULAYEDIOP150/BepiColombo-Pro/actions/workflows/docker.yml/badge.svg)
