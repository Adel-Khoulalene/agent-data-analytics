# 📊 Agent Data Analytics — Multi-CSV & LLM Autonome

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agent-data-analytics-test-fichier-csv.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-121212?style=flat)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-API_Inference-f05032?style=flat)](https://groq.com/)
[![DuckDB](https://img.shields.io/badge/DuckDB-In_Memory_DB-FFF000?style=flat&logo=duckdb&logoColor=black)](https://duckdb.org/)

Application web interactive basée sur un **agent IA autonome** capable d'analyser dynamiquement des jeux de données (CSV, TXT, LOG) à l'aide du langage naturel, de générer des requêtes SQL à la volée via **DuckDB** et de tracer automatiquement des visualisations graphiques.

---

## 🚀 Fonctionnalités Clés

* **Ingestion & Parsing Intelligents** :
  * Détection et conversion automatique des formats de dates complexes (`YYYYMMDD`, formats mixtes `DD/MM/YYYY`, etc.).
  * Gestion automatique des valeurs manquantes (`NA`, `NaN`), des délimiteurs multiples (`;`, `,`, tabulation) et nettoyage des guillemets.
* **Agent Autonome (LangGraph + Groq)** :
  * Génération dynamique de requêtes SQL DuckDB à partir de questions en langage naturel.
  * Boucle d'auto-correction de requêtes SQL en cas d'erreur de syntaxe ou d'exécution.
* **Visualisation Dynamique** :
  * Génération automatique de graphiques (courbes, barres, nuages de points) selon les résultats extraits.
  * Support multi-séries (ex: comparaison par station ou catégorie).
* **Jeux de Données & Requêtes de Démonstration** :
  * Boutons de téléchargement direct de jeux de données d'exemple dans la sidebar.
  * Requêtes de démo pré-remplies et exécutables en un clic.

---

## 🛠️ Architecture Technique
┌─────────────────┐     ┌───────────────────────┐     ┌──────────────────┐
│  Fichier Client │ ──> │ Ingestion & Nettoyage │ ──> │ Base de Données  │
│   (CSV / TXT)   │     │       (Pandas)        │     │  DuckDB (Memory) │
└─────────────────┘     └───────────────────────┘     └─────────┬────────┘
                                                                │
                                                                ▼
┌─────────────────┐     ┌───────────────────────┐     ┌──────────────────┐
│ Visualisation   │ <── │   Execution SQL       │ <── │  Agent LangGraph │
│  (Matplotlib)   │     │      (DuckDB)         │     │    (Groq LLM)    │
└─────────────────┘     └───────────────────────┘     └──────────────────┘

---

## 🛠️ Tech Stack

* **Interface Utilisateur** : Streamlit
* **Analyse & Traitement de Données** : Pandas, NumPy, DuckDB
* **Orchestration IA & LLM** : LangChain, LangGraph, Groq API (`openai/gpt-oss-120b`)
* **Data Visualization** : Matplotlib
* **Déploiement** : Streamlit Cloud

---

## 💻 Installation & Lancement Local

### 1. Cloner le dépôt
```bash
git clone [https://github.com/Adel-Khoulalene/agent-data-analytics.git](https://github.com/Adel-Khoulalene/agent-data-analytics.git)
cd agent-data-analytics

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé API Groq

```bash
GROQ_API_KEY = "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"
```

### 5.Lancer l'application

```bash
streamlit run app.py
```
