# ⚽ FootballIQ: Player Analytics and Recruitment Intelligence System

An AI-powered football intelligence framework that combines deep-learning player embeddings, role archetype discovery, similarity analysis, transfer recommendations, compatibility modeling, squad optimization, and tactical lineup generation using FBref statistics and FIFA attributes.

---

## 🚀 Project Overview

FootballIQ is a football analytics and scouting platform designed to assist clubs, analysts, recruiters, and enthusiasts in evaluating players, discovering replacements, building squads, and analyzing tactical fit.

The system leverages:

- Autoencoder-based representation learning
- Latent player embeddings
- Role archetype modeling
- Similarity learning
- Compatibility analysis
- Transfer recommendation systems
- Tactical squad optimization

The project integrates statistical performance data from FBref with FIFA player attributes to generate meaningful football intelligence.

---

## 🏗️ System Architecture

```text
Dataset
│
├── FBref Statistics
├── FIFA Attributes
│
▼

Data Preprocessing
│
▼

Autoencoder Training
│
▼

Latent Embeddings
│
├── Similarity Engine
├── Compatibility Engine
├── Recruitment Assistant
├── Transfer Engine
├── Role Archetype Engine
├── Squad Builder
├── Starting XI Builder
└── Football Intelligence Reports
```

---

# 📊 Dataset

The project combines:

### FBref Statistics

- Goals
- Assists
- Expected Goals (xG)
- Expected Assists (xA)
- Passing Metrics
- Defensive Actions
- Progressive Actions
- Playing Time

### FIFA Attributes

- Pace
- Shooting
- Passing
- Dribbling
- Defending
- Physicality
- Vision
- Finishing
- Ball Control
- Aggression
- Strength
- Positioning
- Interceptions
- Tackling

### Dataset Size

```text
~2057 Players
```

---

# 🧠 Machine Learning Pipeline

## Autoencoder

The core representation learning model uses an autoencoder to compress player feature vectors into latent embeddings.

### Purpose

- Capture player style
- Reduce dimensionality
- Learn hidden football characteristics
- Enable similarity search

### Output

```text
latent_embeddings.npy
```

---

# 🔍 Similarity Engine

Finds players with similar profiles based on latent embeddings.

### Example

Input:

```text
Pedri
```

Output:

```text
1. Frenkie de Jong
2. Martin Ødegaard
3. Nicolò Barella
4. Warren Zaïre-Emery
5. Federico Valverde
```

---

# 🎭 Role Archetype Engine

Classifies players into football archetypes using prototype-based role discovery.

### Supported Roles

#### Midfield

- Deep Playmaker
- Creative Playmaker
- Ball Winner
- Box-to-Box

#### Attack

- Wide Winger
- Creative Winger
- Inside Forward
- False 9
- Poacher
- Target Forward

#### Defense

- Ball Playing Defender
- Defensive Defender

### Example

Input:

```text
Rodri
```

Output:

```text
Primary Role:
Deep Playmaker
```

---

# 🤝 Compatibility Engine

Measures how well two players complement each other.

Factors considered:

- Latent embedding similarity
- Role fit
- Positional compatibility
- Tactical balance

### Example

```text
Pedri + Rodri
Compatibility Score: 0.94
```

---

# 🔄 Transfer Recommendation Engine

Identifies realistic replacements for a player.

Scoring Components:

```text
40% Embedding Similarity
30% Role Match
20% Position Match
10% Age Profile
```

### Example

Input:

```text
Pedri
```

Output:

```text
1. João Neves
2. Gavi
3. Warren Zaïre-Emery
4. Kobbie Mainoo
5. Andrey Santos
```

---

# 🏗️ Squad Builder

Constructs balanced midfield combinations based on:

- Compatibility
- Tactical roles
- Positional fit
- Chemistry

### Example

```text
Pedri
Martin Ødegaard
Joshua Kimmich

Tactical Score: 0.946
```

---

# ⚽ Starting XI Builder

Generates a complete tactical lineup around a selected player.

### Formation

```text
4-3-3
```

### Example Output

```text
GK  Jordan Pickford

RB  Konrad Laimer
CB  Jan Paul van Hecke
CB  Marcos Senesi
LB  Marc Cucurella

CDM Moisés Caicedo
CM  Dominik Szoboszlai
CM  Enzo Le Fée

RW  Ridle Baku
ST  Florian Sotoca
LW  Sergio Gómez
```

---

# 📈 Football Intelligence Reports

Aggregates insights from all modules.

### Includes

- Role Analysis
- Similar Players
- Compatible Partners
- Transfer Replacements
- Tactical Recommendations
- Squad Construction Suggestions

---

# 🛠️ Tech Stack

### Languages

- Python

### Machine Learning

- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-Learn

### Visualization

- NetworkX
- Matplotlib

### Optimization

- Grey Wolf Optimization (GWO)

---

# 📂 Project Structure

```text
FootballIQ
│
├── data/
│
├── models/
│   ├── autoencoder.py
│   ├── similarity_engine.py
│   ├── compatibility_engine.py
│   ├── prototype_role_engine.py
│   ├── recruitment_assistant.py
│   ├── transfer_engine.py
│   ├── squad_builder.py
│   ├── starting_xi_builder.py
│   ├── football_intelligence_report.py
│   └── latent_embeddings.npy
│
├── data_preprocessing.py
├── dataset_merge.py
├── requirements.txt
└── README.md
```

---

# 🎯 Future Improvements

- Streamlit Dashboard
- Interactive Player Search
- Formation Optimization
- Transfer Budget Constraints
- Market Value Prediction
- Role Discovery using Unsupervised Learning
- Explainable AI for Recommendations
- Multi-Season Player Development Tracking

---

# 📜 License

This project is released under the MIT License.

---

# 👨‍💻 Author

**Rohan Srinivas Ponnana**

Artificial Intelligence & Machine Learning Student

Football Analytics • Machine Learning • Recruitment Intelligence • Representation Learning

---

⭐ If you found this project interesting, consider starring the repository.
