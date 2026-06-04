# ⚽ Football Intelligence Framework

An AI-powered football analytics framework that combines representation learning, player similarity analysis, chemistry optimization, graph-based squad intelligence, and Grey Wolf Optimization (GWO) for scouting, recruitment, and tactical decision-making.

---

## 🚀 Project Overview

Traditional player comparison systems rely heavily on raw statistics and subjective scouting. This project leverages deep learning and optimization techniques to learn hidden football representations and provide intelligent recommendations for:

- Player Similarity Analysis
- Tactical Archetype Discovery
- Chemistry Optimization
- Squad Intelligence Analysis
- Scouting & Recruitment Support
- AI-Driven Squad Building

---

## 🎯 Objectives

- Learn latent football representations using ANN Autoencoders
- Identify stylistically similar players
- Discover hidden tactical archetypes
- Model player chemistry and compatibility
- Build interaction-aware football graphs
- Optimize feature importance using Grey Wolf Optimization (GWO)
- Assist recruitment and squad-building decisions

---

## 📊 Dataset

### FBRef Statistics Dataset

Contains performance metrics such as:

- Goals
- Assists
- xG
- xA
- Progressive Carries
- Progressive Passes
- Key Passes
- Tackles
- Interceptions
- Dribbles
- Minutes Played

### FC 26 Attributes Dataset

Contains player ratings such as:

- Pace
- Shooting
- Passing
- Dribbling
- Defending
- Physicality
- Vision
- Ball Control
- Composure
- Reactions
- Agility

### Hybrid Dataset

The project merges both datasets to create a unified football intelligence dataset containing statistical, technical, physical, and tactical player information.

---

## 🏗️ Methodology

### 1. Data Collection

- FBRef Statistics Dataset
- FC 26 Attributes Dataset

### 2. Data Preprocessing

- Data Cleaning
- Missing Value Handling
- Feature Selection
- Feature Scaling
- Dataset Preparation

### 3. Representation Learning

An ANN Autoencoder learns compressed latent player embeddings that capture hidden football characteristics and playstyles.

### 4. Player Similarity Engine

Cosine Similarity is applied on latent embeddings to identify players with similar football profiles.

### 5. Tactical Archetype Discovery

Player embeddings are clustered to discover hidden tactical roles and archetypes.

### 6. Chemistry Optimization

Player compatibility is calculated using:

- Embedding Similarity
- Positional Compatibility
- Tactical Complementarity
- Role Balance

### 7. Interaction Graph

- Players → Nodes
- Chemistry Scores → Weighted Edges

The graph models tactical relationships and squad-level interactions.

### 8. Grey Wolf Optimization (GWO)

Used to:

- Optimize feature weights
- Improve chemistry scoring
- Reduce manual bias
- Enhance recommendation quality

---

## 🔄 System Flow

```text
FBRef Statistics Dataset
           +
FC 26 Attributes Dataset
           ↓
      Data Merging
           ↓
    Data Preprocessing
           ↓
 ANN Autoencoder Training
           ↓
   Latent Player Embeddings
           ↓
 ┌─────────────────────┬─────────────────────┬─────────────────────┐
 │                     │                     │
 ↓                     ↓                     ↓
Player Similarity   Tactical Archetype   Chemistry Optimization
    Engine             Discovery
                                           ↓
                                  Interaction Graph
                                           ↓
                              Squad Intelligence Analysis
                                           ↓
                        Grey Wolf Optimization (GWO)
                                           ↓
                              Final Recommendations
```

---

## 🧠 Key Features

### Player Similarity Analysis

- Similar Player Discovery
- Replacement Suggestions
- Successor Identification

### Tactical Intelligence

- Latent Football Identity Learning
- Tactical Archetype Discovery
- Playstyle Analysis

### Chemistry Optimization

- Compatibility Scoring
- Best Partnerships
- Tactical Fit Analysis

### Squad Intelligence

- Graph-Based Team Analysis
- Role Coverage Analysis
- Balanced Squad Recommendations

### Optimization

- Feature Weight Optimization
- Chemistry Weight Optimization
- Improved Recommendation Quality

---

## 📈 Expected Outputs

- Top Similar Players
- Tactical Archetype Clusters
- Chemistry Scores
- Interaction Graph Visualization
- Squad Building Recommendations
- Optimized Feature Weights
- Recruitment Insights

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- TensorFlow / Keras
- NetworkX
- Matplotlib
- Grey Wolf Optimization (GWO)
- VS Code

---

## 🔬 Research Contribution

This project adapts interaction-aware representation learning concepts from multi-agent intelligence systems into football analytics by combining:

- Latent Player Embeddings
- Player Similarity Analysis
- Chemistry Optimization
- Interaction Graph Modeling
- Evolutionary Feature Optimization

The framework aims to bridge the gap between traditional football statistics and AI-driven tactical intelligence.

---

## 🔮 Future Enhancements

- Streamlit Dashboard
- Interactive Graph Visualization
- Team-Level Tactical Optimization
- Multi-Season Player Evolution Analysis
- Transfer Recommendation Engine
- Graph Neural Network (GNN) Extension

---

## 👨‍💻 Author

**Rohan Srinivas Ponnana**

B.Tech Artificial Intelligence & Machine Learning  
VIT Chennai
