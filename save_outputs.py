"""
save_outputs.py
---------------
FootballIQ - Output Saver
Saves all visualizations to outputs/ without requiring user input.

Usage:
    python save_outputs.py
    python save_outputs.py --player "Pedri"
"""

import os
import sys
import argparse
import traceback

# Resolve paths relative to this file's location
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, os.path.join(ROOT, "visualizations"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH       = os.path.join(ROOT, "data", "final_merged_dataset.csv")
EMBEDDINGS_PATH = os.path.join(ROOT, "models", "latent_embeddings.npy")


def _save(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [saved] {path}")


def _run(label, fn):
    print(f"\n>> {label}")
    try:
        fn()
    except Exception as e:
        print(f"  [skip] {label} failed: {e}")
        if os.getenv("DEBUG"):
            traceback.print_exc()
        plt.close("all")


# ── 1. Latent space ───────────────────────────────────────────────────────────
def save_latent_space(player):
    from sklearn.manifold import TSNE

    df = pd.read_csv(DATA_PATH).reset_index(drop=True)
    embeddings = np.load(EMBEDDINGS_PATH)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)

    position_colors = {
        "CB": "red",  "RB": "red",  "LB": "red",
        "CDM": "green", "CM": "green", "CAM": "green",
        "RM": "blue", "LM": "blue", "RW": "blue",
        "LW": "blue", "ST": "blue", "GK": "black"
    }
    colors = [position_colors.get(pos, "gray") for pos in df["Position"]]

    plt.figure(figsize=(15, 10))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, alpha=0.6, s=35)

    for name in ["Pedri", "Rodri", "Valverde", "Bellingham", "Haaland", "Palmer", "Vinicius", player]:
        rows = df[df["Player"].str.contains(name, case=False, na=False)]
        if len(rows):
            idx = rows.index[0]
            plt.annotate(rows.iloc[0]["Player"], (emb_2d[idx, 0], emb_2d[idx, 1]),
                         fontsize=10, fontweight="bold")

    plt.title("FootballIQ Latent Embedding Space", fontsize=18, fontweight="bold")
    plt.xlabel("TSNE Dimension 1")
    plt.ylabel("TSNE Dimension 2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    _save("latent_space.png")


# ── 2. Similarity chart ───────────────────────────────────────────────────────
def save_similarity_chart(player):
    from similarity_engine import get_similar_players

    results = get_similar_players(player, top_n=10)
    if not results:
        print(f"  [skip] No similar players found for {player}")
        return

    names  = [p for p, _ in results]
    scores = [s for _, s in results]

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0.25, 0.95, len(scores)))
    bars = ax.barh(names, scores, color=colors, edgecolor="black", linewidth=1.2)
    ax.invert_yaxis()

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=10, fontweight="bold")

    ax.set_title(f"FootballIQ Similarity Analysis\nTop Similar Players to {player}",
                 fontsize=20, fontweight="bold", pad=20)
    ax.set_xlabel("Similarity Score", fontsize=13, fontweight="bold")
    ax.set_ylabel("Players", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.figtext(0.5, 0.01, f"Closest Match: {names[0]} ({scores[0]:.3f})",
                ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save("similarity_chart.png")


# ── 3. Compatibility heatmap ──────────────────────────────────────────────────
def save_compatibility_heatmap(player):
    import seaborn as sns
    from compatibility_engine import compatibility_score, find_player, df

    from similarity_engine import get_similar_players
    similar = [n for n, _ in get_similar_players(player, top_n=4)]
    players = [player] + similar

    valid_players, valid_indices = [], []
    for name in players:
        idx = find_player(name)
        if idx is not None:
            valid_players.append(df.iloc[idx]["Player"])
            valid_indices.append(idx)

    n = len(valid_indices)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = compatibility_score(valid_indices[i], valid_indices[j])

    plt.figure(figsize=(12, 9))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlGn",
                linewidths=1, square=True,
                xticklabels=valid_players, yticklabels=valid_players,
                cbar_kws={"label": "Compatibility Score"})
    plt.title("FootballIQ Player Compatibility Matrix", fontsize=18, fontweight="bold")
    plt.tight_layout()
    _save("compatibility_heatmap.png")


# ── 4. Squad network ──────────────────────────────────────────────────────────
def save_squad_network(player):
    import networkx as nx
    from compatibility_engine import compatibility_score, find_player, df
    from prototype_role_engine import get_primary_role

    from similarity_engine import get_similar_players
    similar = [n for n, _ in get_similar_players(player, top_n=5)]
    players = [player] + similar

    G = nx.Graph()
    player_roles = {}
    for name in players:
        idx = find_player(name)
        if idx is None:
            continue
        role = get_primary_role(df.iloc[idx])
        player_roles[name] = role
        G.add_node(name)

    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            idx1 = find_player(players[i])
            idx2 = find_player(players[j])
            if idx1 is None or idx2 is None:
                continue
            score = compatibility_score(idx1, idx2)
            if score >= 0.67:
                G.add_edge(players[i], players[j], weight=score)

    role_colors = {
        "Creative Playmaker": "#FFD700", "Deep Playmaker": "#00BFFF",
        "Ball Winner": "#FF4D4D",        "Box-to-Box": "#32CD32",
        "Creative Winger": "#FF69B4",    "Wide Winger": "#BA55D3",
        "Inside Forward": "#FF8C00",     "Poacher": "#FFA500",
        "False 9": "#9370DB",            "Target Forward": "#8B4513",
        "Ball Playing Defender": "#20B2AA", "Defensive Defender": "#708090",
        "Goalkeeper": "#000000"
    }

    node_colors = [role_colors.get(player_roles.get(n, ""), "#A9A9A9") for n in G.nodes()]
    node_sizes  = [1800 + G.degree(n) * 700 for n in G.nodes()]
    pos     = nx.kamada_kawai_layout(G)
    weights = [G[u][v]["weight"] * 10 for u, v in G.edges()]

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           edgecolors="black", linewidths=2, alpha=0.95)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7)
    nx.draw_networkx_edge_labels(G, pos,
        edge_labels={(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()},
        font_size=8)

    legend_items = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=10, label=r)
        for r, c in role_colors.items()
    ]
    plt.legend(handles=legend_items, bbox_to_anchor=(1.05, 1),
               loc="upper left", fontsize=8)
    plt.title("FootballIQ Squad Compatibility Network", fontsize=20, fontweight="bold")
    plt.figtext(0.5, 0.02,
        "Node Size = Connectivity | Edge Weight = Compatibility Score | Node Color = Role",
        ha="center", fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    _save("squad_network.png")


# ── 5. Role probability chart ─────────────────────────────────────────────────
def save_role_probability_chart(player):
    from prototype_role_engine import find_player, get_role_scores

    player_row = find_player(player)
    if player_row is None:
        print(f"  [skip] Player '{player}' not found.")
        return

    scores = get_role_scores(player_row)
    top_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    roles  = [r for r, _ in top_roles]
    values = [s for _, s in top_roles]

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(values)))
    bars = ax.barh(roles, values, color=colors, edgecolor="black", linewidth=1.2)
    ax.invert_yaxis()

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=11, fontweight="bold")

    ax.set_title(f"FootballIQ Role Analysis\n{player}", fontsize=20,
                 fontweight="bold", pad=20)
    ax.set_xlabel("Role Similarity Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("FootballIQ Archetypes", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.figtext(0.5, 0.02,
                f"Primary Role: {top_roles[0][0]} ({top_roles[0][1]:.3f})",
                ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save("role_probability_chart.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Save all FootballIQ visualizations to outputs/"
    )
    parser.add_argument("--player", type=str, default="Pedri",
                        help="Focal player for charts (default: Pedri)")
    args = parser.parse_args()
    player = args.player

    print(f"\nFootballIQ Output Saver")
    print(f"Focal player : {player}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print("=" * 50)

    _run("Latent space",           lambda: save_latent_space(player))
    _run("Similarity chart",       lambda: save_similarity_chart(player))
    _run("Compatibility heatmap",  lambda: save_compatibility_heatmap(player))
    _run("Squad network",          lambda: save_squad_network(player))
    _run("Role probability chart", lambda: save_role_probability_chart(player))

    saved = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")]
    print(f"\n{'='*50}")
    print(f"Done. {len(saved)} figure(s) in outputs/")
    for f in sorted(saved):
        print(f"  - {f}")


if __name__ == "__main__":
    main()