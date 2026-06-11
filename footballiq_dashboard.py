"""
footballiq_dashboard.py
-----------------------
FootballIQ - Main Dashboard
Single entry point for the entire FootballIQ platform.

Usage:
    python footballiq_dashboard.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, os.path.join(ROOT, "visualizations"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Display helpers ───────────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def header():
    print("=" * 58)
    print("       FootballIQ  --  Football Intelligence Platform")
    print("=" * 58)

def section(title):
    print(f"\n{'─' * 58}")
    print(f"  {title}")
    print(f"{'─' * 58}")

def ask_player(prompt="Enter player name: "):
    return input(f"\n  {prompt}").strip()

def pause():
    input("\n  [Press Enter to return to menu]")


# ── 1. Similar Players ────────────────────────────────────────────────────────

def menu_similar_players():
    section("1. Similar Players")
    player = ask_player()

    from similarity_engine import get_similar_players
    results = get_similar_players(player, top_n=10)
    if not results:
        print("  No results found.")
        pause()
        return

    print(f"\n  Top players similar to {player}:\n")
    for i, (name, score) in enumerate(results, 1):
        bar = "#" * int(score * 40)
        print(f"  {i:>2}. {name:<28} {score:.3f}  {bar}")

    save = input("\n  Save similarity chart? (y/n): ").strip().lower()
    if save == "y":
        _save_similarity_chart(player, results)

    pause()


def _save_similarity_chart(player, results):
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
    path = os.path.join(OUTPUT_DIR, "similarity_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [saved] {path}")


# ── 2. Role Analysis ──────────────────────────────────────────────────────────

def menu_role_analysis():
    section("2. Role Analysis")
    player = ask_player()

    from prototype_role_engine import find_player, get_primary_role, get_role_scores
    player_row = find_player(player)
    if player_row is None:
        print("  Player not found.")
        pause()
        return

    primary = get_primary_role(player_row)
    scores  = get_role_scores(player_row)
    top5    = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

    print(f"\n  Player       : {player_row['Player']}")
    print(f"  Primary Role : {primary}\n")
    print(f"  {'Role':<28} {'Score':>6}  {'Bar'}")
    print(f"  {'─'*28} {'─'*6}  {'─'*20}")
    for role, score in top5:
        bar = "#" * int(score * 30)
        marker = "  <-- primary" if role == primary else ""
        print(f"  {role:<28} {score:>6.3f}  {bar}{marker}")

    save = input("\n  Save role probability chart? (y/n): ").strip().lower()
    if save == "y":
        _save_role_chart(player_row, top5, primary)

    pause()


def _save_role_chart(player_row, top5, primary):
    roles  = [r for r, _ in top5]
    values = [s for _, s in top5]

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(values)))
    bars = ax.barh(roles, values, color=colors, edgecolor="black", linewidth=1.2)
    ax.invert_yaxis()
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=11, fontweight="bold")
    ax.set_title(f"FootballIQ Role Analysis\n{player_row['Player']}",
                 fontsize=20, fontweight="bold", pad=20)
    ax.set_xlabel("Role Similarity Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("FootballIQ Archetypes", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.figtext(0.5, 0.02, f"Primary Role: {primary} ({values[0]:.3f})",
                ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "role_probability_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [saved] {path}")


# ── 3. Transfer Recommendations ───────────────────────────────────────────────

def menu_transfer():
    section("3. Transfer Recommendations")
    player = ask_player()

    from transfer_engine import find_player, find_replacements
    idx = find_player(player)
    if idx is None:
        print("  Player not found.")
        pause()
        return

    print(f"\n  Finding replacements for {player}...\n")
    try:
        results = find_replacements(idx, top_n=10)
        if not results:
            print("  No replacements found.")
        else:
            print(f"  {'#':<4} {'Player':<28} {'Score':>6}")
            print(f"  {'─'*4} {'─'*28} {'─'*6}")
            for i, (name, score) in enumerate(results, 1):
                print(f"  {i:<4} {name:<28} {score:>6.3f}")
    except Exception as e:
        print(f"  Error: {e}")

    pause()


# ── 4. Compatibility Analysis ─────────────────────────────────────────────────

def menu_compatibility():
    section("4. Compatibility Analysis")
    print("  Enter up to 5 players (leave blank to stop):\n")

    players = []
    for i in range(5):
        name = input(f"  Player {i+1}: ").strip()
        if not name:
            break
        players.append(name)

    if len(players) < 2:
        print("  Need at least 2 players.")
        pause()
        return

    from compatibility_engine import compatibility_score, find_player, df

    valid_players, valid_indices = [], []
    for name in players:
        idx = find_player(name)
        if idx is not None:
            valid_players.append(df.iloc[idx]["Player"])
            valid_indices.append(idx)
        else:
            print(f"  [skip] '{name}' not found.")

    if len(valid_players) < 2:
        print("  Not enough valid players.")
        pause()
        return

    n = len(valid_indices)
    print(f"\n  Compatibility Matrix:\n")
    header_row = f"  {'':28}" + "".join(f"{p[:10]:>12}" for p in valid_players)
    print(header_row)
    print(f"  {'─' * (28 + 12 * n)}")
    for i in range(n):
        row = f"  {valid_players[i]:<28}"
        for j in range(n):
            score = compatibility_score(valid_indices[i], valid_indices[j])
            row += f"{score:>12.3f}"
        print(row)

    save = input("\n  Save compatibility heatmap? (y/n): ").strip().lower()
    if save == "y":
        _save_heatmap(valid_players, valid_indices)

    pause()


def _save_heatmap(valid_players, valid_indices):
    import seaborn as sns
    from compatibility_engine import compatibility_score

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
    path = os.path.join(OUTPUT_DIR, "compatibility_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [saved] {path}")


# ── 5. Squad Builder ──────────────────────────────────────────────────────────

def menu_squad_builder():
    section("5. Squad Builder  --  Best Midfield Trio")

    from squad_builder import best_trio, print_trio
    print("\n  Building best midfield trio from dataset...\n")
    try:
        trio = best_trio()
        print_trio(trio)
    except Exception as e:
        print(f"  Error: {e}")

    pause()


# ── 6. Starting XI Builder ────────────────────────────────────────────────────

def menu_starting_xi():
    section("6. Starting XI Builder  --  4-3-3")
    player = ask_player("Build XI around player: ")

    from starting_xi_builder import find_player, build_xi, print_xi
    idx = find_player(player)
    if idx is None:
        print("  Player not found.")
        pause()
        return

    print(f"\n  Building 4-3-3 around {player}...\n")
    try:
        xi = build_xi(idx)
        print_xi(xi)
    except Exception as e:
        print(f"  Error: {e}")

    pause()


# ── 7. Intelligence Report ────────────────────────────────────────────────────

def menu_report():
    section("7. Football Intelligence Report")
    player = ask_player()

    from football_intelligence_report import generate_report
    print(f"\n  Generating report for {player}...\n")
    try:
        generate_report(player)
    except Exception as e:
        print(f"  Error: {e}")

    pause()


# ── 8. Save All Outputs ───────────────────────────────────────────────────────

def menu_save_all():
    section("8. Save All Visualizations")
    player = ask_player("Focal player for charts (default: Pedri): ")
    if not player:
        player = "Pedri"

    print(f"\n  Running save_outputs.py for {player}...\n")
    ret = os.system(f'python "{os.path.join(ROOT, "save_outputs.py")}" --player "{player}"')
    if ret != 0:
        print("  save_outputs.py encountered errors (see above).")

    pause()


# ── Main menu loop ────────────────────────────────────────────────────────────

MENU = [
    ("Similar Players",            menu_similar_players),
    ("Role Analysis",              menu_role_analysis),
    ("Transfer Recommendations",   menu_transfer),
    ("Compatibility Analysis",     menu_compatibility),
    ("Squad Builder (Midfield Trio)", menu_squad_builder),
    ("Starting XI Builder (4-3-3)",   menu_starting_xi),
    ("Football Intelligence Report",  menu_report),
    ("Save All Visualizations",       menu_save_all),
]

def main():
    while True:
        clear()
        header()
        print()
        for i, (label, _) in enumerate(MENU, 1):
            print(f"  {i}.  {label}")
        print(f"\n  0.  Exit")
        print()

        choice = input("  Select option: ").strip()

        if choice == "0":
            print("\n  Goodbye.\n")
            break

        if choice.isdigit() and 1 <= int(choice) <= len(MENU):
            clear()
            header()
            MENU[int(choice) - 1][1]()
        else:
            print("  Invalid option.")
            pause()


if __name__ == "__main__":
    main()