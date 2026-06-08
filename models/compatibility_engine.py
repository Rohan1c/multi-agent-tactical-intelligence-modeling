import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

from role_engine import (
    get_primary_role,
    prepare_dataframe
)

df = prepare_dataframe(
    pd.read_csv(
        "data/final_merged_dataset.csv"
    )
)

embeddings = np.load(
    "models/latent_embeddings.npy"
)

distance_matrix = euclidean_distances(
    embeddings
)

max_dist = np.max(
    distance_matrix
)

similarity_matrix = (
    1 - distance_matrix / max_dist
)

# ====================================
# ROLE SYNERGY MATRIX
# ====================================

ROLE_SYNERGY = {

    "Creative Playmaker": [
        "Ball Winner",
        "Defensive Defender",
        "Box-to-Box",
        "Deep Playmaker"
    ],

    "Deep Playmaker": [
        "Ball Winner",
        "Box-to-Box",
        "Ball Playing Defender"
    ],

    "Ball Winner": [
        "Creative Playmaker",
        "Deep Playmaker"
    ],

    "Box-to-Box": [
        "Creative Playmaker",
        "Deep Playmaker"
    ],

    "Creative Winger": [
        "Target Forward",
        "Poacher",
        "Inside Forward"
    ],

    "Inside Forward": [
        "Creative Playmaker",
        "Deep Playmaker",
        "Target Forward"
    ],

    "Target Forward": [
        "Creative Winger",
        "Creative Playmaker",
        "Inside Forward"
    ],

    "Poacher": [
        "Creative Winger",
        "Creative Playmaker"
    ],

    "Ball Playing Defender": [
        "Deep Playmaker",
        "Ball Winner"
    ],

    "Defensive Defender": [
        "Ball Playing Defender",
        "Deep Playmaker"
    ]
}

# ====================================
# POSITION COMPATIBILITY
# ====================================

def position_fit(pos1, pos2):

    pos1 = str(pos1)
    pos2 = str(pos2)

    if pos1 == pos2:
        return 0.60

    if "MF" in pos1 and "DF" in pos2:
        return 1.00

    if "DF" in pos1 and "MF" in pos2:
        return 1.00

    if "MF" in pos1 and "FW" in pos2:
        return 0.95

    if "FW" in pos1 and "MF" in pos2:
        return 0.95

    if "DF" in pos1 and "FW" in pos2:
        return 0.75

    if "FW" in pos1 and "DF" in pos2:
        return 0.75

    return 0.50


# ====================================
# ROLE COMPLEMENTARITY
# ====================================

def role_compatibility(player1, player2):

    role1 = get_primary_role(player1)
    role2 = get_primary_role(player2)

    if role1 == role2:
        return 0.25

    if role1 in ROLE_SYNERGY:

        if role2 in ROLE_SYNERGY[role1]:
            return 1.00

    if role2 in ROLE_SYNERGY:

        if role1 in ROLE_SYNERGY[role2]:
            return 1.00

    return 0.50


# ====================================
# STAT COMPLEMENTARITY
# ====================================

def stat_complementarity(player1, player2):

    creativity1 = (
        player1["Vision"] +
        player1["Passing"] +
        player1["Ast_norm"]
    ) / 3

    creativity2 = (
        player2["Vision"] +
        player2["Passing"] +
        player2["Ast_norm"]
    ) / 3

    defending1 = (
        player1["Defending"] +
        player1["Interceptions"] +
        player1["TklW_norm"]
    ) / 3

    defending2 = (
        player2["Defending"] +
        player2["Interceptions"] +
        player2["TklW_norm"]
    ) / 3

    combo1 = creativity1 * defending2
    combo2 = creativity2 * defending1

    score = max(
        combo1,
        combo2
    )

    score /= 10000

    return min(
        score,
        1.0
    )


# ====================================
# MAIN SCORE
# ====================================

def compatibility_score(idx1, idx2):

    player1 = df.iloc[idx1]
    player2 = df.iloc[idx2]

    embedding_similarity = (
        similarity_matrix[idx1][idx2]
    )

    role_score = role_compatibility(
        player1,
        player2
    )

    position_score = position_fit(
        player1["Pos"],
        player2["Pos"]
    )

    stat_score = stat_complementarity(
        player1,
        player2
    )

    final_score = (

        0.25 * embedding_similarity +

        0.35 * role_score +

        0.20 * position_score +

        0.20 * stat_score

    )

    return round(
        final_score,
        4
    )


# ====================================
# BEST PARTNERS
# ====================================

def best_partners(
    player_name,
    top_n=10
):

    player_rows = df[
        df["Player"].str.contains(
            player_name,
            case=False,
            na=False
        )
    ]

    if len(player_rows) == 0:

        print(
            "Player not found."
        )

        return

    player_idx = player_rows.index[0]

    scores = []

    for idx in range(len(df)):

        if idx == player_idx:
            continue

        score = compatibility_score(
            player_idx,
            idx
        )

        scores.append(
            (
                df.iloc[idx]["Player"],
                df.iloc[idx]["Pos"],
                get_primary_role(
                    df.iloc[idx]
                ),
                score
            )
        )

    scores.sort(
        key=lambda x: x[3],
        reverse=True
    )

    print(
        "\nBEST COMPATIBLE PLAYERS\n"
    )

    for player, pos, role, score in scores[:top_n]:

        print(
            f"{player} | {pos} | {role} | {score:.3f}"
        )


if __name__ == "__main__":

    player_name = input(
        "Enter player name: "
    )

    best_partners(
        player_name
    )