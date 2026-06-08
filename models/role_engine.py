import pandas as pd
import numpy as np

# =====================================
# DATA PREPARATION
# =====================================

def prepare_dataframe(df):

    stat_columns = [

        "Ast",
        "Gls",
        "G+A",

        "Compl",
        "PPM",

        "Int",
        "TklW",

        "Crs",

        "Fld",
        "Fls",

        "Sh/90",
        "SoT/90",

        "+/-",
        "+/-90",

        "On-Off"

    ]

    for col in stat_columns:

        if col not in df.columns:
            continue

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        mn = df[col].min()
        mx = df[col].max()

        if pd.isna(mn) or pd.isna(mx):

            df[f"{col}_norm"] = 0

        elif mx == mn:

            df[f"{col}_norm"] = 0

        else:

            df[f"{col}_norm"] = (

                (df[col] - mn)

                /

                (mx - mn)

            ) * 100

    return df


df = prepare_dataframe(
    pd.read_csv(
        "data/final_merged_dataset.csv"
    )
)

# =====================================
# ROLE SCORING
# =====================================

def get_role_scores(player):

    scores = {}

    # =================================
    # DEEP PLAYMAKER
    # =================================

    scores["Deep Playmaker"] = (

        0.15 * player["Passing"]

        +

        0.10 * player["Vision"]

        +

        0.10 * player["Long Passing"]

        +

        0.10 * player["Compl_norm"]

        +

        0.10 * player["PPM_norm"]

        +

        0.10 * player["Ast_norm"]

        +

        0.10 * player["On-Off_norm"]

        +

        0.10 * player["+/-90_norm"]

    )

    # =================================
    # CREATIVE PLAYMAKER
    # =================================

    scores["Creative Playmaker"] = (

        0.20 * player["Vision"]

        +

        0.10 * player["Passing"]

        +

        0.20 * player["Ast_norm"]

        +

        0.15 * player["PPM_norm"]

        +

        0.10 * player["Crs_norm"]

        +

        0.10 * player["Dribbling"]

        +

        0.10 * player["Ball Control"]

        +

        0.15 * player["G+A_norm"]

        +

        0.05 * player["Agility"]

    )

    # =================================
    # BALL WINNER
    # =================================

    scores["Ball Winner"] = (

        0.25 * player["TklW_norm"]

        +

        0.25 * player["Int_norm"]

        +

        0.10 * player["Defending"]

        +

        0.10 * player["Aggression"]

        +

        0.15 * player["On-Off_norm"]

        +

        0.15 * player["+/-90_norm"]

    )

    # =================================
    # BOX TO BOX
    # =================================

    scores["Box-to-Box"] = (

        0.15 * player["Stamina"]

        +

        0.15 * player["Pace"]

        +

        0.10 * player["Physicality"]

        +

        0.10 * player["Aggression"]

        +

        0.10 * player["Passing"]

        +

        0.10 * player["Dribbling"]

        +

        0.10 * player["TklW_norm"]

        +

        0.10 * player["Int_norm"]

        +

        0.10 * player["On-Off_norm"]

    )

    # =================================
    # BALL PLAYING DEFENDER
    # =================================

    scores["Ball Playing Defender"] = (

        0.20 * player["Defending"]

        +

        0.15 * player["Passing"]

        +

        0.15 * player["Vision"]

        +

        0.10 * player["Long Passing"]

        +

        0.10 * player["Composure"]

        +

        0.10 * player["Compl_norm"]

        +

        0.10 * player["Int_norm"]

        +

        0.10 * player["TklW_norm"]

    )

    # =================================
    # DEFENSIVE DEFENDER
    # =================================

    scores["Defensive Defender"] = (

        0.25 * player["Defending"]

        +

        0.20 * player["Interceptions"]

        +

        0.15 * player["Standing Tackle"]

        +

        0.10 * player["Sliding Tackle"]

        +

        0.10 * player["Strength"]

        +

        0.10 * player["Int_norm"]

        +

        0.10 * player["TklW_norm"]

    )
    
        # =================================
    # CREATIVE WINGER
    # =================================

    scores["Creative Winger"] = (

        0.15 * player["Pace"]

        +

        0.15 * player["Dribbling"]

        +

        0.15 * player["Crossing"]

        +

        0.15 * player["Vision"]

        +

        0.10 * player["Ast_norm"]

        +

        0.10 * player["Crs_norm"]

        +

        0.10 * player["G+A_norm"]

        +

        0.10 * player["Agility"]

    )

    # =================================
    # INSIDE FORWARD
    # =================================

    scores["Inside Forward"] = (

        0.20 * player["Pace"]

        +

        0.20 * player["Dribbling"]

        +

        0.15 * player["Finishing"]

        +

        0.10 * player["Positioning"]

        +

        0.10 * player["Shot Power"]

        +

        0.15 * player["G+A_norm"]

        +

        0.10 * player["SoT/90_norm"]

    )

    # =================================
    # TARGET FORWARD
    # =================================

    scores["Target Forward"] = (

        0.15 * player["Finishing"]

        +

        0.15 * player["Strength"]

        +

        0.15 * player["Physicality"]

        +

        0.15 * player["Heading Accuracy"]

        +

        0.10 * player["Positioning"]

        +

        0.15 * player["Gls_norm"]

        +

        0.15 * player["Sh/90_norm"]

    )

    # =================================
    # POACHER
    # =================================

    scores["Poacher"] = (

        0.20 * player["Finishing"]

        +

        0.15 * player["Positioning"]

        +

        0.15 * player["Shot Power"]

        +

        0.10 * player["Composure"]

        +

        0.20 * player["Gls_norm"]

        +

        0.10 * player["SoT/90_norm"]

        +

        0.10 * player["Sh/90_norm"]

    )

    return scores


# =====================================
# PRIMARY ROLE
# =====================================

def get_primary_role(player):

    scores = get_role_scores(
        player
    )

    pos = str(
        player["Pos"]
    )

    if "DF" in pos and "MF" not in pos:

        allowed_roles = [

            "Ball Playing Defender",
            "Defensive Defender"

        ]

    elif "MF" in pos and "FW" not in pos:

        allowed_roles = [

            "Deep Playmaker",
            "Creative Playmaker",
            "Ball Winner",
            "Box-to-Box"

        ]

    elif "FW" in pos and "MF" not in pos:

        allowed_roles = [

            "Creative Winger",
            "Inside Forward",
            "Target Forward",
            "Poacher"

        ]

    else:

        allowed_roles = list(
            scores.keys()
        )

    filtered_scores = {

        role: score

        for role, score

        in scores.items()

        if role in allowed_roles

    }

    role = max(
        filtered_scores,
        key=filtered_scores.get
    )

    defending = player["Defending"]
    passing = player["Passing"]
    vision = player["Vision"]

    stamina = player["Stamina"]
    pace = player["Pace"]

    aggression = player["Aggression"]

    # Rice / Caicedo

    if (

        defending > 80

        and

        aggression > 75

        and

        player["TklW_norm"] > 40

    ):

        return "Ball Winner"

    # Valverde / Llorente

    if (

        stamina > 80

        and

        pace > 78

        and

        aggression > 70
        
        and 
        
        player["TklW_norm"] > 35

    ):

        return "Box-to-Box"

    # Rodri

    if (

        passing > 85

        and

        vision > 85

        and

        defending > 80

        and

        player["Compl_norm"] > 60

    ):

        return "Deep Playmaker"

    return role


# =====================================
# TOP ROLES
# =====================================

def get_top_roles(
    player,
    top_n=5
):

    scores = get_role_scores(
        player
    )

    pos = str(
        player["Pos"]
    )

    if "DF" in pos and "MF" not in pos:

        allowed = [

            "Ball Playing Defender",
            "Defensive Defender"

        ]

    elif "MF" in pos and "FW" not in pos:

        allowed = [

            "Deep Playmaker",
            "Creative Playmaker",
            "Ball Winner",
            "Box-to-Box"

        ]

    elif "FW" in pos and "MF" not in pos:

        allowed = [

            "Creative Winger",
            "Inside Forward",
            "Target Forward",
            "Poacher"

        ]

    else:

        allowed = list(
            scores.keys()
        )

    filtered = {

        role: score

        for role, score

        in scores.items()

        if role in allowed

    }

    return sorted(

        filtered.items(),

        key=lambda x: x[1],

        reverse=True

    )[:top_n]


# =====================================
# CLI TEST
# =====================================

if __name__ == "__main__":

    player_name = input(
        "Enter player name: "
    )

    rows = df[
        df["Player"].str.contains(
            player_name,
            case=False,
            na=False
        )
    ]

    if len(rows) == 0:

        print(
            "Player not found."
        )

        exit()

    player = rows.iloc[0]

    print(
        f"\nPlayer: {player['Player']}"
    )

    print(
        f"Position: {player['Pos']}"
    )

    print(
        "\nTop Roles:\n"
    )

    for role, score in get_top_roles(
        player
    ):

        print(
            f"{role:<25} {round(score,2)}"
        )

    print(
        "\nPrimary Role:"
    )

    print(
        get_primary_role(
            player
        )
    )