import pandas as pd

from compatibility_engine import (
    df,
    compatibility_score
)

from role_engine import (
    get_primary_role,
    get_role_scores
)

# =====================================
# PLAYER SEARCH
# =====================================

def find_player(player_name):

    exact = df[
        df["Player"].str.lower()
        ==
        player_name.lower()
    ]

    if len(exact) > 0:
        return exact.index[0]

    partial = df[
        df["Player"].str.contains(
            player_name,
            case=False,
            na=False
        )
    ]

    if len(partial) > 0:
        return partial.index[0]

    return None


# =====================================
# MIDFIELD TEMPLATES
# =====================================

MIDFIELD_TEMPLATES = {

    "Creative Playmaker": [

        "Ball Winner",
        "Box-to-Box"

    ],

    "Deep Playmaker": [

        "Creative Playmaker",
        "Box-to-Box"

    ],

    "Ball Winner": [

        "Creative Playmaker",
        "Box-to-Box"

    ],

    "Box-to-Box": [

        "Creative Playmaker",
        "Ball Winner"

    ]
}


# =====================================
# ROLE SPECIALIZATION
# =====================================

def role_specialization(idx):

    player = df.iloc[idx]

    role = get_primary_role(
        player
    )

    scores = get_role_scores(
        player
    )

    role_score = scores[
        role
    ]

    total_score = sum(
        scores.values()
    )

    if total_score == 0:
        return 0

    return (
        role_score
        /
        total_score
    )


# =====================================
# MIDFIELD QUALITY
# =====================================

def midfield_quality(idx):

    player = df.iloc[idx]

    role = get_primary_role(
        player
    )

    if role == "Creative Playmaker":

        return (

            player["Vision"]

            +

            player["Passing"]

            +

            player["Ast_norm"]

            +

            player["G+A_norm"]

        ) / 400

    elif role == "Ball Winner":

        return (

            player["TklW_norm"]

            +

            player["Int_norm"]

            +

            player["Aggression"]

            +

            player["Defending"]

        ) / 400

    elif role == "Box-to-Box":

        return (

            player["Stamina"]

            +

            player["Pace"]

            +

            player["Physicality"]

            +

            player["On-Off_norm"]

        ) / 400

    elif role == "Deep Playmaker":

        return (

            player["Passing"]

            +

            player["Vision"]

            +

            player["Compl_norm"]

            +

            player["PPM_norm"]

        ) / 400

    return 0


# =====================================
# ROLE FIT
# =====================================

def role_fit(

    anchor_role,
    candidate_role

):

    required_roles = MIDFIELD_TEMPLATES.get(

        anchor_role,

        []

    )

    if candidate_role in required_roles:
        return 1

    return 0


# =====================================
# SMART CANDIDATE SCORE
# =====================================

def candidate_score(

    anchor_idx,
    candidate_idx

):

    chemistry = compatibility_score(

        anchor_idx,

        candidate_idx

    )

    quality = midfield_quality(
        candidate_idx
    )

    specialization = role_specialization(
        candidate_idx
    )

    return (

        0.50 * chemistry

        +

        0.25 * quality

        +

        0.25 * specialization

    )
    
    # =====================================
# BUILD CANDIDATE POOLS
# =====================================

def build_candidate_pools(
    anchor_idx
):

    anchor_role = get_primary_role(
        df.iloc[anchor_idx]
    )

    required_roles = MIDFIELD_TEMPLATES.get(
        anchor_role,
        []
    )

    candidate_pools = {}

    for role in required_roles:

        candidates = []

        for idx in range(len(df)):

            if idx == anchor_idx:
                continue

            player = df.iloc[idx]

            pos = str(
                player["Pos"]
            )

            if "MF" not in pos:
                continue

            candidate_role = get_primary_role(
                player
            )

            if candidate_role != role:
                continue

            score = candidate_score(

                anchor_idx,

                idx

            )

            candidates.append(

                (
                    idx,
                    score
                )

            )

        candidates.sort(

            key=lambda x: x[1],

            reverse=True

        )

        candidate_pools[role] = [

            idx

            for idx, score

            in candidates[:20]

        ]

    return candidate_pools


# =====================================
# TRIO CHEMISTRY
# =====================================

def trio_chemistry(

    a,
    b,
    c

):

    return (

        compatibility_score(a, b)

        +

        compatibility_score(a, c)

        +

        compatibility_score(b, c)

    ) / 3


# =====================================
# FINAL TACTICAL SCORE
# =====================================

def tactical_score(

    anchor_idx,
    idx1,
    idx2

):

    chemistry = trio_chemistry(

        anchor_idx,

        idx1,

        idx2

    )

    quality = (

        midfield_quality(idx1)

        +

        midfield_quality(idx2)

    ) / 2

    specialization = (

        role_specialization(idx1)

        +

        role_specialization(idx2)

    ) / 2

    return (

        0.50 * chemistry

        +

        0.30 * quality

        +

        0.20 * specialization

    )


# =====================================
# FIND BEST TRIO
# =====================================

def best_trio(
    anchor_idx
):

    anchor_role = get_primary_role(
        df.iloc[anchor_idx]
    )

    if anchor_role not in MIDFIELD_TEMPLATES:

        return None, 0

    pools = build_candidate_pools(
        anchor_idx
    )

    role1 = MIDFIELD_TEMPLATES[
        anchor_role
    ][0]

    role2 = MIDFIELD_TEMPLATES[
        anchor_role
    ][1]

    pool1 = pools[role1]
    pool2 = pools[role2]

    best_score = -1

    best_team = None

    for p1 in pool1:

        for p2 in pool2:

            if p1 == p2:
                continue

            score = tactical_score(

                anchor_idx,

                p1,

                p2

            )

            if score > best_score:

                best_score = score

                best_team = (

                    anchor_idx,

                    p1,

                    p2

                )

    return (

        best_team,

        best_score

    )


# =====================================
# PRINT RESULT
# =====================================

def print_trio(
    trio,
    score
):

    if trio is None:

        print(
            "No valid trio found."
        )

        return

    print(
        "\nBEST TACTICAL MIDFIELD\n"
    )

    for idx in trio:

        player = df.iloc[idx]

        role = get_primary_role(
            player
        )

        print(

            f"{player['Player']} "

            f"| {player['Pos']} "

            f"| {role}"

        )

    print(
        f"\nTactical Score: "
        f"{score:.3f}"
    )


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    player_name = input(
        "Enter player name: "
    )

    player_idx = find_player(
        player_name
    )

    if player_idx is None:

        print(
            "Player not found."
        )

        exit()

    trio, score = best_trio(
        player_idx
    )

    print_trio(
        trio,
        score
    )