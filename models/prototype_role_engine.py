import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(
    "data/final_merged_dataset.csv"
)


# =====================================
# FEATURES
# =====================================

FEATURE_COLS = [

    "Pace",
    "Shooting",
    "Passing",
    "Dribbling",
    "Defending",
    "Physicality",

    "Vision",
    "Crossing",
    "Finishing",
    "Ball Control",

    "Interceptions",
    "Standing Tackle",

    "Strength",
    "Aggression"

]


# =====================================
# NORMALIZE
# =====================================

scaler = MinMaxScaler()

df[FEATURE_COLS] = scaler.fit_transform(
    df[FEATURE_COLS]
)


# =====================================
# POSITION GATING
# =====================================

POSITION_ROLE_MAP = {

    "GK": [

        "Goalkeeper"

    ],

    "CB": [

        "Ball Playing Defender",
        "Defensive Defender"

    ],

    "RB": [

        "Ball Playing Defender",
        "Wide Winger"

    ],

    "LB": [

        "Ball Playing Defender",
        "Wide Winger"

    ],

    "CDM": [

        "Deep Playmaker",
        "Ball Winner"

    ],

    "CM": [

        "Deep Playmaker",
        "Box-to-Box",
        "Creative Playmaker"

    ],

    "CAM": [

        "Creative Playmaker",
        "False 9"

    ],

    "RW": [

        "Wide Winger",
        "Inside Forward"

    ],

    "LW": [

        "Wide Winger",
        "Inside Forward"

    ],

    "RM": [

        "Wide Winger",
        "Creative Winger"

    ],

    "LM": [

        "Wide Winger",
        "Creative Winger"

    ],

    "ST": [

        "Poacher",
        "Target Forward",
        "False 9"

    ]
}


# =====================================
# EXEMPLARS
# =====================================

ROLE_EXEMPLARS = {

    "Deep Playmaker": [

        "Rodri",
        "Kimmich"

    ],

    "Creative Playmaker": [

        "Kevin De Bruyne",
        "Pedri",
        "Bernardo Silva"

    ],

    "Ball Winner": [

        "Caicedo",
        "Declan Rice",
        "Tchouameni"

    ],

    "Box-to-Box": [

        "Valverde",
        "Bellingham"

    ],

    "Wide Winger": [

        "Raphinha",
        "Saka"

    ],

    "Creative Winger": [

        "Wirtz",
        "Palmer"

    ],

    "Inside Forward": [

        "Son",
        "Kvaratskhelia"

    ],

    "Poacher": [

        "Haaland",
        "Kane"

    ],

    "Target Forward": [

        "Lukaku",
        "Osimhen"

    ],

    "False 9": [

        "Benzema",
        "Griezmann"

    ],

    "Ball Playing Defender": [

        "Van Dijk",
        "Saliba"

    ],

    "Defensive Defender": [

        "Milenkovic",
        "Tomori"

    ]
}

# =====================================
# BUILD PROTOTYPES
# =====================================

def build_prototypes(verbose=False):

    prototypes = {}

    if verbose:
        print("\nROLE CENTROIDS\n")
    

    for role, names in ROLE_EXEMPLARS.items():

        vectors = []

        found = 0

        for name in names:

            player_rows = df[
                df["Player"]
                .str.contains(
                    name,
                    case=False,
                    na=False
                )
            ]

            if len(player_rows) == 0:
                continue

            found += 1

            player = player_rows.iloc[0]

            vectors.append(
                player[
                    FEATURE_COLS
                ].values
            )

        if verbose:
            print(
                f"{role}: "
                f"{found}/{len(names)}"
            )

        if len(vectors) == 0:
            continue

        centroid = np.mean(
            vectors,
            axis=0
        )

        prototypes[
            role
        ] = centroid

    return prototypes


# =====================================
# GLOBAL PROTOTYPES
# =====================================

ROLE_PROTOTYPES = (
    build_prototypes(verbose=False)
)


# =====================================
# ROLE SCORES
# =====================================

def get_role_scores(player):

    position = str(
        player["Position"]
    )

    if position == "GK":

        return {
            "Goalkeeper": 1.0
        }

    allowed_roles = (
        POSITION_ROLE_MAP
        .get(
            position,
            list(
                ROLE_PROTOTYPES.keys()
            )
        )
    )

    player_vector = player[
        FEATURE_COLS
    ].values.reshape(
        1,
        -1
    )

    scores = {}

    for role in allowed_roles:

        if role not in ROLE_PROTOTYPES:
            continue

        prototype = (
            ROLE_PROTOTYPES[
                role
            ]
            .reshape(
                1,
                -1
            )
        )

        similarity = (
            cosine_similarity(
                player_vector,
                prototype
            )[0][0]
        )

        scores[
            role
        ] = float(
            similarity
        )

    return scores


# =====================================
# PRIMARY ROLE
# =====================================

def get_primary_role(player):

    scores = get_role_scores(
        player
    )

    if len(scores) == 0:

        return "Undefined"

    best_role = max(
        scores,
        key=scores.get
    )

    best_score = scores[
        best_role
    ]

    # confidence threshold

    if best_score < 0.80:

        return "Undefined"

    return best_role


# =====================================
# FIND PLAYER
# =====================================

def find_player(player_name):

    exact = df[
        df["Player"]
        .str.lower()
        ==
        player_name.lower()
    ]

    if len(exact) > 0:

        return exact.iloc[0]

    partial = df[
        df["Player"]
        .str.contains(
            player_name,
            case=False,
            na=False
        )
    ]

    if len(partial) > 0:

        return partial.iloc[0]

    return None


# =====================================
# TESTER
# =====================================

if __name__ == "__main__":

    player_name = input(
        "Enter player name: "
    )

    player = find_player(
        player_name
    )

    if player is None:

        print(
            "Player not found."
        )

        exit()

    scores = (
        get_role_scores(
            player
        )
    )

    sorted_scores = sorted(

        scores.items(),

        key=lambda x: x[1],

        reverse=True

    )

    print("\n")
    print(
        f"Player: "
        f"{player['Player']}"
    )

    print(
        f"Position: "
        f"{player['Position']}"
    )

    print(
        "\nTop Prototype Roles:\n"
    )

    for role, score in (
        sorted_scores[:5]
    ):

        print(
            f"{role:<25}"
            f"{score:.4f}"
        )

    print("\n")
    print(
        "Primary Role:"
    )

    print(
        get_primary_role(
            player
        )
    )