import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

df = pd.read_csv(
    "data/final_merged_dataset.csv"
)

embeddings = np.load(
    "models/latent_embeddings.npy"
)

df = df.drop_duplicates(
    subset=["Player"]
).reset_index(drop=True)

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30
)

emb_2d = tsne.fit_transform(
    embeddings
)

plt.figure(figsize=(12,8))

plt.scatter(
    emb_2d[:,0],
    emb_2d[:,1],
    alpha=0.6
)

sample_players = [

    "Pedri",
    "Rodri",
    "Haaland",
    "Valverde",
    "Bellingham",
    "Saliba",
    "Palmer"

]

for player in sample_players:

    rows = df[
        df["Player"]
        .str.contains(
            player,
            case=False,
            na=False
        )
    ]

    if len(rows):

        idx = rows.index[0]

        plt.annotate(

            rows.iloc[0]["Player"],

            (
                emb_2d[idx,0],
                emb_2d[idx,1]
            )

        )

plt.title(
    "FootballIQ Latent Embedding Space"
)

plt.tight_layout()

plt.show()