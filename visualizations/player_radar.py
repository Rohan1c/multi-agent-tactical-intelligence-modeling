import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

player_name = input(
    "Enter player: "
)

df = pd.read_csv(
    "data/final_merged_dataset.csv"
)

player = df[
    df["Player"]
    .str.contains(
        player_name,
        case=False,
        na=False
    )
]

if len(player) == 0:

    print("Player not found")
    exit()

player = player.iloc[0]

features = [

    "Pace",
    "Shooting",
    "Passing",
    "Dribbling",
    "Defending",
    "Physicality"

]

values = [

    player[f]

    for f in features

]

values += values[:1]

angles = np.linspace(
    0,
    2*np.pi,
    len(features),
    endpoint=False
).tolist()

angles += angles[:1]

fig = plt.figure(
    figsize=(8,8)
)

ax = plt.subplot(
    111,
    polar=True
)

ax.plot(
    angles,
    values,
    linewidth=2
)

ax.fill(
    angles,
    values,
    alpha=0.25
)

ax.set_xticks(
    angles[:-1]
)

ax.set_xticklabels(
    features
)

plt.title(
    player["Player"]
)

plt.show()