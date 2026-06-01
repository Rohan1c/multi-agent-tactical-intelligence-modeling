import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

df = pd.read_csv("data/players_data-2025_2026.csv")

selected_features = [

    # Availability / tactical usage
    "Age",
    "MP",
    "Starts",
    "Min",
    "Mn/Start",
    "Mn/Sub",
    "Min%",
    "Compl",
    "PPM",
    "On-Off",

    # Attacking output
    "Gls",
    "Ast",
    "G+A",
    "G-PK",
    "PK",
    "PKatt",

    # Shooting profile
    "Sh",
    "SoT",
    "Sh/90",
    "SoT/90",
    "G/Sh",

    # Creativity / progression proxies
    "Crs",
    "Fld",
    "Fls",

    # Tactical impact
    "+/-",
    "+/-90",

    # Defensive contribution
    "TklW",
    "Int",

    # Discipline
    "CrdY"
]

df = df.drop_duplicates(subset=["Player"])

for col in selected_features:

    if df[col].dtype != "object":

        df[col] = df[col].fillna(df[col].median())

df = df[["Player", "Pos"] + selected_features]

df = df.reset_index(drop=True)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df[selected_features])

input_dim = scaled_data.shape[1]

input_layer = Input(shape=(input_dim,))

encoded = Dense(64, activation="relu")(input_layer)
encoded = Dense(32, activation="relu")(encoded)
encoded = Dense(16, activation="relu")(encoded)

decoded = Dense(32, activation="relu")(encoded)
decoded = Dense(64, activation="relu")(decoded)
decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(input_layer, decoded)

encoder = Model(input_layer, encoded)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.fit(
    scaled_data,
    scaled_data,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)

latent_embeddings = encoder.predict(scaled_data)

print(latent_embeddings[:5])

np.save(
    "models/latent_embeddings.npy",
    latent_embeddings
)

encoder.save("models/encoder_model.keras")

print("Embeddings saved successfully!")
print("Encoder model saved!")

