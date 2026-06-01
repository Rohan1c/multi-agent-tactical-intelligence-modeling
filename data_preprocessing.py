import pandas as pd
from sklearn.preprocessing import StandardScaler

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

print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df[selected_features])

print("\nScaled Data Sample:\n")
print(scaled_data[:5])