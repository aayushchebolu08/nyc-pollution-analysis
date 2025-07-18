import pandas as pd

# Load the file
df = pd.read_csv("Air_Quality.csv")

# Filter for NO2 and PM2.5 data from 2022
df_filtered = df[
    (df["Name"].isin(["Nitrogen Dioxide (NO2)", "Fine particles (PM 2.5)"]))
]

# Group by neighborhood and pollutant, take average values
top_pollution = (
    df_filtered.groupby(["Geo Place Name", "Name"])["Data Value"]
    .mean()
    .reset_index()
    .sort_values("Data Value", ascending=False)
)

print(top_pollution.head(10))
