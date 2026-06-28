import pandas as pd
import folium
from folium.plugins import HeatMap
import numpy as np

#Load cleaned citation data (6.76M rows)
df = pd.read_parquet("citations_clean.parquet")
coords = pd.read_csv("geocoded_locations.csv")

#Load geocoding cache
coords = pd.read_csv("geocoded_locations.csv")

#Join coordinates on every ticket, ungeoded tickets get NaN coordinates
df = df.merge(coords, left_on="clean_location",right_on="address",how="left")

#Check what we got on our map
on_map = df["lat"].notna().sum()
print(f"{on_map:,} of {len(df):,} tickets geocoded ({on_map/len(df):.1%})")

# Aggreggate to per-location counts (what's plotted on the map)
# heat format: clean_location, lat, lon, and ticket_count
heat = (
    df.dropna(subset=["lat", "lon"])
      .groupby(["clean_location", "lat", "lon"])
      .size()
      .reset_index(name="ticket_count")
)
print(f"{len(heat):,} unique map points")
print(heat.sort_values("ticket_count", ascending=False).head(10))


#Now actually construct the heat map

#Center map on center of data
center = [heat["lat"].median(), heat["lon"].median()]

m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

#  SQRT-SCALE the weights — compresses 1..13000 down to ~0..9 so the whole city's relative intensity is visible
heat = heat.assign(weight=np.sqrt(heat["ticket_count"]))

heat_data = heat[["lat","lon","ticket_count"]].values.tolist()      #Lat, lon, weight per point (each address is a point)

HeatMap(
    heat_data,
    radius=10,            # smaller: tighter to actual blocks, less blobbing
    blur=6,              # less smear between points
    max_zoom=13,
    min_opacity=0.15,    # faint points still show instead of vanishing
    gradient={0.0: "blue", 0.3: "cyan", 0.5: "lime",
              0.7: "yellow", 1.0: "red"},
).add_to(m)

m.save("heatmap.html")
print("Saved heatmap.html — open it in a browser")