import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example Polars DataFrame (you can replace with your real data)
df = pl.DataFrame({
    "t": list(range(21)),
    "n(0,t)": [0,10,10,10,10,10,9,11.1,15.6,20.6,25.2,28.3,29.4,25.3,21,13.4,3.9,0,0,0,0],
    "n(1,t)": [0,0,10,10,10,13.3,21.1,26.3,28.5,29.4,29.8,29.9,23.3,18.9,16.7,15.7,15.3,9.2,0,0,0],
    "n(2,t)": [0,0,0,10,20,26.7,28.9,29.6,29.9,30,30,20,16.7,15.6,15.2,15.1,15,15,14.2,4.2,0]
}, strict=False)

# Convert to Pandas for plotting
pdf = df.to_pandas()

# Set 't' as index
pdf.set_index('t', inplace=True)

# Plot
plt.figure(figsize=(8,6))
sns.heatmap(pdf, annot=True, fmt=".1f", cmap="RdYlGn_r", cbar=False)
plt.savefig("heatmap.png")