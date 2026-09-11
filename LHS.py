import pandas as pd
from scipy.stats import qmc

# Number of parameters (dimensions)
d = 6
# Rule of thumb: 50 samples per parameter
n_sims = d * 50

# Parameter bounds based on your previous grid
# Order: [ob, oc, mnu, h, w0, wa]
l_bounds = [0.04, 0.20, 0.00, 61.0, -1.3, -0.6]
u_bounds = [0.06, 0.34, 0.15, 73.0, -0.7,  0.6]

# Generate the Latin Hypercube (seeded for reproducibility)
sampler = qmc.LatinHypercube(d=d, seed=42)
sample = sampler.random(n=n_sims)

# Scale the 0-1 samples to your actual cosmological bounds
scaled = qmc.scale(sample, l_bounds, u_bounds)

# Format into a DataFrame, round to 5 decimals for clean terminal output
df = pd.DataFrame(scaled, columns=["ob", "oc", "mnu", "h", "w0", "wa"])
df.round(5)
# Add a run index (1 to 300) to easily track files
df.insert(0, "run_id", range(1, len(df) + 1))

# Save to CSV
df.to_csv("lhs_params.csv", index=False)
print(f"Successfully generated lhs_params.csv with {n_sims} simulations.")