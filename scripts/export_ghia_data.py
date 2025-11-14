#!/usr/bin/env python3
"""Export Ghia benchmark data to CSV files."""

import pandas as pd
from pathlib import Path
from validation.ghia_benchmark import GhiaBenchmark

# Setup output directory
output_dir = Path(__file__).parent.parent / "data" / "validation" / "ghia"
output_dir.mkdir(parents=True, exist_ok=True)

# Export data for each Reynolds number
for Re in GhiaBenchmark.available_reynolds_numbers():
    data = GhiaBenchmark.get_data(Re)

    # U velocity along vertical centerline (x=0.5)
    u_df = pd.DataFrame({
        'y': data['y'],
        'u': data['u']
    })
    u_file = output_dir / f"ghia_Re{Re}_u_centerline.csv"
    u_df.to_csv(u_file, index=False)
    print(f"Saved: {u_file}")

    # V velocity along horizontal centerline (y=0.5)
    v_df = pd.DataFrame({
        'x': data['x'],
        'v': data['v']
    })
    v_file = output_dir / f"ghia_Re{Re}_v_centerline.csv"
    v_df.to_csv(v_file, index=False)
    print(f"Saved: {v_file}")

print(f"\nExported Ghia benchmark data for Re = {GhiaBenchmark.available_reynolds_numbers()}")
