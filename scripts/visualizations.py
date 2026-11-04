#!/usr/bin/env python
"""Project visualizations. Usage: python scripts/visualizations.py [folder]"""
import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "hlca"
contract = json.loads((folder / "contract.json").read_text(encoding="utf-8"))


def save_table(table, title, filename):
    """Draw a DataFrame whose last row is the total. A count of 0 is shaded red."""
    shown = table.map(lambda value: value if isinstance(value, str) else f"{value:,}")
    fig, ax = plt.subplots(figsize=(1, 0.37 * (len(table) + 1)))  # the saved image grows to fit the table
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", color="#1F3A5F")
    drawn = ax.table(cellText=shown.values, colLabels=shown.columns, cellLoc="center", loc="center")
    drawn.auto_set_font_size(False)
    drawn.auto_set_column_width(range(table.shape[1]))
    drawn.scale(1, 1.7)
    for (row, col), cell in drawn.get_celld().items():
        cell.set_text_props(fontweight="bold" if row in (0, len(table)) else "normal", color="white" if row == 0 else "black")
        cell.set(edgecolor="#D0D5DD", facecolor="#1F3A5F" if row == 0 else "#FBE3E0" if shown.iat[row - 1, col] == "0" else "white")
    fig.savefig(folder / filename, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {folder / filename}")


rows, counts, donors = [], {}, set()
for entry in contract["files"]:
    obs = ad.read_h5ad(folder / entry["file"], backed="r").obs
    name = entry["group"].replace("_", " ").capitalize()
    counts[name] = obs[contract["label_column"]].value_counts()
    donors |= set(obs["donor_id"].astype(str))
    rows.append([name, ", ".join(entry["datasets"]), len(obs), obs["donor_id"].nunique(), int((counts[name] > 0).sum())])

by_type = pd.DataFrame(counts).fillna(0).astype(int)
by_type["All groups"] = by_type.sum(axis=1)
by_type = by_type.sort_values("All groups", ascending=False)
rows.append(["Total", f"{sum(len(e['datasets']) for e in contract['files'])} datasets", sum(r[2] for r in rows), len(donors), int((by_type["All groups"] > 0).sum())])
save_table(pd.DataFrame(rows, columns=["Group", "Source datasets", "Cells", "Donors", "Cell types"]), "How the HLCA core was split", "table_split_summary.png")

by_type.loc["Total"] = by_type.sum()
save_table(by_type.rename_axis("Cell type").reset_index(), "Cells per cell type in each group", "table_cell_types.png")
