import pandas as pd

# read the corrected labels (semicolon file with a title line)
labels = pd.read_csv("labels.csv", sep=";", skiprows=1)

# keep only quantitative figures
q = labels[labels["Label"] == "Q"].copy()

# make type names match the paper
FIX = {
    "Bar Stacked Divergent": "Bar Stacked Diverging",
    "Funnel chart": "Alluvial",       # it is a Sankey flow
    "Bar + Line": "Bar",              # bars are the main mark
    "Distribuzioni": "Frequency",     # all 6 are density curves
}
q["Type"] = q["Type"].replace(FIX)

# no type in the paper fits these, so leave them out
q = q[q["Type"] != "Network graph"]

print(len(q), "Q figures")
print(q["Type"].value_counts())

q.to_csv("q_labels.csv", index=False)