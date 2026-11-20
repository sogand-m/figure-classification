import os
import glob
import pandas as pd
import re

types = pd.read_csv("GraphTypes.csv")
is_quant = dict(zip(types["Type"], types["Quantitative"]))

# names the ones that are missing or spelled differently
is_quant["Bar Stacked Divergent"] = 1   # CSV spells it "Diverging"
is_quant["Dendogram"] = 1               # CSV spells it "Dendrogram"
is_quant["Bar + Line"] = 1              # mixed chart, not in the taxonomy
is_quant["Distribuzioni"] = 1           # Italian for "distributions"

figures = pd.read_csv("ESEM2022.csv")
figures = figures[figures["Type"].notna()]          # drop papers with no figures

figures["Quantitative"] = figures["Type"].map(is_quant)
figures["Label"] = figures["Quantitative"].map({1: "Q", 0: "NonQ"})
figures["Paper"] = figures["DOI"].str.split("/").str[-1]

unknown = figures[figures["Quantitative"].isna()]["Type"].unique()
if len(unknown) > 0:
    print("WARNING - unknown types:", unknown)

print(figures["Label"].value_counts())

paths = glob.glob("../Figures_ESEM2022/*.png")
print(f"{len(paths)} images found")


pattern = re.compile(r"^(.+)_p(\d+)_fig(\d+)\.png$")

files = []
for path in paths:
    name = os.path.basename(path)
    m = pattern.match(name)
    if m:
        files.append({"Paper": m.group(1), "Fig": int(m.group(3)), "Image": name})

images = pd.DataFrame(files)

merged = figures.merge(images, on=["Paper", "Fig"], how="left")
print(merged[["Image", "Type", "Label"]].head())
print(merged["Image"].isna().sum(), "figures have no image")

output = merged[["Image", "Paper", "Fig", "Type", "Label"]]
output.to_csv("labels.csv", index=False)
print("saved labels.csv")

extra = set(images["Image"]) - set(merged["Image"].dropna())
if extra:
    print("images with no row in the notebook:", extra)