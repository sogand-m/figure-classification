import pandas as pd
from sklearn.metrics import f1_score

RESULTS = "qwen_type_v1.csv"

VALID = ["Alluvial", "Area", "Bar", "Bar Grouped", "Bar Stacked",
         "Bar Stacked Diverging", "Beeswarm", "Boxplot", "Bubble", "Bump",
         "Dendrogram", "Donut", "Dot", "Forest", "Frequency", "Heatmap",
         "Interval", "Line", "Pie", "Radar", "Scatter", "Slope", "Sunburst",
         "Surface", "Treemap", "Venn", "Violin"]

res = pd.read_csv(RESULTS)
res = res[res["Answer"] != "ERROR"]

# clean the answers: remove spaces and a final full stop
res["Answer"] = res["Answer"].str.strip().str.rstrip(".")
res["Correct"] = res["Answer"] == res["Type"]

n = len(res)
print(n, "figures scored")
print(f"accuracy: {res['Correct'].mean():.1%}  ({res['Correct'].sum()}/{n})")

types = sorted(res["Type"].unique())
print(f"macro-F1: {f1_score(res['Type'], res['Answer'], labels=types, average='macro'):.3f}")

invalid = res[~res["Answer"].isin(VALID)]
print(len(invalid), "answers are not a type name:", invalid["Answer"].tolist())

print("\ncorrect per type:")
print(res.groupby("Type")["Correct"].agg(correct="sum", total="count"))

t = res["Seconds"]
print(f"\ntime: mean {t.mean():.1f} s, median {t.median():.1f} s, total {t.sum()/60:.1f} min")