import pandas as pd

labels = pd.read_csv("labels.csv")

results = pd.read_csv("qwen_v1.csv")
results = results[results["Answer"] != "ERROR"]

print(results["Answer"].value_counts())

compare = labels.merge(results, on="Image")
compare["Correct"] = compare["Label"] == compare["Answer"]

print(len(compare), "figures compared")
print("accuracy:", compare["Correct"].mean())
print(pd.crosstab(compare["Label"], compare["Answer"]))