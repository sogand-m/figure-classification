import pandas as pd

res = pd.read_csv("qwen_type_v1.csv")
res = res[res["Answer"] != "ERROR"]
res["Answer"] = res["Answer"].str.strip().str.rstrip(".")

wrong = res[res["Answer"] != res["Type"]]
wrong = wrong.sort_values("Type")

print(len(wrong), "wrong answers\n")
print(wrong[["Image", "Type", "Answer"]].to_string(index=False))

wrong[["Image", "Type", "Answer"]].to_csv("qwen_type_v1_errors.csv", index=False)