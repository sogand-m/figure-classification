import pandas as pd
import ollama

PROMPT = "Does this image show a chart, plot or graph that displays data values? Answer only yes or no."

labels = pd.read_csv("labels.csv")
labels = labels[labels["Image"].notna()]

rows = []
for i, image in enumerate(labels["Image"], start=1):
    try:
        res = ollama.chat(
            model="qwen2.5vl:7b",
            messages=[{
                "role": "user",
                "content": PROMPT,
                "images": [f"../Figures_ESEM2022/{image}"]
            }],
            options={"num_ctx": 16384}
        )
        answer = res["message"]["content"].strip()
    except Exception as e:
        answer = "ERROR"
        print("  failed:", e)

    rows.append({"Image": image, "Answer": answer})
    print(f"{i}/{len(labels)}  {image} -> {answer}")
    pd.DataFrame(rows).to_csv("qwen_v2.csv", index=False)

print("saved qwen_v1.csv")