import ollama

res = ollama.chat(
    model="llava",
    messages=[{
        "role": "user",
        "content": "Does this figure show quantitative data (a chart or plot with numbers, axes, bars, lines, points)? Answer with one word only: Q or NonQ.",
        "images": ["../Figures_ESEM2022/3544902.3546233_p5_fig1.png"]
    }]
)

print(res["message"]["content"])