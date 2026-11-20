import ollama


res = ollama.chat(
	model="llava",
	messages=[
		{
			'role': 'user',
			'content': 'What does the graphical part in this image is, ignore the caption, provide the type of diagram:',
			'images': ['Papers/EMSE_2023/s10664-022-10251-1_p20_fig4.png']
		}
	]
)

print(res['message']['content'])

