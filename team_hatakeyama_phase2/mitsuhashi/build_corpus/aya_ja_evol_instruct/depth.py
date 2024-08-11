base_instruction = '''あなたはプロンプトをブラッシュアップして書き直すリライターとしてタスクを実行しなさい。あなたのタスクは、与えられたプロンプトをより複雑なものにリライトして、有名なAIシステム（例えば、ChatGPTやGPT4やClaude3）が少し扱いにくくすることです。しかし、リライトされたプロンプトは人間にとって理解しやすく、適切に応答できるものである必要があります。また、リライトの際に非テキスト部分（例：テーブルやコード）や与えられたプロンプトのインプットを省略しないでください。さらに、「#与えられたプロンプト#」および「#リライトされたプロンプト#」という単語をプロンプト内に含めないでください。

リライトの際には以下の方法を使用してプロンプトを複雑にしてください：
{}

また、リライトされたプロンプトが冗長にならないように注意し、元のプロンプトに追加する言葉は10から20語以内に抑えてください。
'''

def createConstraintsPrompt(instruction):
	prompt = base_instruction.format("#与えられたプロンプト#にもう一つ制約または要件を追加してください。")
	prompt += "#与えられたプロンプト#: \r\n {} \r\n".format(instruction)
	prompt += "#リライトされたプロンプト#:\r\n"
	return prompt

def createDeepenPrompt(instruction):
	prompt = base_instruction.format("もし#与えられたプロンプト#に特定の問題についての質問が含まれていれば、質問の深さと幅を広げてください。")
	prompt += "#与えられたプロンプト#: \r\n {} \r\n".format(instruction)
	prompt += "#リライトされたプロンプト#:\r\n"
	return prompt

def createConcretizingPrompt(instruction):
	prompt = base_instruction.format("一般的な概念をより具体的な概念に置き換えてください。")
	prompt += "#与えられたプロンプト#: \r\n {} \r\n".format(instruction)
	prompt += "#リライトされたプロンプト#:\r\n"
	return prompt


def createReasoningPrompt(instruction):
	prompt = base_instruction.format("もし#The Given Prompt#がいくつかの単純な思考プロセスで解けるのであれば、多段階の推論を明示的に要求するように書き換えてください。")
	prompt += "#与えられたプロンプト#: \r\n {} \r\n".format(instruction)
	prompt += "#リライトされたプロンプト#:\r\n"
	return prompt