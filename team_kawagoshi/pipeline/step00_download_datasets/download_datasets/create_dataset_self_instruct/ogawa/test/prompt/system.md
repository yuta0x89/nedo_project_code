以下のタスク指示例を参考に、10種類の多様なinputとoutputを考えてください。これらのタスク指示はGPTモデルに与えられ、その指示を完了するためのGPTモデルを評価します。

要件は以下の通りです：
1. 同じ動詞や形容詞などを繰り返さないようにして、多様性を最大限に引き出してください。
2. 出力は指示に対する適切な応答であるべきです。出力は1か0であることを確認してください。
3. 生成するinputとoutputは全て必ず日本語で生成してください。
4. 文法的に正しいデータと誤りを含んでいるデータを5:5でバランス良く生成してください。
5. 誤りを含むデータについては、多様な誤りを含むようにしてください。難し過ぎる誤りではなく、ぱっと見で分かるレベルの誤りを含んでください。
6. 回答フォーマット例に記載のJSON形式で回答してください。

タスク指示例：
$FewShotExamples

回答フォーマット例(JSON):
[
  {
    "id": 1,
    "instruction": "以下の文章が文法的に正しいかどうかを評価してください。文法的に正しい場合には1を、誤りを含んでいる場合には0を選択してください。",
    "input": "<入力文章>",
    "output": "<1 or 0>"
  },
  {
    "id": 2,
    "instruction": "以下の文章が文法的に正しいかどうかを評価してください。文法的に正しい場合には1を、誤りを含んでいる場合には0を選択してください。",
    "input": "<入力文章>",
    "output": "<1 or 0>"
  },
  {
    "id": 3,
    "instruction": "以下の文章が文法的に正しいかどうかを評価してください。文法的に正しい場合には1を、誤りを含んでいる場合には0を選択してください。",
    "input": "<入力文章>",
    "output": "<1 or 0>"
  }
]