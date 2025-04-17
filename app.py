from flask import Flask, request, jsonify, render_template
from vllm import LLM, SamplingParams
import os

app = Flask(__name__, template_folder='.')

MODEL_PATH = "/path/to/your/model"
llm = LLM(model=MODEL_PATH)
sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    problem = data.get("problem", "").strip()
    if not problem:
        return jsonify({"prediction": "⚠️ Please enter a math problem."})
    try:
        prompt = f"### Instruction:\n{problem}\n\n### Response: Let's think step by step."
        output = llm.generate([prompt], sampling_params)[0].outputs[0].text.strip()
        return jsonify({"prediction": output})
    except Exception as e:
        return jsonify({"prediction": f"🚨 Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
