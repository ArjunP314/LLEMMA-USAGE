import argparse
import json
from vllm import LLM, SamplingParams
import util  # Ensure this is in your directory or package

INVALID_ANS = "[invalid]"

def remove_boxed(s):
    # Strip boxed format, assuming \boxed{} or similar
    return s.replace("\\boxed{", "").replace("}", "").strip()

def process_results(question, completion, correct_answer):
    extract_ans_temp = completion.split("The answer is:")[-1].split(".\n")[0].strip()
    extract_ans = extract_ans_temp[:-1] if extract_ans_temp.endswith('.') else extract_ans_temp
    extract_ans = extract_ans.strip()

    return util.is_equiv(extract_ans, correct_answer)

def evaluate_single_problem(model_path, problem):
    print(f"Evaluating: {problem}")
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048)
    prompt = f"### Instruction:\n{problem}\n\n### Response: Let's think step by step."
    llm = LLM(model=model_path)
    completion = llm.generate([prompt], sampling_params)[0].outputs[0].text.strip()
    return completion

def batch_data(data_list, batch_size=1):
    return [data_list[i:i+batch_size] for i in range(0, len(data_list), batch_size)]

def test_hendrycks_math(model_path, data_path, start=0, end=None, batch_size=1, tensor_parallel_size=1):
    from jsonlines import Reader
    hendrycks_math_ins, hendrycks_math_answers = [], []

    with open(data_path, "r", encoding="utf8") as f:
        for idx, item in enumerate(Reader(f)):
            if idx < start or (end is not None and idx >= end):
                continue
            instruction = item["instruction"]
            solution = item["output"]
            answer = remove_boxed(util.last_boxed_only_string(solution))
            prompt = f"### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
            hendrycks_math_ins.append(prompt)
            hendrycks_math_answers.append(answer)

    stop_tokens = ["Question:", "Instruction:", "Response:"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048, stop=stop_tokens)
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)

    results = []
    for batch in batch_data(hendrycks_math_ins, batch_size=batch_size):
        completions = llm.generate(batch, sampling_params)
        for i, output in enumerate(completions):
            generated_text = output.outputs[0].text.strip()
            correct = process_results(batch[i], generated_text, hendrycks_math_answers.pop(0))
            results.append(correct)

    acc = sum(results) / len(results)
    print(f"Accuracy: {acc * 100:.2f}%")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--data_file", type=str, help="Path to JSONL test data (for batch mode)")
    parser.add_argument("--problem", type=str, help="A single math problem to evaluate")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, help="End index for evaluation")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.problem:
        # Single problem mode
        result = evaluate_single_problem(args.model, args.problem)
        print("\nModel Prediction:")
        print(result)
    elif args.data_file:
        # Batch mode
        test_hendrycks_math(
            model_path=args.model,
            data_path=args.data_file,
            start=args.start,
            end=args.end,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size
        )
    else:
        print("Please specify either --problem for single evaluation or --data_file for batch evaluation.")
