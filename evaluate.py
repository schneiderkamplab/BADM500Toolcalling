import typer
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = typer.Typer(pretty_exceptions_enable=False)


class Evaluator():
    def __init__(self, checkpoint: str, model: str):
        self.checkpoint = checkpoint
        self.model_name = model
        model_path = checkpoint or model

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    @staticmethod
    def _jsonl_reader(dataset_path: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def _extract_user_and_expected(messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        user_text = ""
        expected_assistant: dict[str, Any] = {}

        for msg in messages:
            if msg.get("role") == "user":
                user_text = msg.get("content", "")
            if msg.get("role") == "assistant":
                expected_assistant = msg

        return user_text, expected_assistant

    def _generate(self, user_text: str, max_new_tokens: int = 128) -> str:
        messages = [{"role": "user", "content": user_text}]

        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except Exception:
            # Fallback for tokenizers without a chat template.
            prompt = f"<|user|>\n{user_text}\n<|assistant|>\n"
            inputs = self.tokenizer(prompt, return_tensors="pt")

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_only = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_only, skip_special_tokens=False).strip()

    @staticmethod
    def _normalize_args(args: dict[str, Any]) -> dict[str, Any]:
        return {k: str(v).strip().lower() for k, v in args.items()}

    @staticmethod
    def _parse_model_tool_calls(text: str) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []

        # Format 1: <tool_call>name|{"arg":"value"}</tool_call>
        pattern_json = re.compile(r"<tool_call>\s*([a-zA-Z0-9_]+)\s*\|\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
        for match in pattern_json.finditer(text):
            name = match.group(1).strip()
            args_raw = match.group(2).strip()
            try:
                arguments = json.loads(args_raw)
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append({"name": name, "arguments": arguments})

        # Format 2: <tool_call>name(arg='v', unit='celsius')</tool_call>
        if not tool_calls:
            pattern_fn = re.compile(r"<tool_call>\s*([a-zA-Z0-9_]+)\((.*?)\)\s*</tool_call>", re.DOTALL)
            for match in pattern_fn.finditer(text):
                name = match.group(1).strip()
                args_text = match.group(2).strip()
                arguments: dict[str, Any] = {}
                if args_text:
                    for chunk in re.split(r",\s*", args_text):
                        if "=" not in chunk:
                            continue
                        key, value = chunk.split("=", 1)
                        arguments[key.strip()] = value.strip().strip("\"'")
                tool_calls.append({"name": name, "arguments": arguments})

        return tool_calls

    @staticmethod
    def _parse_expected_tool_calls(assistant_message: dict[str, Any]) -> list[dict[str, Any]]:
        return assistant_message.get("tool_calls") or []

    @staticmethod
    def _is_tool_expected(expected_tool_calls: list[dict[str, Any]]) -> bool:
        return len(expected_tool_calls) > 0

    @staticmethod
    def _is_tool_initiated(pred_tool_calls: list[dict[str, Any]]) -> bool:
        return len(pred_tool_calls) > 0

    def evaluate(self, dataset_path: str, max_new_tokens: int = 128, verbose: bool = False) -> dict[str, Any]:
        rows = self._jsonl_reader(dataset_path)

        total = 0
        initiation_correct = 0
        function_correct = 0
        arguments_correct = 0

        detailed: list[dict[str, Any]] = []

        for i, row in tqdm(enumerate(rows, start=1), total=len(rows), desc="Evaluating: "):
            messages = row.get("messages", [])
            user_text, expected_assistant = self._extract_user_and_expected(messages)
            expected_tool_calls = self._parse_expected_tool_calls(expected_assistant)

            pred_text = self._generate(user_text=user_text, max_new_tokens=max_new_tokens)
            pred_tool_calls = self._parse_model_tool_calls(pred_text)

            expected_tool = self._is_tool_expected(expected_tool_calls)
            pred_tool = self._is_tool_initiated(pred_tool_calls)

            init_ok = expected_tool == pred_tool
            fn_ok = False
            args_ok = False

            if expected_tool and pred_tool:
                expected_names = [tc.get("name", "") for tc in expected_tool_calls]
                pred_names = [tc.get("name", "") for tc in pred_tool_calls]
                fn_ok = expected_names == pred_names

                if fn_ok and len(expected_tool_calls) == len(pred_tool_calls):
                    args_ok = True
                    for exp_tc, pred_tc in zip(expected_tool_calls, pred_tool_calls):
                        exp_args = self._normalize_args(exp_tc.get("arguments", {}))
                        pred_args = self._normalize_args(pred_tc.get("arguments", {}))
                        if exp_args != pred_args:
                            args_ok = False
                            break
            elif not expected_tool and not pred_tool:
                fn_ok = True
                args_ok = True

            total += 1
            initiation_correct += int(init_ok)
            function_correct += int(fn_ok)
            arguments_correct += int(args_ok)

            detailed.append(
                {
                    "index": i,
                    "user": user_text,
                    "expected_tool_calls": expected_tool_calls,
                    "predicted_tool_calls": pred_tool_calls,
                    "raw_prediction": pred_text,
                    "initiation_correct": init_ok,
                    "function_correct": fn_ok,
                    "arguments_correct": args_ok,
                }
            )

            if verbose:
                status = lambda ok: "✓" if ok else "✗"
                print(f"\n[{i}/{len(rows)}] ── {user_text}")
                print(f"  {status(init_ok)} initiation   expected_tool={expected_tool}  predicted_tool={pred_tool}")
                expected_names = [tc.get("name", "") for tc in expected_tool_calls]
                pred_names = [tc.get("name", "") for tc in pred_tool_calls]
                print(f"  {status(fn_ok)} function     expected={expected_names}  predicted={pred_names}")
                for j, (exp_tc, pred_tc) in enumerate(zip(expected_tool_calls, pred_tool_calls)):
                    exp_args = self._normalize_args(exp_tc.get("arguments", {}))
                    pred_args = self._normalize_args(pred_tc.get("arguments", {}))
                    match = exp_args == pred_args
                    print(f"  {status(match)} args[{j}]      expected={exp_args}  predicted={pred_args}")
                if not expected_tool_calls and not pred_tool_calls:
                    print(f"  {status(True)} args         (no tool call expected or predicted)")
                print(f"  raw: {pred_text!r}")

        metrics = {
            "total": total,
            "initiation_accuracy": initiation_correct / total if total else 0.0,
            "function_accuracy": function_correct / total if total else 0.0,
            "arguments_accuracy": arguments_correct / total if total else 0.0,
            "exact_match_accuracy": sum(
                int(x["initiation_correct"] and x["function_correct"] and x["arguments_correct"])
                for x in detailed
            ) / total if total else 0.0,
        }

        return {
            "metrics": metrics,
            "details": detailed,
        }


@dataclass
class EvalConfig:
    checkpoint: str
    model: str
    dataset_path: str
    max_new_tokens: int
    verbose: bool
    output_json: str | None


@app.command()
def main(
    checkpoint: str = "",
    model: str = "google/gemma-3-270m-it",
    dataset_path: str = "tools/data/example.jsonl",
    max_new_tokens: int = 128,
    verbose: bool = False,
    output_json: str = "",
):
    config = EvalConfig(
        checkpoint=checkpoint,
        model=model,
        dataset_path=dataset_path,
        max_new_tokens=max_new_tokens,
        verbose=verbose,
        output_json=output_json or None,
    )

    evaluator = Evaluator(checkpoint=config.checkpoint, model=config.model)
    result = evaluator.evaluate(
        dataset_path=config.dataset_path,
        max_new_tokens=config.max_new_tokens,
        verbose=config.verbose,
    )

    print("\nEvaluation Metrics")
    print("-" * 40)
    for key, value in result["metrics"].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    if config.output_json:
        output_path = Path(config.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved full report to: {output_path}")

    

if __name__ == "__main__":
    app()