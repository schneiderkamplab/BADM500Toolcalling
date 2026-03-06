#!/usr/bin/env python3
import os
import asyncio
import random
import json
from copy import deepcopy
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

CLIENT_URLS = [
    "https://0.0.0.0:8000/v1/",
]
API_KEY = ""
CONCURRENCY = len(CLIENT_URLS)
BATCH_SIZE = 20
MODEL_NAME = "MODEL_NAME"  # Replace with your actual model name,

clients = [AsyncOpenAI(base_url=url, api_key=API_KEY) for url in CLIENT_URLS]

def get_client():
    return random.choice(clients)


async def gen_tool_calls(question: str, answer: str, client, model=MODEL_NAME) -> str:
    """Generate tool calls for a given question and answer using a specified model."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n"
                        f"Answer: {answer}\n"
                        "Generate the tool calls that can be used to answer this question. ONLY OUTPUT THE TOOL CALLS ITSELF: \n\n"
                    ),
                }
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during comm: {e}")
        return ""  # Fallback to original text on error

async def make_samples(batch: dict) -> dict:
    async def process_sample(sample: dict) -> dict:
        new_sample = deepcopy(sample) # not always needed 
        sample_seed = sample['instruction'] # get seed from instruction
        # call the model to generate tool calls
        tool_calls = await gen_tool_calls(sample_seed, sample['output'], get_client())
        new_sample['tool_calls'] = tool_calls.strip()  # add tool calls to the
        return new_sample
    
    tasks = [process_sample(sample) for sample in batch]
    return await asyncio.gather(*tasks)


async def _augment_async(infile: str, out_dir: Path):
    print(f"📦 Loading dataset split: {infile}")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{os.path.basename(infile)}-tool-calling.jsonl"

    with open(infile, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Batches: "):
        batch = dataset[i:i+BATCH_SIZE]     
        results = await make_samples(batch)

        with output_path.open("a", encoding="utf-8") as f:
            for aug in results:
                f.write(json.dumps(aug, ensure_ascii=False) + "\n")

def augment(
    infile: str = "seeds.jsonl",
    out_dir: Path = Path("MyToolCallingData"),
):
    asyncio.run(_augment_async(infile, out_dir))

if __name__ == "__main__":
    augment()
