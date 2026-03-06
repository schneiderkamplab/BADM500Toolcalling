# BADM500: Fine-tuning Gemma 270M for Tool Calling with TRL

<p align="center">
  <img src="assets/overview.png" alt="Description" width="600"/>
</p>

This example demonstrates how to fine-tune Google's Gemma 270m model for tool calling using the TRL (Transformer Reinforcement Learning) library form Hugginface

## Overview 

This example of training pipeline includes:
1. **Synthetic Dataset Creation**: 10 examples covering weather queries, calculations, web searches, and regular conversation.
2. **Completion-Only Training**: Only compute loss on assistant responses.

## Installation
You can use all command environment managers: `uv`, `pip`, or `conda` 

Example for pip (prepend `uv` for using uv): 
```bash
pip install -r requirements.txt
```

## Dataset Format
The training data uses a simple format with special tokens:

```
<|user|>
What's the weather in Paris?
<|assistant|>
<tool_call>get_weather(location='Paris', unit='celsius')</tool_call>
```

For regular conversations without tool calls:
```
<|user|>
Hello, how are you?
<|assistant|>
I'm doing well, thank you! How can I help you today?
```

## Available Tools in *your* Dataset

1. **get_weather**: Get the current weather for a given location.
2. **correct_grammar**: Correct the given sentence.
3. **generate_image**: Generate an image from a given description.
4. **speech_synthesis**: Generate speech based on a given text input.
5. **search_web**: Search the web for information relevant to the given prompt.

## Tool Verification 
To verify the tool-call conversations you created, please run the verifier for sanity 
```bash
    python3 tools/validate_tools.py tools/data/example.jsonl
```

## Running the Training
The training is rooted in `train.py`. 
```bash
python train.py
```
To see how to parse arguments use:
```bash
python train.py --help
```

## Tracking - wandb
In the `Trainer` you can parse in: `report_to="wandb"` which will prompt you with the necessary steps to log your loss and potentially your accuracy and rewards - if you use that in your project.

### Training Configuration
Example configuration:
- **Model**: google/gemma-3-270m-it (instruction-tuned)
- **Batch Size**: 2 (with gradient accumulation of 4)
- **Learning Rate**: 2e-4
- **Epochs**: 30
- **Optimizer**: adamw

We encourage students to explore different settings and new parameters! E.g., look into other optimizers, LR-Schedulars; even adjusting batch-size can help performance.

## Expected Output

After training, the model should be able to:
- Recognize when to call tools vs respond normally.
- Format tool calls properly.
- Extract correct parameters from user queries.
- Handle multiple tool calls in one response.

## Output Files

After training, you'll find:
- `.gemma-270m-tool-calling/checkpoint-<STEPS>`: Checkpoints during training.
- `.gemma-270m-tool-calling`: Final trained model with.

## Using the Trained Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("gemma-270m-tool-calling")
tokenizer = AutoTokenizer.from_pretrained("gemma-270m-tool-calling")

# Inference
prompt = "<|user|>\nWhat's the weather in Tokyo?\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## Tips for Better Results

1. **Increase dataset size**: 100-1000 or even more examples work much better.
2. **Add variety**: Include edge cases, ambiguous queries, and multi-turn conversations.
3. **Balance dataset**: Ensure equal distribution of tool calls vs regular responses.
4. **Train longer**: More epochs or larger datasets need more training.
5. **Tune hyperparameters**: Experiment with learning rate, batch size, optimizer, and other details of the training.

## Dataset Improvements
It it the students tasks to exstend and improve the dataset both in size and diversity to work better and generalize to an arbitrary conversation with a user. Note that, right now, this boilerplate does not even include a validation dataset. You are quite free to think out of the box regarding the data (e.g. conversations etc), but you must keep the format of the messages (the so-called chat-template), so we later on can parse the outputs of all your models to test accuracy.

### Tools that your project should support 
````Python
    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            }
        },
        {
            "name": "correct_grammar",
            "description": "Correct grammatical errors in text using GEC API",
            "parameters": {
                "text": {"type": "string", "description": "Text to check and correct"}
            }
        },
        {
            "name": "generate_image",
            "description": "Generate an image from a text description",
            "parameters": {
                "prompt": {"type": "string", "description": "Description of the image to generate"},
                "style": {"type": "string", "description": "Art style (optional)"}
            }
        },
        {
            "name": "text_to_speech",
            "description": "Convert text to speech audio",
            "parameters": {
                "text": {"type": "string", "description": "Text to convert to speech"},
                "voice": {"type": "string", "description": "Voice type (male/female/neutral)"}
            }
        },
        {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "query": {"type": "string", "description": "Search query"}
            }
        }
    ]
````

## Common Issues

**Out of Memory**: Reduce batch size or increase gradient accumulation steps.
**Poor performance**: Add more diverse training examples.
**Model not calling tools**: Ensure balanced dataset with both tool and non-tool examples.

## Resources

- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Gemma Model Card](https://huggingface.co/google/gemma-2-270m-it)
- [vLLM](https://vllm.ai)


## Data generation Overview
Install an inference engine like vLLM or SGLang. This means you can use optimizations like dynamic batching (look it up!). See vLLM installation guide [here](https://docs.vllm.ai/en/latest/getting_started/quickstart/). Installing it with uv is faster and often more stable. 


## Running Long Jobs in a Detachable Terminal Environment

When working on remote compute instances (such as those on **uCloud**), it is often necessary to run programs that take a long time to finish. If you run these programs in a normal terminal session, they will stop if your connection drops or if you close the terminal.

To avoid this, you can run your work inside a **detachable terminal session**. Tools such as **tmux**, **screen**, and **byobu** allow you to create persistent terminal sessions that continue running even after you disconnect. You can later reconnect to the same session and continue exactly where you left off.

The typical workflow is:

1. Start a detachable session.
2. Run commands inside it.
3. Detach from the session while the processes continue running.
4. Reattach later to check progress or continue working.

In this guide we will use **tmux**, which is one of the most widely used tools for this purpose.

---

## Basic tmux Usage

 Start a tmux Session

Start a new tmux session:

```bash
tmux
```

It is usually better to give the session a name:

```bash
tmux new -s mysession
```

You are now inside a tmux session. Any processes started here will keep running even if you disconnect.

---

# Switching Between Windows

Move to the next window:

```
Ctrl + b, then n
```

Move to the previous window:

```
Ctrl + b, then p
```

Jump to a specific window:

```
Ctrl + b, then <number>
```

Example:

```
Ctrl + b, then 2
```

---

# Listing Windows

To see all windows in the current session:

```
Ctrl + b, then w
```

This opens an interactive menu where you can select a window.

---

# Detaching From a Session

You can leave the tmux session without stopping the programs running inside it.

Detach from the session with:

```
Ctrl + b, then d
```

You will return to the normal terminal while the tmux session continues running in the background.

---

# Listing Running Sessions

To see all active tmux sessions:

```bash
tmux ls
```

Example output:

```
mysession: 2 windows (created Fri Mar 6 10:00:00 2026)
```

---

# Reattaching to a Session

To reconnect to a running session:

```bash
tmux attach -t mysession
```

You will return to the session exactly as you left it.

---

## Example Workflow

Start a new session:

```bash
tmux new -s training
```

Create a second window for monitoring:

```
Ctrl + b, then c
```

In the first window, run a long job:

```bash
python train_model.py
```

Switch to another window:

```
Ctrl + b, then 0
```

Run a monitoring tool:

```bash
htop
```

Detach from the session:

```
Ctrl + b, then d
```

Later, reconnect to the session:

```bash
tmux attach -t training
```

All windows and running processes will still be active.

---

# Why tmux Is Useful

Using tmux allows you to:

- Run long jobs safely on remote machines
- Keep processes alive even if your connection drops
- Work with multiple terminal windows in one session
- Reattach to your work at any time

This makes tmux an essential tool when working on remote servers, HPC clusters, or cloud computing platforms such as **uCloud**.

# Have a lot of fun!
<p align="center">
  <img src="assets/psk_cartoon.png" alt="Description" width="400"/>
</p>
