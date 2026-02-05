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
4. **Speech Synthesis**: Generate speech based on a given text input.
5. **search_web**: Search the web for information relevant to the given prompt.

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

# Have a lot of fun!
<p align="center">
  <img src="assets/psk_cartoon.png" alt="Description" width="400"/>
</p>
