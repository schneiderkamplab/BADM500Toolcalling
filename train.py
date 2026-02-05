"""
Fine-tune Gemma 270M for tool calling using TRL (Transformer Reinforcement Learning)
This script demonstrates supervised fine-tuning with a synthetic tool calling dataset
"""
import typer
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig # DPOTrainer, GRPOTrainer, RewardTrainer
import json

app = typer.Typer(pretty_exceptions_enable=False)

# ============================================================================
# 1. CREATE SYNTHETIC TOOL CALLING DATASET
# ============================================================================

def create_tool_calling_dataset():
    """Create a small synthetic dataset for tool calling training"""

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
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "expression": {"type": "string", "description": "Math expression to evaluate"}
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
    
    # Create training examples
    examples = [
        {
            "messages": [
                {"role": "user", "content": "What's the weather like in Paris?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}
                ]}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Can you tell me the weather in New York in Fahrenheit?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"name": "get_weather", "arguments": {"location": "New York", "unit": "fahrenheit"}}
                ]}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 25 * 34 + 100?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"name": "calculate", "arguments": {"expression": "25 * 34 + 100"}}
                ]}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Calculate the square root of 144"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"name": "calculate", "arguments": {"expression": "sqrt(144)"}}
                ]}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Find information about quantum computing"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"name": "search_web", "arguments": {"query": "quantum computing"}}
                ]}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Search for the latest news on AI developments"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"name": "search_web", "arguments": {"query": "latest AI developments news"}}
                ]}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What's the temperature in Tokyo and calculate 15 + 27"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"name": "get_weather", "arguments": {"location": "Tokyo", "unit": "celsius"}},
                    {"name": "calculate", "arguments": {"expression": "15 + 27"}}
                ]}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Tell me a joke"},
                {"role": "assistant", "content": "Why don't scientists trust atoms? Because they make up everything!"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What's the weather in London and search for British museums"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"name": "get_weather", "arguments": {"location": "London", "unit": "celsius"}},
                    {"name": "search_web", "arguments": {"query": "British museums"}}
                ]}
            ]
        }
    ]    
    return examples


def format_chat_for_training(example, tokenizer):
    """
    Format messages into training text with special tokens for tool calls
    Uses a simple format: <tool_call>function_name(arg1=val1, arg2=val2)</tool_call>
    """
    messages = example["messages"]
    formatted_text = ""
    
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        
        if role == "user":
            formatted_text += f"<|user|>\n{content}\n"
        elif role == "assistant":
            formatted_text += f"<|assistant|>\n"
            
            if "tool_calls" in msg and msg["tool_calls"]:
                for tool_call in msg["tool_calls"]:
                    name = tool_call["name"]
                    args = tool_call["arguments"]
                    args_str = ", ".join([f"{k}='{v}'" for k, v in args.items()])
                    formatted_text += f"<tool_call>{name}({args_str})</tool_call>\n"
            else:
                formatted_text += f"{content}\n"
    
    formatted_text += tokenizer.eos_token
    return {"text": formatted_text}


# ============================================================================
# 2. SETUP MODEL AND TOKENIZER
# ============================================================================
def setup_model_and_tokenizer(model_id="google/gemma-3-270m-it"):    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.eos_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"  
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16, # Note. Gemma familiy is sensitive to dtype
        trust_remote_code=True
    )
    
    return model, tokenizer

def format_chat_with_tools(example):
    """
    Format messages for Gemma's chat template with tool calling support
    Uses a structured format that the model can learn
    """
    messages = example["messages"]
    
    conversation = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        
        if role == "user":
            conversation.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        
        elif role == "assistant":
            assistant_text = "<start_of_turn>model\n"
            
            # Check for tool calls
            if "tool_calls" in msg and msg["tool_calls"]:
                for tool_call in msg["tool_calls"]:
                    name = tool_call["name"]
                    args = tool_call["arguments"]
                    args_json = json.dumps(args)
                    assistant_text += f"<tool_call>{name}|{args_json}</tool_call>\n"
            else:
                assistant_text += f"{content}\n"
            
            assistant_text += "<end_of_turn>"
            conversation.append(assistant_text)
    
    # Join all turns
    text = "\n".join(conversation)
    
    return {"text": text}


# ============================================================================
# 3. TRAINING
# ============================================================================
@app.command()
def main(
    model_id: str = "google/gemma-3-270m-it",
    num_epochs: int = 300,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    logging_steps: int = 5,
    optimizer: str = "adamw_torch",
    lr_scheduler_type: str = "cosine",
    warmup_steps: int = 10,
    out_dir: str = "./gemma-270m-tool-calling",
):
    print("Creating dataset...")
    examples = create_tool_calling_dataset() 
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(examples)
    
    print(f"Dataset size: {len(dataset)} examples")
    
    # Setup model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_id=model_id)
    
    # Format dataset
    print("Formatting dataset...")
    formatted_dataset = dataset.map(
        format_chat_with_tools,
        remove_columns=dataset.column_names
    )
    
    # Print example
    print("\n=== Example formatted text ===")
    print(formatted_dataset[0]["text"])
    print("=" * 50 + "\n")

    training_args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_strategy="epoch",
        optim=optimizer,
        lr_scheduler_type=lr_scheduler_type,
        report_to="none",
        packing=False,  # Don't pack sequences
        dataset_text_field="text",
        dataloader_pin_memory=False, # Set this to true if using nvidia GPU
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving model...")
    trainer.save_model(out_dir) 
    tokenizer.save_pretrained(out_dir) # we just save the tokenizer in the same dir, do as you please
    
    print("Training complete!")


if __name__ == "__main__":
    app()
 