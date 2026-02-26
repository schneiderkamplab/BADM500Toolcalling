"""
JSONL Format Checker using Pydantic
Validates conversation files with allowed tool calls:
  - get_weather, correct_grammar, generate_image, speech_synthesis, search_web
"""
import os
import json
import typer
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator

app = typer.Typer(pretty_exceptions_enable=False)

# ---------------------------------------------------------------------------
# Tool argument Pydantic schemas
# ---------------------------------------------------------------------------

class GetWeatherArgs(BaseModel):
    location: str
    unit: Literal["celsius", "fahrenheit"]


class CorrectGrammarArgs(BaseModel):
    text: str


class GenerateImageArgs(BaseModel):
    prompt: str
    style: str


class SpeechSynthesisArgs(BaseModel):
    text: str
    voice: str


class SearchWebArgs(BaseModel):
    query: str


# ---------------------------------------------------------------------------
# Tool call schema
# ---------------------------------------------------------------------------

ALLOWED_TOOLS = Literal[
    "get_weather",
    "correct_grammar",
    "generate_image",
    "speech_synthesis",
    "search_web",
]

TOOL_ARGS_MAP = {
    "get_weather": GetWeatherArgs,
    "correct_grammar": CorrectGrammarArgs,
    "generate_image": GenerateImageArgs,
    "speech_synthesis": SpeechSynthesisArgs,
    "search_web": SearchWebArgs,
}


class ToolCall(BaseModel):
    name: ALLOWED_TOOLS = Field(..., description="Must be one of the 5 allowed tool names")
    arguments: dict = Field(..., description="Tool arguments")

    @model_validator(mode="after")
    def validate_arguments(self) -> "ToolCall":
        args_model = TOOL_ARGS_MAP.get(self.name)
        if args_model:
            # This will raise ValidationError if arguments don't match
            args_model(**self.arguments)
        return self


# ---------------------------------------------------------------------------
# Message schemas
# ---------------------------------------------------------------------------

class UserMessage(BaseModel):
    role: Literal["user"]
    content: str


class AssistantMessage(BaseModel):
    role: Literal["assistant"]
    content: str
    tool_calls: Optional[list[ToolCall]] = None

    @model_validator(mode="after")
    def content_or_tool_calls(self) -> "AssistantMessage":
        has_content = bool(self.content and self.content.strip())
        has_tools = bool(self.tool_calls)
        if not has_content and not has_tools:
            raise ValueError("Assistant message must have content or tool_calls")
        return self


Message = Union[UserMessage, AssistantMessage]


# ---------------------------------------------------------------------------
# Top-level conversation schema
# ---------------------------------------------------------------------------

class Conversation(BaseModel):
    messages: list[Message] = Field(..., min_length=1)

    @field_validator("messages", mode="before")
    @classmethod
    def parse_messages(cls, messages):
        parsed = []
        for msg in messages:
            role = msg.get("role")
            if role == "user":
                parsed.append(UserMessage(**msg))
            elif role == "assistant":
                parsed.append(AssistantMessage(**msg))
            else:
                raise ValueError(f"Unknown role: {role!r}. Must be 'user' or 'assistant'.")
        return parsed

    @model_validator(mode="after")
    def validate_structure(self) -> "Conversation":
        if not self.messages:
            raise ValueError("messages list cannot be empty")
        if self.messages[0].role != "user":
            raise ValueError("First message must be from 'user'")
        return self



@app.command()
def check_jsonl(filepath: str) -> None:
    print(f"\n{'='*60}")
    print(f"Checking: {filepath}")
    print(f"{'='*60}\n")

    errors_found = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip blank lines

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Line {line_num}] ❌ JSON parse error: {e}")
                errors_found += 1
                continue

            try:
                convo = Conversation(**data)
                # Summarise what was validated
                tool_names = []
                for msg in convo.messages:
                    if isinstance(msg, AssistantMessage) and msg.tool_calls:
                        tool_names.extend(tc.name for tc in msg.tool_calls)
                tool_summary = f" | tools: {tool_names}" if tool_names else ""
                print(f"[Line {line_num}] ✅ Valid ({len(convo.messages)} messages{tool_summary})")
            except Exception as e:
                print(f"[Line {line_num}] ❌ Validation error: {e}")
                errors_found += 1

    print(f"\n{'='*60}")
    if errors_found == 0:
        print("✅ All entries passed validation.")
    else:
        print(f"❌ {errors_found} entry/entries failed validation.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    app()