from typing import List, Optional
from pathlib import Path
import dspy
import litellm
from pydantic import BaseModel


class TextEntities(dspy.Signature):
    """Extract key entities from the source text. Extracted entities are subjects or objects.
    This is for an extraction task, please be THOROUGH and accurate to the reference text."""

    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")


class ConversationEntities(dspy.Signature):
    """Extract key entities from the conversation Extracted entities are subjects or objects.
    Consider both explicit entities and participants in the conversation.
    This is for an extraction task, please be THOROUGH and accurate."""

    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")


class EntitiesResponse(BaseModel):
    """Structured response for entity extraction."""

    entities: List[str]


def _load_entities_prompt() -> str:
    """Load the entities prompt template from file."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "entities.txt"
    return prompt_path.read_text()


def _get_entities_litellm(
    input_data: str,
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
) -> List[str]:
    prompt_template = _load_entities_prompt()
    user_prompt = f"""
Here is the text to extract entities from:

<article>
{input_data}
</article>
    """

    # Build schema with additionalProperties: false (required by OpenAI)
    schema = EntitiesResponse.model_json_schema()
    schema["additionalProperties"] = False

    if model.startswith("hosted_vllm/"):
        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract entities that act as subjects or objects in the source text. "
                        "Include unnamed but distinct participants such as a woman or a robot. "
                        "Keep names faithful to the text and do not invent entities. "
                        "Return only JSON. /no_think"
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=2048,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "entities_response",
                    "strict": True,
                    "schema": schema,
                },
            },
            api_key=api_key,
            api_base=api_base,
        )
        entities = EntitiesResponse.model_validate_json(response.choices[0].message.content).entities
        return list(dict.fromkeys(entity.strip() for entity in entities if entity.strip()))

    kwargs = {
        "model": model,
        "input": [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "entities_response",
                "schema": schema,
                "strict": True,
            }
        },
    }

    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    response = litellm.responses(**kwargs)
    # print(response.model_dump_json(indent=2))
    parsed = EntitiesResponse.model_validate_json(response.output[-1].content[0].text)
    return parsed.entities


def get_entities(
    input_data: str,
    is_conversation: bool = False,
    use_litellm_prompt: bool = False,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
) -> List[str]:
    if use_litellm_prompt and not is_conversation:
        return _get_entities_litellm(
            input_data,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
        )

    extract = (
        dspy.Predict(ConversationEntities)
        if is_conversation
        else dspy.Predict(TextEntities)
    )
    result = extract(source_text=input_data)
    return result.entities
