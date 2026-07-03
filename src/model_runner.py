"""
Model runner module for LLM stylometric fingerprinting.

This module defines reusable classes for calling different LLM providers.
It keeps API logic outside notebooks and supports a clean, modular,
reproducible project structure.

Main responsibilities:
- load model configuration;
- call provider APIs;
- generate raw model responses;
- apply length-control rewriting when needed;
- return standardized output records for model_outputs_raw.csv.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_API_RETRIES,
    MODEL_CONFIGS,
    RAW_OUTPUT_COLUMNS,
    RAW_OUTPUTS_PATH,
    RETRY_WAIT_SECONDS,
    WORD_MAX,
    WORD_MIN,
    create_required_directories,
)
from src.prompt_loader import load_prompts, validate_prompts
from src.text_utils import (
    build_controlled_prompt,
    build_length_rewrite_prompt,
    count_words,
    get_length_status,
    is_within_word_range,
    trim_to_word_limit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


class BaseModelRunner(ABC):
    """
    Abstract base class for all LLM provider runners.

    Each provider-specific runner must implement the `_call_api` method.
    The shared generation workflow is handled here.
    """

    def __init__(
        self,
        model_key: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        word_min: int = WORD_MIN,
        word_max: int = WORD_MAX,
    ) -> None:
        """
        Initialize a model runner.

        Parameters
        ----------
        model_key : str
            Provider key in MODEL_CONFIGS.
            Examples: 'OpenAI', 'Claude', 'Gemini', 'DeepSeek', 'Mistral'.
        temperature : float
            Sampling temperature.
        max_tokens : int
            Maximum output tokens.
        word_min : int
            Minimum acceptable word count.
        word_max : int
            Maximum acceptable word count.
        """
        if model_key not in MODEL_CONFIGS:
            available = list(MODEL_CONFIGS.keys())
            raise ValueError(f"Unknown model_key '{model_key}'. Available: {available}")

        load_dotenv(dotenv_path=ENV_PATH, override=True)

        self.model_key = model_key
        self.config = MODEL_CONFIGS[model_key]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.word_min = word_min
        self.word_max = word_max

        self.provider = self.config["provider"]
        self.model_family = self.config["model_family"]

        self.model_name = os.getenv(
            self.config["model_name_env"],
            self.config["default_model"],
        )

        self.api_key = os.getenv(self.config["api_key_env"])

        if not self.api_key:
            raise ValueError(
                f"Missing API key for {model_key}. "
                f"Expected environment variable: {self.config['api_key_env']}"
            )

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """
        Provider-specific API call.

        Parameters
        ----------
        prompt : str
            Prompt to send to the model.

        Returns
        -------
        str
            Generated text response.
        """
        raise NotImplementedError

    def generate_text_with_retries(self, prompt: str) -> tuple[str, str]:
        """
        Generate text with retry handling for temporary API failures.

        This function uses exponential backoff. For example, if
        RETRY_WAIT_SECONDS is 10, the waiting pattern is:

        10 seconds -> 20 seconds -> 40 seconds -> ...

        Parameters
        ----------
        prompt : str
            Prompt to send to the model.

        Returns
        -------
        tuple[str, str]
            Generated response and error message.
            If successful, error message is an empty string.
        """
        last_error = ""

        for attempt in range(1, MAX_API_RETRIES + 1):
            try:
                response = self._call_api(prompt)

                if not response or not response.strip():
                    raise ValueError("Empty response returned by API.")

                return response.strip(), ""

            except Exception as exc:
                last_error = str(exc)

                # Exponential backoff:
                # attempt 1 -> RETRY_WAIT_SECONDS
                # attempt 2 -> RETRY_WAIT_SECONDS * 2
                # attempt 3 -> RETRY_WAIT_SECONDS * 4
                wait_time = RETRY_WAIT_SECONDS * (2 ** (attempt - 1))

                print(
                    f"{self.model_key} API error on attempt "
                    f"{attempt}/{MAX_API_RETRIES}: {last_error}"
                )

                if attempt < MAX_API_RETRIES:
                    print(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)

        return "", last_error

    def generate_for_prompt(self, prompt_row: pd.Series) -> dict[str, Any]:
        """
        Generate one model response for one prompt row.

        Workflow:
        1. Read the original prompt.
        2. Wrap it in the controlled prompt template.
        3. Generate the first raw response.
        4. Count words.
        5. If the response is outside 180-220 words, ask the model to rewrite it.
        6. Save both the raw response and the final length-controlled response.

        Parameters
        ----------
        prompt_row : pd.Series
            One row from Prompts.csv.

        Returns
        -------
        dict[str, Any]
            One standardized output record for model_outputs_raw.csv.
        """
        prompt_id = str(prompt_row.get("prompt_id", "")).strip()
        genre = str(prompt_row.get("genre", "")).strip()
        theme = str(prompt_row.get("theme", "")).strip()
        topic = str(prompt_row.get("topic", "")).strip()
        original_prompt_text = str(prompt_row.get("prompt_text", "")).strip()

        controlled_prompt_text = build_controlled_prompt(original_prompt_text)

        text_id = f"{self.model_family}_{self.model_name}_{prompt_id}".replace(" ", "_")
        date_generated = datetime.now().isoformat(timespec="seconds")

        raw_response, error_message = self.generate_text_with_retries(
            controlled_prompt_text
        )

        if error_message:
            return self._build_failed_record(
                text_id=text_id,
                prompt_id=prompt_id,
                genre=genre,
                theme=theme,
                topic=topic,
                original_prompt_text=original_prompt_text,
                controlled_prompt_text=controlled_prompt_text,
                date_generated=date_generated,
                error_message=error_message,
            )

        raw_word_count = count_words(raw_response)
        raw_is_valid, _ = is_within_word_range(
            raw_response,
            word_min=self.word_min,
            word_max=self.word_max,
        )

        final_response = raw_response
        final_word_count = raw_word_count

        # If raw response is valid, it is also the length-controlled response.
        length_controlled_response = raw_response

        generation_attempt = 1
        was_rewritten_for_length = False
        notes = "Raw response was within target word range."

        if not raw_is_valid:
            rewrite_prompt = build_length_rewrite_prompt(raw_response)
            rewritten_response, rewrite_error = self.generate_text_with_retries(
                rewrite_prompt
            )

            generation_attempt = 2
            was_rewritten_for_length = True

            if rewritten_response and not rewrite_error:
               length_controlled_response = rewritten_response
               final_response = rewritten_response
               final_word_count = count_words(final_response)

               if final_word_count > self.word_max:
                  final_response = trim_to_word_limit(final_response, max_words=self.word_max)
                  length_controlled_response = final_response
                  final_word_count = count_words(final_response)
                  notes = (
                      "Raw response was outside target range and was rewritten. "
                      "The rewritten response was still too long, so it was trimmed to the maximum word limit."
        )
               else:
                 notes = "Raw response was outside target range and was rewritten."

            else:
            # Keep raw response if rewrite fails, but trim if it is too long.
               final_response = raw_response
               final_word_count = raw_word_count

               if final_word_count > self.word_max:
                  final_response = trim_to_word_limit(final_response, max_words=self.word_max)
                  final_word_count = count_words(final_response)
                  notes = (
                      "Raw response was outside target range. Rewrite failed, "
                      "so the raw response was trimmed to the maximum word limit."
        )
               else:
                   notes = (
                        "Raw response was outside target range. "
                        "Rewrite failed, so raw response was kept."
        )

               length_controlled_response = final_response

        length_status = get_length_status(
            final_word_count,
            word_min=self.word_min,
            word_max=self.word_max,
        )

        within_range = self.word_min <= final_word_count <= self.word_max

        return {
            "text_id": text_id,
            "prompt_id": prompt_id,
            "genre": genre,
            "theme": theme,
            "topic": topic,
            "condition": "controlled_prompt",
            "model_family": self.model_family,
            "model_name": self.model_name,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "date_generated": date_generated,
            "original_prompt_text": original_prompt_text,
            "controlled_prompt_text": controlled_prompt_text,
            "raw_output_text": raw_response,
            "length_controlled_output_text": length_controlled_response,
            "output_text": final_response,
            "raw_word_count": raw_word_count,
            "final_word_count": final_word_count,
            "length_status": length_status,
            "within_10_percent_word_limit": within_range,
            "generation_attempt": generation_attempt,
            "was_rewritten_for_length": was_rewritten_for_length,
            "api_status": "success",
            "error_message": "",
            "notes": notes,
        }

    def _build_failed_record(
        self,
        text_id: str,
        prompt_id: str,
        genre: str,
        theme: str,
        topic: str,
        original_prompt_text: str,
        controlled_prompt_text: str,
        date_generated: str,
        error_message: str,
    ) -> dict[str, Any]:
        """
        Build a standardized failed API record.

        Parameters
        ----------
        text_id : str
            Unique generated text ID.
        prompt_id : str
            Prompt identifier.
        genre : str
            Prompt genre.
        theme : str
            Prompt theme.
        topic : str
            Prompt topic if available.
        original_prompt_text : str
            Original prompt text.
        controlled_prompt_text : str
            Controlled prompt sent to API.
        date_generated : str
            Generation timestamp.
        error_message : str
            API error message.

        Returns
        -------
        dict[str, Any]
            Failed generation record.
        """
        return {
            "text_id": text_id,
            "prompt_id": prompt_id,
            "genre": genre,
            "theme": theme,
            "topic": topic,
            "condition": "controlled_prompt",
            "model_family": self.model_family,
            "model_name": self.model_name,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "date_generated": date_generated,
            "original_prompt_text": original_prompt_text,
            "controlled_prompt_text": controlled_prompt_text,
            "raw_output_text": "",
            "length_controlled_output_text": "",
            "output_text": "",
            "raw_word_count": 0,
            "final_word_count": 0,
            "length_status": "failed",
            "within_10_percent_word_limit": False,
            "generation_attempt": 0,
            "was_rewritten_for_length": False,
            "api_status": "failed",
            "error_message": error_message,
            "notes": "API call failed.",
        }


class OpenAIModelRunner(BaseModelRunner):
    """
    Runner for OpenAI models.
    """

    def _call_api(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content or ""


class ClaudeModelRunner(BaseModelRunner):
    """
    Runner for Anthropic Claude models.
    """

    def _call_api(self, prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)

        response = client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        text_parts: list[str] = []

        for block in response.content:
            block_type = getattr(block, "type", None)
            block_text = getattr(block, "text", "")

            if block_type == "text" and isinstance(block_text, str):
                text_parts.append(block_text)

        return "\n".join(text_parts).strip()


class GeminiModelRunner(BaseModelRunner):
    """
    Runner for Google Gemini models using the Google GenAI SDK.

    Gemini 2.5 models can use internal thinking tokens. For this project,
    we disable thinking because the task is controlled text generation,
    not complex reasoning.
    """

    def _call_api(self, prompt: str) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=900,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0,
                ),
            ),
        )

        text = getattr(response, "text", None)

        if isinstance(text, str) and text.strip():
            return text.strip()

        text_parts: list[str] = []

        candidates = getattr(response, "candidates", []) or []

        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", []) if content else []

            for part in parts:
                part_text = getattr(part, "text", "")

                if isinstance(part_text, str) and part_text.strip():
                    text_parts.append(part_text.strip())

        return "\n".join(text_parts).strip()


class DeepSeekModelRunner(BaseModelRunner):
    """
    Runner for DeepSeek models using the OpenAI-compatible API format.
    """

    def _call_api(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
        )

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content or ""


class MistralModelRunner(BaseModelRunner):
    """
    Runner for Mistral AI models.
    """

    def _call_api(self, prompt: str) -> str:
        from mistralai.client import Mistral

        client = Mistral(api_key=self.api_key)

        response = client.chat.complete(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts: list[str] = []

            for item in content:
                item_text = getattr(item, "text", "")

                if isinstance(item_text, str):
                    text_parts.append(item_text)

            return "\n".join(text_parts).strip()

        return ""
    

def get_model_runner(
    model_key: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> BaseModelRunner:
    """
    Factory function that returns the correct model runner.

    Parameters
    ----------
    model_key : str
        Model provider key: OpenAI, Claude, Gemini, DeepSeek, or Mistral.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum output tokens.

    Returns
    -------
    BaseModelRunner
        Provider-specific model runner.
    """
    runners = {
        "OpenAI": OpenAIModelRunner,
        "Claude": ClaudeModelRunner,
        "Gemini": GeminiModelRunner,
        "DeepSeek": DeepSeekModelRunner,
        "Mistral": MistralModelRunner,
    }

    if model_key not in runners:
        available = list(runners.keys())
        raise ValueError(f"Unknown model '{model_key}'. Available models: {available}")

    return runners[model_key](
        model_key=model_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def ensure_raw_output_file(output_path: Path = RAW_OUTPUTS_PATH) -> None:
    """
    Create model_outputs_raw.csv with the correct schema if it does not exist.

    Parameters
    ----------
    output_path : Path
        Path to raw output CSV.
    """
    create_required_directories()

    if not output_path.exists():
        pd.DataFrame(columns=RAW_OUTPUT_COLUMNS).to_csv(output_path, index=False)


def append_records_to_csv(
    records: list[dict[str, Any]],
    output_path: Path = RAW_OUTPUTS_PATH,
) -> None:
    """
    Append generated records to the raw output CSV.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Generated response records.
    output_path : Path
        Path to output CSV.
    """
    ensure_raw_output_file(output_path)

    new_data = pd.DataFrame(records)

    for column in RAW_OUTPUT_COLUMNS:
        if column not in new_data.columns:
            new_data[column] = ""

    new_data = new_data[RAW_OUTPUT_COLUMNS]

    file_exists = output_path.exists()
    has_content = file_exists and output_path.stat().st_size > 0

    new_data.to_csv(
        output_path,
        mode="a",
        index=False,
        header=not has_content,
    )


def generate_outputs_from_prompts(
    prompts_path: Path,
    model_key: str,
    output_path: Path = RAW_OUTPUTS_PATH,
    limit: int | None = None,
    start_index: int = 0,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> pd.DataFrame:
    """
    Generate model outputs from a prompt CSV file.

    Parameters
    ----------
    prompts_path : Path
        Path to Prompts.csv.
    model_key : str
        Model provider key.
    output_path : Path
        Path where raw outputs should be saved.
    limit : int | None
        Optional number of prompts to process.
    start_index : int
        Row index to start from.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum output tokens.

    Returns
    -------
    pd.DataFrame
        Generated output records.
    """
    prompts_df = load_prompts(prompts_path)
    validate_prompts(prompts_df)

    prompts_df = prompts_df.iloc[start_index:]

    if limit is not None:
        prompts_df = prompts_df.head(limit)

    runner = get_model_runner(
        model_key=model_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    records = []

    for _, prompt_row in tqdm(
        prompts_df.iterrows(),
        total=len(prompts_df),
        desc=f"Generating outputs with {model_key}",
    ):
        record = runner.generate_for_prompt(prompt_row)
        records.append(record)

        append_records_to_csv([record], output_path=output_path)

    return pd.DataFrame(records)