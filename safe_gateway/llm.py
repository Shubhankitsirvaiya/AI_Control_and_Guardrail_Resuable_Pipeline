import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI

from .prompts import PromptRegistry
from .rag import generate_answer as generate_fallback_answer


class OpenAIAnswerGenerator:
    def __init__(self, prompts: PromptRegistry | None = None):
        load_dotenv()
        self.prompts = prompts or PromptRegistry()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.output_parser = StrOutputParser()

    def generate(self, query: str, contexts: list[str]) -> tuple[str, dict[str, str]]:
        system_prompt, user_prompt = self.prompts.build_answer_prompt(query, contexts)
        prompt_record = {
            "model": self.model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

        if not self.client:
            return generate_fallback_answer(query, contexts), prompt_record

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_answer = response.choices[0].message.content or ""
        return self.output_parser.parse(raw_answer), prompt_record
