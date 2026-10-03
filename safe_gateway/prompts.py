import os
from pathlib import Path

from dotenv import load_dotenv


class PromptRegistry:
    def __init__(self, prompt_dir: str = "prompts"):
        load_dotenv()
        self.prompt_dir = Path(prompt_dir)
        self.use_case_context_path = Path(
            os.getenv("DOCUMENT_USE_CASE_CONTEXT_PATH", "prompts/use_case_legal_context.txt")
        )

    def load(self, name: str) -> str:
        path = self.prompt_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Prompt file {path} was not found.")
        return path.read_text(encoding="utf-8")

    def load_use_case_context(self) -> str:
        if not self.use_case_context_path.exists():
            raise FileNotFoundError(f"Use-case context file {self.use_case_context_path} was not found.")
        return self.use_case_context_path.read_text(encoding="utf-8")

    def build_answer_prompt(self, query: str, contexts: list[str]) -> tuple[str, str]:
        system_prompt = self.load("answer_system.txt")
        user_template = self.load("answer_user.txt")
        context = "\n\n---\n\n".join(contexts)
        user_prompt = user_template.format(
            use_case_context=self.load_use_case_context(),
            query=query,
            context=context,
        )
        return system_prompt, user_prompt

    def build_document_validation_prompt(
        self,
        filename: str,
        pages: list[str],
        format_instructions: str,
    ) -> tuple[str, str]:
        system_prompt = self.load("document_validation_system.txt")
        user_template = self.load("document_validation_user.txt")
        page_text = "\n\n".join(
            f"Page {index}\n{page[:2500]}" for index, page in enumerate(pages, start=1)
        )
        user_prompt = user_template.format(
            use_case_context=self.load_use_case_context(),
            filename=filename,
            pages=page_text,
            format_instructions=format_instructions,
        )
        return system_prompt, user_prompt

    def build_input_validation_prompt(
        self,
        query: str,
        user_role: str,
        document_preview: str,
        format_instructions: str,
    ) -> tuple[str, str]:
        system_prompt = self.load("input_validation_system.txt")
        user_template = self.load("input_validation_user.txt")
        user_prompt = user_template.format(
            use_case_context=self.load_use_case_context(),
            user_role=user_role,
            query=query,
            document_preview=document_preview,
            format_instructions=format_instructions,
        )
        return system_prompt, user_prompt

    def build_output_validation_prompt(
        self,
        query: str,
        answer: str,
        retrieved_context: list[str],
        format_instructions: str,
    ) -> tuple[str, str]:
        system_prompt = self.load("output_validation_system.txt")
        user_template = self.load("output_validation_user.txt")
        user_prompt = user_template.format(
            use_case_context=self.load_use_case_context(),
            query=query,
            retrieved_context="\n\n---\n\n".join(retrieved_context),
            answer=answer,
            format_instructions=format_instructions,
        )
        return system_prompt, user_prompt
