import os

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from openai import OpenAI
from pydantic import BaseModel, Field

from .prompts import PromptRegistry
from .rag import tokenize


class PageAdjustment(BaseModel):
    page: int
    adjustment: int
    reason: str


class DocumentConfidenceResult(BaseModel):
    confidence: int
    allowed: bool
    summary: str
    page_adjustments: list[PageAdjustment] = Field(default_factory=list)
    prompt_record: dict[str, str] = Field(default_factory=dict)


class DocumentConfidenceLLMOutput(BaseModel):
    page_adjustments: list[PageAdjustment] = Field(
        description="One +5 or -5 relevance adjustment per reviewed page."
    )
    summary: str = Field(description="Short summary explaining the suitability decision.")


class DocumentConfidenceEvaluator:
    def __init__(self, prompts: PromptRegistry | None = None):
        load_dotenv()
        self.prompts = prompts or PromptRegistry()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.confidence_threshold = int(os.getenv("DOCUMENT_CONFIDENCE_THRESHOLD", "45"))
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.output_parser = PydanticOutputParser(pydantic_object=DocumentConfidenceLLMOutput)

    def evaluate(self, filename: str, pages: list[str]) -> DocumentConfidenceResult:
        pages_to_check = pages[:7]
        system_prompt, user_prompt = self._build_prompt(filename, pages_to_check)
        prompt_record = {
            "model": self.model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

        if not pages_to_check:
            return DocumentConfidenceResult(
                confidence=0,
                allowed=False,
                summary="No parseable document text was found.",
                prompt_record=prompt_record,
            )

        if not self.client:
            result = self._heuristic_evaluate(filename, pages_to_check)
            result.prompt_record = prompt_record
            return result

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            result = self._parse_llm_result(content)
        except Exception:
            result = self._heuristic_evaluate(filename, pages_to_check)
            result.summary = "Fallback keyword confidence evaluation used because primary LLM validation failed."
        result.prompt_record = prompt_record
        return result

    def _build_prompt(self, filename: str, pages: list[str]) -> tuple[str, str]:
        return self.prompts.build_document_validation_prompt(
            filename=filename,
            pages=pages,
            format_instructions=self.output_parser.get_format_instructions(),
        )

    def _parse_llm_result(self, content: str) -> DocumentConfidenceResult:
        parsed = self.output_parser.parse(content)
        adjustments = []
        confidence = 50
        for item in parsed.page_adjustments[:7]:
            adjustment = 5 if int(item.adjustment) > 0 else -5
            confidence += adjustment
            adjustments.append(
                PageAdjustment(
                    page=int(item.page),
                    adjustment=adjustment,
                    reason=item.reason,
                )
            )
            if confidence >= 70:
                break
        confidence = max(0, min(100, confidence))
        return DocumentConfidenceResult(
            confidence=confidence,
            allowed=confidence >= self.confidence_threshold,
            summary=parsed.summary,
            page_adjustments=adjustments,
        )

    def _heuristic_evaluate(self, filename: str, pages: list[str]) -> DocumentConfidenceResult:
        legal_terms = {
            "agreement",
            "contract",
            "clause",
            "compliance",
            "confidentiality",
            "liability",
            "regulation",
            "privacy",
            "policy",
            "terms",
            "legal",
            "obligation",
            "risk",
            "governing",
            "jurisdiction",
        }
        confidence = 50
        adjustments: list[PageAdjustment] = []
        for index, page in enumerate(pages[:7], start=1):
            tokens = set(tokenize(page))
            adjustment = 5 if tokens & legal_terms else -5
            confidence += adjustment
            adjustments.append(
                PageAdjustment(
                    page=index,
                    adjustment=adjustment,
                    reason="Legal-use-case keywords found." if adjustment > 0 else "Low legal-use-case signal.",
                )
            )
            if confidence >= 70:
                break
        confidence = max(0, min(100, confidence))
        return DocumentConfidenceResult(
            confidence=confidence,
            allowed=confidence >= self.confidence_threshold,
            summary="Fallback keyword confidence evaluation used because OpenAI is not configured.",
            page_adjustments=adjustments,
        )
