import os

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from openai import OpenAI
from pydantic import BaseModel, Field

from .models import GatewayDecision, RiskLevel, UserRole, ValidationFinding, ValidationResult
from .prompts import PromptRegistry


class LLMFinding(BaseModel):
    rule_id: str = Field(description="Stable identifier for the validation finding.")
    message: str = Field(description="Short explanation of the safety finding.")
    risk_level: RiskLevel = Field(description="Finding risk level: low, medium, or high.")


class LLMValidationOutput(BaseModel):
    decision: GatewayDecision = Field(description="ALLOW, BLOCK, or ESCALATE_FOR_APPROVAL.")
    risk_level: RiskLevel = Field(description="Overall risk level.")
    findings: list[LLMFinding] = Field(default_factory=list)


class LLMValidationResponse(BaseModel):
    validation: ValidationResult
    prompt_record: dict[str, str] = Field(default_factory=dict)
    used_llm: bool = False


class LLMSafetyEvaluator:
    def __init__(self, prompts: PromptRegistry | None = None):
        load_dotenv()
        self.prompts = prompts or PromptRegistry()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.output_parser = PydanticOutputParser(pydantic_object=LLMValidationOutput)

    def validate_input(
        self,
        query: str,
        user_role: UserRole,
        document_text: str,
    ) -> LLMValidationResponse:
        document_preview = document_text[:5000]
        system_prompt, user_prompt = self.prompts.build_input_validation_prompt(
            query=query,
            user_role=user_role.value,
            document_preview=document_preview,
            format_instructions=self.output_parser.get_format_instructions(),
        )
        return self._validate(system_prompt, user_prompt)

    def validate_output(
        self,
        query: str,
        answer: str,
        retrieved_context: list[str],
    ) -> LLMValidationResponse:
        system_prompt, user_prompt = self.prompts.build_output_validation_prompt(
            query=query,
            answer=answer,
            retrieved_context=retrieved_context,
            format_instructions=self.output_parser.get_format_instructions(),
        )
        return self._validate(system_prompt, user_prompt)

    def _validate(self, system_prompt: str, user_prompt: str) -> LLMValidationResponse:
        prompt_record = {
            "model": self.model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
        if not self.client:
            return LLMValidationResponse(validation=ValidationResult(), prompt_record=prompt_record)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            parsed = self.output_parser.parse(response.choices[0].message.content or "{}")
        except Exception:
            return LLMValidationResponse(validation=ValidationResult(), prompt_record=prompt_record)

        findings = [
            ValidationFinding(
                rule_id=finding.rule_id,
                message=finding.message,
                risk_level=finding.risk_level,
            )
            for finding in parsed.findings
        ]
        return LLMValidationResponse(
            validation=ValidationResult(
                decision=parsed.decision,
                risk_level=parsed.risk_level,
                findings=findings,
            ),
            prompt_record=prompt_record,
            used_llm=True,
        )
