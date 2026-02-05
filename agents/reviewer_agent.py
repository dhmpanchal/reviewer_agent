from __future__ import annotations

from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

import constants

load_dotenv()


class ReviewerAgent:
    def __init__(self, tag_name: str) -> None:
        self.tag_name = tag_name
        self.parser = JsonOutputParser()
        self.llm = ChatOpenAI(
            model=constants.OLLAMA_MODEL_ID,
            base_url=constants.OLLAMA_MODEL_BASE_URL,
            api_key="ollama",
            temperature=0.2,
            max_tokens=600,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        self.prompt_template = """
You are a clinical QA reviewer (reflexion agent).

You will be given:
- Task prompt
- Retrieved patient context
- The task output (JSON) produced by another model

Your job is to evaluate whether the task output is correct and supported by the patient context.

Checks:
- Every listed condition must be supported by explicit evidence in the patient context. If not, flag it.
- Dates must be in MM-DD-YYYY. If unknown, the model should not invent exact dates; it can flag as missing/unclear.
- status must be either ongoing or cleaned up; flag inconsistencies.
- Identify if obvious major/chronic conditions appear in the context but are missing from the output.

Output JSON STRICTLY in this format:
{{
  "verdict": "ok" | "needs_fix",
  "comment": "2-3 short lines with the review"
}}

Be concise. Do not include any extra keys or text.

Task prompt:
{task_prompt}

Patient context:
{patient_context}

Task output JSON:
{task_output}
"""

        self.review_prompt = ChatPromptTemplate.from_template(self.prompt_template)

    def review(self, task_prompt: str, patient_context: str, task_output: Any) -> Dict[str, str]:
        chain = self.review_prompt | self.llm | self.parser
        return chain.invoke(
            {
                "task_prompt": task_prompt,
                "patient_context": patient_context,
                "task_output": task_output,
            },
            config=RunnableConfig(tags=[f"doc:{self.tag_name}"]),
        )
