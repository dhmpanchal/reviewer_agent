from typing import List, Any
from langchain_openai import ChatOpenAI
from langchain_community.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.chains import LLMChain
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import PydanticOutputParser

import os
import re
from vector_stores.vector_helper import VectorHelper
from schemas.rag_tool_parameters import PatientSearchInput
import constants


from dotenv import load_dotenv
load_dotenv()

class FetchDiagnosisChain:
    def __init__(self, tag_name) -> None:
        self.tag_name = tag_name
        self.parser = JsonOutputParser()
        self.task_llm = ChatOpenAI(
            model=constants.OLLAMA_MODEL_ID,
            base_url=constants.OLLAMA_MODEL_BASE_URL,  # Ollama endpoint
            api_key="ollama",
            temperature=0.3,
            max_tokens=2500,
            model_kwargs={
                "response_format": {"type": "json_object"}
            }
        )
        self.prompt_template = """
            You are clinical experts who review the given patient infromation give below as a context. Your task is to extract a list of all major or chronic medical conditions mentioned in the given patient infromations and need to find dates of the medical conditions from when it detected and calculate the proper dates in formats and from when it was cleaned up or still it is on going.

            For each condition, provide:
            - key: The name of the condition (e.g., IBS, Depression, Diabetes)
            - value: A brief reason, context, or supporting detail
            - start_date: The start date of the condition mention in the patient information
            - end_date: The end date of the condition mention in the patient information
            - status: ongoing or cleaned up

            Return the output strictly following these format instructions:  
            {{
            "major_conditions": [
                {{"key": "Condition Name", "value": "Supporting detail or reason","start_date":"MM-DD-YYYY","end_date":"MM-DD-YYYY","status":"ongoing/cleaned up"}},
                {{"key": "Condition Name", "value": "Supporting detail or reason","start_date":"MM-DD-YYYY","end_date":"MM-DD-YYYY","status":"ongoing/cleaned up"}},
                ...
            ]
            }}

            If no major conditions are found:
            {{
            "major_conditions": []
            }}

            context:
            {patient_info}

            Ensure that the output is strictly in JSON format without any additional text.
        """
        self.task_prompt = ChatPromptTemplate.from_template(self.prompt_template)
    
    def run_task_chain(self, patient_info: str):
        task_chain = self.task_prompt | self.task_llm | self.parser
        response_text = task_chain.invoke(
            {"patient_info": patient_info},
            config=RunnableConfig(tags=[f"doc:{self.tag_name}"]),
        )
        return response_text