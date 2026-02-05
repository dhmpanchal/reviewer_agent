from typing import List, Any
from langchain_openai import ChatOpenAI
from langchain_community.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.chains import LLMChain
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import PydanticOutputParser

import os
import re
from vector_stores.vector_helper import VectorHelper
from schemas.rag_tool_parameters import PatientSearchInput
import constants
from tools.rag_tool import rag_patient_retrieval

from dotenv import load_dotenv
load_dotenv()


class RetrievalAgent:
    """
    This class implements a retrieval agent that uses a vector database to retrieve relevant documents
    based on a given query.
    """
    def __init__(self, tag_name) -> None:
        self.tag_name = tag_name
        self.vectorstore = VectorHelper()
        self.rag_llm = ChatOpenAI(
            model=constants.RETRIEVAL_MODEL_ID,
            base_url=constants.OLLAMA_MODEL_BASE_URL,  # Ollama endpoint
            api_key="ollama",
            temperature=0.3,
        )
        self.tools = []
        self.agent_system_prompt = SystemMessage(
            content="""
            You are a retrieval-only agent.

            Your task:
            - Read the user request carefully
            - Generate a focused semantic search query to retrieve relevant medical information
            - Call the tool patient_document_search with:
                - query = your generated semantic search query 
                - file_path = the patient document path provided in the conversation context
            - DO NOT analyze or summarize results

            The tool output will be returned directly.
            """
        )
        self.rag_agent = create_agent(
            model=self.rag_llm,          # normal llama3.2
            tools=self.get_tools(),
            system_prompt=self.agent_system_prompt,
        )
    
    def get_tools(self):
        self.tools.append(rag_patient_retrieval)
        return self.tools
    
    def run_retrieval_agent(self, user_query: str, file_path: str) -> str:
        result = self.rag_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"""
                        User question: {user_query}
                        file_path: {file_path}
                        """
                    )
                ]
            },
            config=RunnableConfig(
                tags=[f"doc:{self.tag_name}"]
            )
        )

        # Last message MUST be tool output (string)
        return result["messages"][-1].content