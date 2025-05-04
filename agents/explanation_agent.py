import logging 
import json
from typing import Dict,Any,List,Optional
from datetime import SystemMessage, HumanMessage
import re

logger = logging.getLogger(__name__)

class ExplanationAgent:
    """
    Agent responsible for generating explanations of the AI system's analysis
    and recommendation for healthcare providers and patients.
    """
    def __init__(self,config: Dict[str,Any],knowledge_agent=None,llm=None):
        """
        Initialize the explanation agent
        
        Args:
            config: Configuration dictionary
            knowledge_agent: Optional KnowledgeAgent for medical knowledge lookup
            llm: LLM instance from langchain
        """
        self.config= config
        self.knowledge_agent=knowledge_agent
        self.llm =llm

        self.templates=self._load_templates()
        
        