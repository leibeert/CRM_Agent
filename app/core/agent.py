from typing import List, Dict, Any
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool
from langchain.chains import LLMChain
from config.settings import settings
import re

class RecruitmentPromptTemplate:
    def __init__(self, template: str, tools: List[Tool]):
        self.template = template
        self.tools = tools
        self.input_variables = ["input", "agent_scratchpad"]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("agent_scratchpad")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"\nAction: {action}\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class RecruitmentAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=settings.TEMPERATURE,
            model_name=settings.MODEL_NAME,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()

    def find_matching_candidates(self, job_description: str) -> List[Dict[str, Any]]:
        """
        Find candidates matching the job description using LangChain agent
        """
        try:
            # Use the agent to find matching candidates
            result = self.agent_executor.run(
                f"Find candidates matching this job description: {job_description}"
            )
            
            # Parse the result to get candidates
            # This is a placeholder - you'll need to implement proper parsing
            # based on your database structure
            return [
                {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "skills": ["Python", "FastAPI", "SQL"],
                    "experience": 5,
                    "education": "Bachelor's in Computer Science"
                },
                {
                    "name": "Jane Smith",
                    "email": "jane@example.com",
                    "skills": ["Python", "Django", "PostgreSQL"],
                    "experience": 3,
                    "education": "Master's in Software Engineering"
                }
            ]
        except Exception as e:
            print(f"Error finding candidates: {str(e)}")
            return []

    def analyze_job_description(self, job_description: str) -> Dict[str, Any]:
        """
        Analyze job description to extract requirements using LangChain
        """
        try:
            # Use the agent to analyze the job description
            result = self.agent_executor.run(
                f"Analyze this job description and extract requirements: {job_description}"
            )
            
            # Parse the result to get requirements
            requirements = {
                "skills": [],
                "experience": 0,
                "education": "",
                "other_requirements": []
            }
            
            # TODO: Implement proper parsing of the agent's response
            # This is a placeholder implementation
            
            return requirements
            
        except Exception as e:
            print(f"Error analyzing job description: {str(e)}")
            return {
                "skills": [],
                "experience": 0,
                "education": "",
                "other_requirements": []
            }

    def _create_tools(self) -> List[BaseTool]:
        """
        Create tools for the LangChain agent
        """
        return [
            Tool(
                name="SearchCandidates",
                func=self._search_candidates,
                description="Search for candidates in the database based on criteria"
            ),
            Tool(
                name="AnalyzeRequirements",
                func=self._analyze_requirements,
                description="Analyze job requirements from a description"
            )
        ]

    def _create_agent(self) -> AgentExecutor:
        """
        Create the LangChain agent
        """
        template = """
        You are a recruitment expert. Your task is to help find the best candidates for job positions.
        
        Use the following tools:
        {tools}
        
        Use the following format:
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Question: {input}
        {agent_scratchpad}
        """

        prompt = RecruitmentPromptTemplate(
            template=template,
            tools=self.tools
        )

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            allowed_tools=[tool.name for tool in self.tools],
            stop=["\nObservation:"],
            handle_parsing_errors=True
        )

        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True
        )

    def _search_candidates(self, query: str) -> str:
        """
        Search for candidates in the database
        This is a placeholder - implement actual database search
        """
        # TODO: Implement actual database search
        return "Found candidates: John Doe, Jane Smith"

    def _analyze_requirements(self, description: str) -> str:
        """
        Analyze job requirements from a description
        This is a placeholder - implement actual requirement analysis
        """
        # TODO: Implement actual requirement analysis
        return "Required skills: Python, SQL, FastAPI" 