import guidance 
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import BaseTool, StructuredTool, tool

from langchain_community.tools.tavily_search import TavilySearchResults
import os

os.environ["TAVILY_API_KEY"] = "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"
tool = TavilySearchResults(max_results=1)

lm = guidance.models.LlamaCpp("/Users/velocity/Documents/Holder/Project/CodingStuff/VICUNA/llama.cpp/models/Mistral/mistral-7b-instruct-v0.1.Q6_K.gguf", n_gpu_layers=-1)
guidance.models = lm

valid_answers = ['Action', 'Final Answer']
valid_tools = ['Google Search']

prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer the following questions as best you can. You have access to the following tools:

Google Search: A wrapper around Google Search. Useful for when you need to answer questions about current events. The input is the question to search relavant information.

Strictly use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Google Search]
Action Input: the input to the action, should be a question.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

For examples:
Question: How old is CEO of Microsoft wife?
Thought: First, I need to find who is the CEO of Microsoft.
Action: Google Search
Action Input: Who is the CEO of Microsoft?
Observation: Satya Nadella is the CEO of Microsoft.
Thought: Now, I should find out Satya Nadella's wife.
Action: Google Search
Action Input: Who is Satya Nadella's wife?
Observation: Satya Nadella's wife's name is Anupama Nadella.
Thought: Then, I need to check Anupama Nadella's age.
Action: Google Search
Action Input: How old is Anupama Nadella?
Observation: Anupama Nadella's age is 50.
Thought: I now know the final answer.
Final Answer: Anupama Nadella is 50 years old.

### Input:
{{question}}

### Response:
Question: {{question}}
Thought: {{gen 'thought' stop='\\n'}}
Action: {{select 'tool_name' options=valid_tools}}
Action Input: {{gen 'actInput' stop='\\n'}}
Observation:{{search actInput}}
Thought: {{gen 'thought2' stop='\\n'}}
Final Answer: {{gen 'final' stop='\\n'}}"""

def searchGoogle(t):    
    return tool.invoke(t)

# prompt = lm + prompt_template
prompt = prompt_template
result = prompt(question='Is Eminem a football player?', search=searchGoogle, valid_answers=valid_answers, valid_tools=valid_tools)