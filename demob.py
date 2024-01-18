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
import getpass
import os


os.environ["TAVILY_API_KEY"] = "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"
tool = TavilySearchResults(max_results=1)

tools = [tool]


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 45  # Change this value based on your model and your GPU VRAM pool.
n_batch = 2048  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    max_tokens = 4000,
    # n_predict = 1000,
    n_ctx=2048,
    model_path="/Users/velocity/Documents/Holder/Project/CodingStuff/VICUNA/llama.cpp/models/Mistral/mixtral-8x7b-instruct-v0.1.Q6_K.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
    temperature = 0.0,

)


instructions = """You are an assistant who has access to google search and thinks things out step by step."""
base_prompt = hub.pull("hwchase17/react")
prompt = base_prompt.partial(instructions=instructions)

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_executor.invoke({"input": "How much will the apple vision pro cost"})
agent_executor.invoke({"input": "What happened at the burning man floods"})
# agent_executor.invoke({"input": "What was the GPT4 Release date"})



