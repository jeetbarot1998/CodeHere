from langchain.agents import initialize_agent, Tool
from langchain_ollama import OllamaLLM
from googlesearch import search

def search_web(query):
    results = list(search(query, num_results=3))
    return "\n".join(results)

tools = [Tool(name="Search", func=search_web, description="Search the web")]

llm = OllamaLLM(model="gemma3:4B")

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

response = agent.invoke("What are the top AI models in 2025?")
print(response)
