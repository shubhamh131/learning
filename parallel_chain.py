from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()
model1 = ChatGroq(model="openai/gpt-oss-20b")
model2 = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

prompt1 = PromptTemplate(
    template="Genrate short note on provided topic {text}", input_variables=["text"]
)
prompt2 = PromptTemplate(
    template="Genrate quiz on provided topic {text}", input_variables=["text"]
)
prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into single document \n {notes} and {quiz}",
    input_variables=["notes", "quiz"],
)
parser = StrOutputParser()


parallel_chains = RunnableParallel(
    {"notes": prompt1 | model1 | parser, "quiz": prompt2 | model2 | parser}
)

chain = prompt3 | model1 | parser

merge_chain = parallel_chains | chain


text = """Deep Agents overview

Copy page

Deep Agents is the agent harness for complex, multi-step work. It adds planning, a virtual filesystem for context, subagent spawning, and long-term memory on top of LangChain and LangGraph.
Start with Managed Deep Agents to run production agents without having to manage infrastructure. Alternatively, for maximum flexibility while using the same Deep Agents harness, use the Deep Agents SDK and host agents yourself, or run a custom LangSmith Deployment.
​
"""
result = merge_chain.invoke({"text": text})

print(result)
