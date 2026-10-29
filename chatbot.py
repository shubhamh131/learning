from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

chathistory = [SystemMessage(content="You are a helpful assistant")]

while True:
    user_input = input("You: ")
    chathistory.append(HumanMessage(user_input))
    if user_input == "exit":
        break
    result = model.invoke(chathistory)
    chathistory.append(AIMessage(result.content))
    print("AI: ", result.content)
