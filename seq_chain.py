from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableBranch, RunnableLambda


load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="give the sentiment of the feedback"
    )


parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template="classify the sentiment of the following feedback text into negative or positive \n {text} \n{format_instruction}",
    input_variables=["text"],
    partial_variables={"format_instruction": parser2.get_format_instructions()},
)

parser = StrOutputParser()

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="give a responce for positive feedback {text}", input_variables=["text"]
)
prompt3 = PromptTemplate(
    template="give a responce for this negative feedback {text}",
    input_variables=["text"],
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "could find sentiment"),
)

chain = classifier_chain | branch_chain

result = chain.invoke({"text": "the movie is terrible"})

print(result)
