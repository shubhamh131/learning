from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv


load_dotenv()
model = ChatGroq(model="openai/gpt-oss-20b")


class Review(TypedDict):
    key_themes: Annotated[
        list[str], "write dowm all key themes disscussed in the review"
    ]
    summary: Annotated[str, "A brief summary of review"]
    sentiment: Annotated[
        str, "Return sentiment of review either Negative, Posivtive, Mixed"
    ]
    pros: Annotated[Optional[list[str]], "write pros inside a list"]
    cons: Annotated[Optional[list[str]], "write cons inside a list"]


structured_model = model.with_structured_output(Review, method="json_schema")

result = structured_model.invoke(
    "*Tenet* is a visually stunning and ambitious film with an intriguing concept, impressive action sequences, and an intense soundtrack. Christopher Nolan delivers a unique cinematic experience that rewards viewers who enjoy complex storytelling. However, the plot can feel confusing, the dialogue is sometimes difficult to follow, and the characters could have been developed more deeply. While the time-manipulation concept is fascinating, the film's complexity occasionally comes at the expense of emotional connection. Overall, *Tenet* is a technically brilliant and thought-provoking movie, but its storytelling may not appeal to everyone."
)

print(result)
