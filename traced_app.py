from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

os.environ["LANGSMITH_TRACING"]= "true"
os.environ["LANGSMITH_API_KEY"]=langsmith_api_key
os.environ["LANGSMITH_PROJECT"]="my-langsmith-assignment"
os.environ["OPENAI_API_KEY"]=openai_api_key

prompt = ChatPromptTemplate.from_template("{questions}\nTopic:{topic}")
    
model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model | StrOutputParser()

questions = [
    "what is this",
    "where can we use it",
    "How can we use it in a real-world example?, give me a use case"
]
for q in questions:
    answer = chain.invoke({"questions":q,"topic":"Langsmith",})
    print(answer)