from langchain_huggingface import (
    HuggingFacePipeline,
    ChatHuggingFace
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature":0.7
    }
)

model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_template(
"""
You are an expert teacher.

Generate {num_questions} multiple choice questions.

Topic:
{topic}

Difficulty:
{difficulty}

Rules:

1. Four options only.
2. One correct answer.
3. Provide explanation.
4. Number every question.
5. Output must be clean.

Format:

Question 1:
A.
B.
C.
D.

Answer:
Explanation:
"""
)

parser = StrOutputParser()

chain = prompt | model | parser

topic = input("Enter Topic: ")

difficulty = input(
    "Difficulty (Beginner/Intermediate/Advanced): "
)

num_questions = int(
    input("Number of Questions: ")
)
result = chain.invoke(
    {
        "topic": topic,
        "difficulty": difficulty,
        "num_questions": num_questions
    }
)
print(result)