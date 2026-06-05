import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint # import the ChatHuggingFace class from the langchain_huggingface module
from dotenv import load_dotenv   # import the load_dotenv function from the dotenv module to load environment variables from a .env file
load_dotenv(r"E:\Langchain Model\.env")              # load the environment variables from the .env file


print("TOKEN =", os.getenv("HUGGINGFACEHUB_API_TOKEN"))
print("Current Directory:", os.getcwd())

llm= HuggingFaceEndpoint(                                     ## create an instance of the HuggingFaceEndpoint class with the specified model, in this case, "TinyLlama/TinyLlama-1.1B-Chat-v1.0". This will allow us to interact with the TinyLlama 1.1B Chat model using the langchain library.
    repo_id= "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)             # create an instance of the ChatHuggingFace class with the specified model, in this case, "gpt-4". This will allow us to interact with the GPT-4 model using the langchain library.

result=model.invoke("What is the capital of France?")         # invoke the model with a prompt, in this case, "What is the capital of France?", and get the response. This will send a request to the HuggingFace API and return the answer to the question about the capital of France.
print(result.content)                           # print the content of the result to the console. The result is an object that contains various information about the response from the model, and we are specifically interested in the 'content' attribute, which holds the actual answer to our question.

