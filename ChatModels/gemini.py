from langchain_google_genai import ChatGoogleGenerativeAI # import the ChatGoogleGenerativeAI class from the langchain_google_genai module
from dotenv import load_dotenv   # import the load_dotenv function from the dotenv module to load environment variables from a .env file
load_dotenv()              # load the environment variables from the .env file  

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")             # create an instance of the ChatGoogleGenerativeAI class with the specified model, in this case, "gemini-2.5-flash". This will allow us to interact with the Gemini 2.5 Flash model using the langchain library.

result=model.invoke("Name all continents")         # invoke the model with a prompt, in this case, "What is the capital of France?", and get the response. This will send a request to the Google GenAI API and return the answer to the question about the capital of France.

print(result.content)                           # print the content of the result to the console. The result is an object that contains various information about the response from the model, and we are specifically interested in the 'content' attribute, which holds the actual answer to our question.    