from langchain_anthropic import ChatAnthropic # import the ChatAnthropic class from the langchain_anthropic module
from dotenv import load_dotenv   # import the load_dotenv function from the dotenv module to load environment variables from a .env file
load_dotenv()              # load the environment variables from the .env file

model= ChatAnthropic(model="claude-sonnet-4.1-20241022",temperature=0.7,max_completion_tokens=1024)             # create an instance of the ChatAnthropic class with the specified model, in this case, "claude-2". This will allow us to interact with the Claude 2 model using the langchain library.

result=model.invoke("What is the capital of France?")         # invoke the model with a prompt, in this case, "What is the capital of France?", and get the response. This will send a request to the Anthropic API and return the answer to the question about the capital of France.

print(result.content)                           # print the content of the result to the console. The result is an object that contains various information about the response from the model, and we are specifically interested in the 'content' attribute, which holds the actual answer to our question.    

