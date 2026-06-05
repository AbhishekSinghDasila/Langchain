from langchain_openai import ChatOpenAI # import the ChatOpenAI class from the langchain_openai module
from dotenv import load_dotenv   # import the load_dotenv function from the dotenv module to load environment variables from a .env file
load_dotenv()              # load the environment variables from the .env file

model=ChatOpenAI(model="gpt-4",temperature=0.7,max_completion_tokens=1024)             # create an instance of the ChatOpenAI class with the specified model, in this case, "gpt-4". This will allow us to interact with the GPT-4 model using the langchain library.

result=model.invoke("What is the capital of France?")         # invoke the model with a prompt, in this case, "What is the capital of France?", and get the response. This will send a request to the OpenAI API and return the answer to the question about the capital of France.

print(result.content)                           # print the content of the result to the console. The result is an object that contains various information about the response from the model, and we are specifically interested in the 'content' attribute, which holds the actual answer to our question.


## use of temperature parameter in the ChatOpenAI class:
# The temperature parameter in the ChatOpenAI class controls the randomness of the model's responses.
#  A lower temperature (e.g., 0.2) will make the model's responses more deterministic and focused,
#  while a higher temperature (e.g., 0.8) will make the responses more diverse and creative. 
# In this example, we set the temperature to 0.7, which allows for a good balance between creativity and coherence in the model's responses.  


# The max_completion_tokens parameter in the ChatOpenAI class specifies the maximum number of tokens that the model can generate in its response.
# Tokens are the basic units of text that the model processes, and they can be as short as one character or as long as one word. By setting max_completion_tokens to 1024, we are allowing the model to generate responses that can be up to 1024 tokens long. 
# This is useful for ensuring that the model's responses are not too short and can provide more detailed information when needed. 

