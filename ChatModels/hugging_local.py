from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline # import the HuggingFaceHub class from the langchain_huggingface module


llm = HuggingFacePipeline.from_model_id(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",         # create an instance of the HuggingFacePipeline class using the from_model_id method, which allows us to specify the model we want to use by its ID. In this case, we are using the "TinyLlama/TinyLlama-1.1B-Chat-v1.0" model from Hugging Face.
    task="text-generation",                       # specify the task we want to perform with the model, which is "text-generation" in this case. This tells the HuggingFacePipeline that we want to use the model for generating text based on a given prompt.
    pipeline_kwargs=dict(
        max_length=1024,                          # set the maximum length of the generated text to 1024 tokens. This means that the model will generate responses that can be up to 1024 tokens long, which allows for more detailed and comprehensive answers.
        temperature=0.7,                         # set the temperature parameter to 0.7, which controls the randomness of the model's responses. A value of 0.7 allows for a good balance between creativity and coherence in the generated text.       
    ) 
)

model = ChatHuggingFace(llm=llm)             # create an instance of the ChatHuggingFace class with the specified model, in this case, "gpt-4". This will allow us to interact with the GPT-4 model using the langchain library.    

result = model.invoke("HOW TO FIND THAT THE ANIMAL IS THE CAT")         # invoke the model with a prompt, in this case, "What is the capital of France?", and get the response. This will send a request to the HuggingFace API and return the answer to the question about the capital of France.

print(result.content)                           # print the content of the result to the console. The result is an object that contains various information about the response from the model, and we are specifically interested in the 'content' attribute, which holds the actual answer to our question.

