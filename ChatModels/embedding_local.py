from langchain_huggingface import HuggingFaceEmbeddings  ## this is the new import statement for HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  # Initialize HuggingFaceEmbeddings with default settings    

text = "Hello, my cat  is cute"

document =[
    "Hello, my dog is cute",
    "The cat is on the roof",
    "The dog is in the garden"
]

embedding_vector = embeddings.embed_query(text)  # Generate embedding for a single query

embedding_vector2 = embeddings.embed_documents(document)  # Generate embeddings for a list of documents

print(str(embedding_vector2))      # Print the resulting embedding vector


## Before running this code, make sure to install the required libraries:
# pip install langchain_huggingface 
# pip install sentence-transformers
