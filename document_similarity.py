from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise  import cosine_similarity  ## this is the new import statement for HuggingFaceEmbeddings
import numpy as np

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  # Initialize HuggingFaceEmbeddings with default settings    


document = ["The cat is on the roof and the cat is sleeping",
             "The dog is in the garden and the dog is barking",
             "The lion is in the savannah and the lion is hunting",
             "The bird is in the sky and the bird is flying"]

query_embedding = embeddings.embed_documents(document)  # Generate embedding of the whole document list

query = "Tell me about the lion"

query_embedding2 = embeddings.embed_query(query)  # Generate embedding for a single query

score= cosine_similarity([query_embedding2], query_embedding)[0]  # Calculate cosine similarity between the query embedding and the document embeddings

index, score = sorted(list(enumerate(score)), key=lambda x: x[1])[-1]  # Print the sorted list of document indices and their corresponding similarity scores

print(f"Most similar document: {document[index]} with a similarity score of {score:.4f}")  # Print the most similar document and its similarity score

