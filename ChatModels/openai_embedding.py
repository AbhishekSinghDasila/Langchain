from langchain_openai import OpenAIEmbeddings   ## this is the new import statement for OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=32)  # Initialize OpenAIEmbeddings with default settings

result = openai_embeddings.embed_query("Hello, my dog is cute")  # Generate embedding for a single query

print(result)  # Print the resulting embedding vector

