import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

# Similarity search

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'

# Erase old database file in docs/chroma/ if any
import shutil
try:
    shutil.rmtree(persist_directory)
except FileNotFoundError:
    pass

embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

print(vectordb._collection.count())

texts = [
    """Le capital est octroyé une fois par année pour une hospitalisation de plus de 24 heures..""",
    """Un capital bonus est alloué une fois par année.""",
    """Souscription au capital possible jusqu’à 50 ans.""",
]
print(f'Texts: {texts}')
smalldb = Chroma.from_texts(texts, embedding=embedding)

question = "Quand est-ce que le capital est octroyé ?"

print(f'Question: "{question}"')

# print similarity search results
print("Similarity search results:")
print(smalldb.similarity_search(question, k=2))
print()
print("Max marginal relevance search results:")
print(smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3))
