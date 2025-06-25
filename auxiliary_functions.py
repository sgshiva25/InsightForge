from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
import pinecone
import openai
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from apikey import openai_api_key, pinecone_api_key
import os
from openai import OpenAI
from pinecone import Pinecone

os.environ['OPENAI_API_KEY'] = openai_api_key
load_dotenv(find_dotenv())
model = OpenAIEmbeddings(model="text-embedding-3-small")
client = OpenAI()

os.environ['PINECONE_API_KEY'] = pinecone_api_key
load_dotenv(find_dotenv())


pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('langchainproject1')

def find_match(input):
    input_em = model.embed_query(input)
    result = index.query(vector=input_em, top_k=2, include_metadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):

    response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # or "gpt-4"
    messages=[
        {"role": "system", "content": "Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"},
        {"role": "user", "content": query}
    ],
    temperature=0.7,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0)
    return response.choices[0].message.content

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

