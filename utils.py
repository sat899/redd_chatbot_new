#Load libraries
from dotenv import load_dotenv
import os
from openai import OpenAI
import tiktoken
import PyPDF2
import pandas as pd
import pickle
from typing import List
import numpy as np
import faiss
#from scipy import spatial
#from sklearn.metrics.pairwise import cosine_similarity
import base64

#create_candidate_item_prompt function
def create_candidate_item_prompt(items):
    candidates_string = ", ".join([item.replace('_', ' ') for item in items])
    candidate_prompt_string = f"The candidate items that you can recommend are (in no particular order): {candidates_string}."
    return candidate_prompt_string

#create_item_descriptions_prompt function
def create_item_descriptions_prompt(item_descriptions):
    formatted_descriptions = '\n'.join(f"{i + 1}. {desc}" for i, desc in enumerate(item_descriptions))
    item_descriptions_prompt_string = f"This is a description of each candidate item (in the same order as the candidate list):\n{formatted_descriptions}"
    return item_descriptions_prompt_string

#create_links_prompt function
# def create_links_prompt(links):
#     links_string = ", ".join([link.replace('_', ' ') for link in links])
#     links_prompt_string = f"The links for each candidate item are (in the same order as the candidate items): {links_string}."
#     return links_prompt_string

#create_links_prompt function
def create_links_prompt(links):
    # Ensure all entries are strings and handle non-string or missing values
    links_string = ", ".join([str(link).replace('_', ' ') if pd.notna(link) else "N/A" for link in links])
    links_prompt_string = f"The links for each candidate item are (in the same order as the candidate items): {links_string}. Note: you should only recommend these links and never anything else"
    return links_prompt_string

#Specify the embedding model to be used
EMBEDDING_MODEL = "text-embedding-3-small"

#Load the environment variables from the .env file
load_dotenv()
key = os.environ.get("OPENAI_API_KEY")

#Initialise client
client = OpenAI(api_key=key)

#Initialize tokenizer for the embedding model
encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)

#Set file paths for embedding caches
#embedding_cache_path_items = "items_embeddings_cache.pkl"
embedding_cache_path_docs = "documents_embeddings_cache.pkl"

#Load caches if they exist, else initialize empty caches
#Code from https://cookbook.openai.com/examples/recommendation_using_embeddings
# try:
#     embedding_cache_items = pd.read_pickle(embedding_cache_path_items)
# except FileNotFoundError:
#     embedding_cache_items = {}

try:
    embedding_cache_docs = pd.read_pickle(embedding_cache_path_docs)
except FileNotFoundError:
    embedding_cache_docs = {}

#Save the caches back to disk (optional, to ensure they exist from the start)
# with open(embedding_cache_path_items, "wb") as embedding_cache_file_items:
#     pickle.dump(embedding_cache_items, embedding_cache_file_items)

with open(embedding_cache_path_docs, "wb") as embedding_cache_file_docs:
    pickle.dump(embedding_cache_docs, embedding_cache_file_docs)

#get_embedding function
#Code from: https://platform.openai.com/docs/guides/embeddings/use-cases
def get_embedding(text, model=EMBEDDING_MODEL):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

#embedding_from_string function
#Code from https://cookbook.openai.com/examples/recommendation_using_embeddings
def embedding_from_string(
    string: str,
    model: str,
    embedding_cache: dict,
    embedding_cache_path: str
) -> list:
    """Return embedding of a given string, using a cache to avoid recomputing."""
    
    # Check if the embedding is already in the cache
    if (string, model) not in embedding_cache:
        # If not, compute the embedding and store it in the cache
        embedding_cache[(string, model)] = get_embedding(string, model)
        
        # Save the updated cache to disk
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    
    # Return the cached or newly computed embedding
    return embedding_cache[(string, model)]

#get_item_embeddings function
#Adapted from recommendations_from_strings function. Original at: https://platform.openai.com/docs/guides/embeddings/use-cases
# def get_item_embeddings(strings: List[str], model=EMBEDDING_MODEL):
#     """Get embeddings for all recommendation strings."""
#     item_embeddings = [
#         embedding_from_string(
#             string, model=model, embedding_cache=embedding_cache_items, embedding_cache_path=embedding_cache_path_items
#         ) for string in strings
#     ]
#     return item_embeddings

# #cache_item_embeddings function
# def cache_item_embeddings(strings, model=EMBEDDING_MODEL, cache_path="item_embeddings.pkl"):
#     if os.path.exists(cache_path):
#         with open(cache_path, "rb") as file:
#             return pickle.load(file)
#     embeddings = get_item_embeddings(strings, model)
#     with open(cache_path, "wb") as file:
#         pickle.dump(embeddings, file)
#     return embeddings

#load_documents function
def load_documents(directory):
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            with open(os.path.join(directory, filename), 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + '\n'
                documents[filename] = text
    texts = list(documents.values())  # Take the values from the dict and turn into a single list
    return texts

#split_text_into_chunks function
def split_text_into_chunks(text, max_tokens=8192):
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = encoding.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

#cache_document_chunks function
def cache_document_chunks(directory, cache_path="background_docs_chunks.pkl"):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as file:
            return pickle.load(file)
    documents = load_documents(directory)
    chunks = []
    for doc in documents:
        chunks.extend(split_text_into_chunks(doc))
    with open(cache_path, "wb") as file:
        pickle.dump(chunks, file)
    return chunks

#get_doc_embeddings function
def get_doc_embeddings(texts: List[str], model=EMBEDDING_MODEL):
    """Get embeddings for all document texts."""
    doc_embeddings = [
        embedding_from_string(
            text, model=model, embedding_cache=embedding_cache_docs, embedding_cache_path=embedding_cache_path_docs
        ) for text in texts
    ]
    return doc_embeddings

#cache_doc_embeddings function
def cache_doc_embeddings(chunks, model=EMBEDDING_MODEL, cache_path="doc_embeddings.pkl"):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as file:
            return pickle.load(file)
    embeddings = get_doc_embeddings(chunks, model=model)
    with open(cache_path, "wb") as file:
        pickle.dump(embeddings, file)
    return embeddings

#load_faiss_index function
def load_faiss_index(path="faiss_index.bin"):
    return faiss.read_index(path)

#save_faiss_index function
def save_faiss_index(index, path="faiss_index.bin"):
    faiss.write_index(index, path)

#create_faiss_index function
def create_faiss_index(embeddings, index_path="faiss_index.bin"):
    if os.path.exists(index_path):
        return load_faiss_index(index_path)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    save_faiss_index(index, index_path)
    return index

#extract_interactions function
# def extract_interactions(user_data, candidate_items):

#     interacted_items = [
#         column for column in user_data.columns 
#         if user_data[column].values[0] == 1 and column in candidate_items
#     ]
    
#     interactions_string = ", ".join(interacted_items).replace('_', ' ')
#     interaction_prompt_string = f"The user has interacted with the following items in the past (in no particular order): {interactions_string}."
#     return interacted_items, interaction_prompt_string

#create_item_faiss_index function
# def create_item_faiss_index(item_embeddings):
#     dimension = len(next(iter(item_embeddings.values())))
#     index = faiss.IndexFlatL2(dimension)
#     embeddings_array = np.array(list(item_embeddings.values()))
#     index.add(embeddings_array)
#     return index, list(item_embeddings.keys())

# #average_embeddings function
# def average_embeddings(interacted_items, item_embeddings):
#     user_embeddings = [item_embeddings[item] for item in interacted_items]
#     average_embedding = np.mean(user_embeddings, axis=0)
#     return average_embedding

# #retrieve_similar_items function
# def retrieve_similar_items(query_embedding, index, item_list, top_k=3):
#     _, indices = index.search(np.array([query_embedding]), top_k)
#     return [item_list[i] for i in indices[0]]

# #new_item_distances function
# #Adapted from distances_from_embeddings function. Original at: https://github.com/openai/openai-python/blob/release-v0.28.1/openai/embeddings_utils.py
# def new_item_distances(avg_embedding, item_embeddings, interacted_items, distance_metric="cosine"):
#     distance_metrics = {
#         "cosine": spatial.distance.cosine,
#         "L1": spatial.distance.cityblock,
#         "L2": spatial.distance.euclidean,
#         "Linf": spatial.distance.chebyshev,
#     }
    
#     distances = []
    
#     for item, embedding in item_embeddings.items():
#         if item not in interacted_items: #only calculating the distances to the items the user has not interacted with
#             distance = distance_metrics[distance_metric](avg_embedding, embedding)
#             distances.append((item, distance))
        
#         distances.sort(key=lambda x: x[1])  # Sort by distance
        
#     return distances

# #get_similar_items function
# def get_similar_items(interacted_items, item_embeddings):

#     #If statement to check if user has interacted with any items
#     if not interacted_items:
#         similiar_item_prompt_string = "The user has not interacted with any items yet, so recommendations should be based on their profile information and biography"
#         top_3 = "The user has not interacted with any items yet, so recommendations should be based on their profile information and biography"
#     else:
#         # Calculate the average embedding for the items the user has already interacted with
#         avg_embedding = average_embeddings(interacted_items, item_embeddings)

#         # Get a ranked list of the most similar items and their distances
#         ordered_items = new_item_distances(avg_embedding, item_embeddings, interacted_items)
#         top_3 = [item[0] for item in ordered_items[:3]]

#         #Create a prompt of similar items
#         similiar_item_prompt_string = "This is a list of the most similar items to the ones that the user has already interacted with. The items are listed in order of their similarity based on their respective distance values (between 0 and 1). A lower distance value means that the item is more similar to the items that the user has already interacted with. Items with low distance values are therefore good candidates for recommendation:\n"
#         for item, distance in ordered_items:
#             similiar_item_prompt_string += f"- {item}: {distance}\n"
    
#     return similiar_item_prompt_string, top_3

#read_file function
def read_file(filename):
    with open(filename, 'r') as file:
        data = file.read()
    return data

#get_base64_image function
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

#retrieve_documents function
def retrieve_documents(query, index, document_chunks, model=EMBEDDING_MODEL):
    query_embedding = np.array(get_embedding(query, model=model)).reshape(1, -1)
    _, indices = index.search(query_embedding, 1)
    return document_chunks[indices[0][0]]

#retrieve_documents function
#def retrieve_documents(query, doc_embeddings, document_chunks, model=EMBEDDING_MODEL):
    #query_embedding = get_embedding(query, model=model)
    #similarities = cosine_similarity([query_embedding], doc_embeddings)
    #most_similar_idx = np.argmax(similarities)
    #return document_chunks[most_similar_idx]