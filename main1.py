import os
import json
import numpy as np

from numpy.linalg import norm
import ollama

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

## the above import are up to code. 
CHROMA_PATH = "chroma"
DATA_PATH = "data"
## this was implemented too

# Step load the documents from the data directory
def load_pdfs_from_directory(directory_path):
    document_loader= PyPDFDirectoryLoader(directory_path)
    return document_loader.load()

def split_documents(documents: list[Document]):
    # Step 2: Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)
    
##def add_to_chroma(chunks: list[Document]):
  ##  if os.path.exists(CHROMA_PATH):
   ##     #load the existing Chroma database
    ##    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
           
            if line:
                buffer.append(line.strip())
            elif len(buffer):
                paragraphs.append(" ".join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append(" ".join(buffer))
        return paragraphs
    
def get_embeddings(filename, modelname, chunks):
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    
    embeddings = [
        ollama.embeddings(model= modelname, prompt = chunk) ["embedding"] for chunk in chunks
    ]
    
    save_embeddings(filename, embeddings)
    return embeddings

def save_embeddings(filename, embeddings):
    # create a directory if it doesn't exist
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    # dump the embeddings to a json file
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)
    
def load_embeddings(filename):
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f) 
    ## chucnk embeddings and save them to a file
    

def find_most_similar(needle, haystack):
    # find the most similar embedding in the haystack to the needle
    # return the index of the most similar embedding
    needle_norm = norm(needle)
    similarities_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarities_scores, range(len(haystack))), reverse=True) 

## need to get the embeddings for the prompt and then find the most similar embedding in the haystack
    
def main():
    
    documents = load_pdfs_from_directory(DATA_PATH)
    chunks = split_documents(documents)
    print("Documents:", len(documents))
    print("Chunks:", len(chunks))
    print(chunks[0].page_content)
    
    SYSTEM_PROMPT = "You are a helpful assistant that answers questions based on the provided context. Answer the questions as truthfully as possible using the provided context, and if the answer is not contained within the text below, say 'I don't know.' "
    filename = "pdf_chunks"
    paragraphs = [chunk.page_content for chunk in chunks]
    embeddings = get_embeddings(filename,'mxbai-embed-large:latest', paragraphs)
    ## join multiple paragraphs into a single string and then chunk it into smaller pieces
    ## use the SemanticSplitterNodeParser to chunk the text into smaller pieces
    
    
    prompt = input("Enter your prompt: ")
    prompt_embedding = ollama.embeddings(model='mxbai-embed-large:latest', prompt=prompt)["embedding"]
    

    most_similar = find_most_similar(prompt_embedding, embeddings)[:5]  
    
    response = ollama.chat(
        model="phi3:latest",
        messages=[
            {
             "role": "system",
             "content": SYSTEM_PROMPT + "\n".join(paragraphs[item[1]] for item in most_similar),
             },
            
            {"role": "user", "content": prompt}],)  
    
    
    print("\n\n")
    print(response["message"]["content"])
    
            

if __name__ == "__main__":
    main()
    
    ## 
    ## chunk the text into smaller pieces
    ## parse text into paragraphs
    ## chunking are parses
    ## Pass it a list of chunks and get embeddings for each chunk
