# Local RAG LLM

A local Retrieval-Augmented Generation (RAG) application built with Python, Ollama, and vector embeddings.

## Overview

This project demonstrates how a RAG pipeline can retrieve relevant information from documents and provide that context to a locally running large language model.

The current implementation manually builds the core RAG workflow to better understand how retrieval works before integrating higher-level frameworks such as LangChain.

## Architecture

Document
↓
Text Parsing / Chunking
↓
Ollama Embeddings
↓
Vector Similarity Search
↓
Top-K Relevant Chunks
↓
Local LLM (Phi-3)
↓
Grounded Response

## Technologies

- Python
- Ollama
- Phi-3
- mxbai-embed-large
- NumPy
- Vector Embeddings
- Cosine Similarity
- Semantic Search

## Current Features

- Parse text documents into chunks
- Generate and cache document embeddings
- Generate embeddings for user queries
- Calculate cosine similarity between vectors
- Retrieve the most relevant document chunks
- Provide retrieved context to a local LLM
- Generate answers grounded in retrieved documents

## In Progress

- LangChain integration
- Improved document chunking
- Vector store integration
- Retrieval pipeline improvements

## Goal

The goal of this project is to understand the underlying mechanics of Retrieval-Augmented Generation before implementing the same architecture using LangChain and other RAG frameworks.
