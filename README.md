# Local RAG LLM 🤖

A local Retrieval-Augmented Generation (RAG) system built with Python, Ollama, LangChain, and vector embeddings.

Ask questions about your own documents without sending them to a hosted LLM API.

> Because sometimes your PDFs deserve their own private ChatGPT.

## Overview

This project explores how Retrieval-Augmented Generation works from the ground up.

Rather than immediately hiding the pipeline behind a framework, I first implemented the core RAG workflow manually: document parsing, embeddings, cosine similarity, retrieval, and local LLM inference.

The project is now being expanded with LangChain to compare the manual implementation with a framework-based approach.

Currently, the application can load PDF documents, split them into chunks, generate vector embeddings, retrieve relevant information, and provide that context to a locally running Phi-3 model.

## Architecture

```text
PDF Documents
      ↓
LangChain Document Loader
      ↓
RecursiveCharacterTextSplitter
      ↓
Document Chunks
      ↓
Ollama Embeddings
(mxbai-embed-large)
      ↓
Vector Similarity Search
      ↓
Top-K Relevant Chunks
      ↓
Local LLM
(Phi-3)
      ↓
Grounded Response
