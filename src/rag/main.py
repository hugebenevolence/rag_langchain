import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from src.base.llm_model import get_hf_llm
from src.rag.rag_chain import build_rag_chain, InputQA, OutputQA

llm = get_hf_llm(temperature=0.9)
genai_docs = "./data_source/generative_ai"

genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using LangChain’s Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/check")
async def check():
    return {"status": "ok"}

@app.post("/generative_ai", response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    answer = genai_chain.invoke(inputs.question)
    return {"answer": answer}

add_routes(app, genai_chain, playground_type="default", path="/generative_ai")

# from pydantic import BaseModel, Field
# # import sys
# # sys.path.insert(0, "d:\DAI_NHAN\Projects\rag_langchain")
# from src.rag.file_loader import Loader
# from src.rag.vectorstore import VectorDB
# from src.rag.offline_rag import Offline_RAG


# class InputQA(BaseModel):
#     question: str = Field(..., title="Question to ask the model")


# class OutputQA(BaseModel):
#     answer: str = Field(..., title="Answer from the model")


# def build_rag_chain(llm, data_dir, data_type):
#     doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers=2)
#     retriever = VectorDB(documents=doc_loaded).get_retriever()
#     rag_chain = Offline_RAG(llm).get_chain(retriever)

#     return rag_chain


# if __name__ == "__main__":
#     from src.base.llm_model import get_hf_llm
    
#     # Initialize LLM
#     print("Loading LLM model...")
#     llm = get_hf_llm(temperature=0.9)
    
#     # Build RAG chain
#     print("Building RAG chain...")
#     genai_docs = "./data_source/generative_ai"
#     genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")
    
#     # Test query
#     print("Testing RAG chain...")
#     question = "What is generative AI?"
#     answer = genai_chain.invoke(question)
#     print(f"Question: {question}")
#     print(f"Answer: {answer}")
