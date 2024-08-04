"""Pipeline's initializer."""

import pathlib

import langchain_huggingface

from . import (
    chunking,
    generating,
    loading,
    persisting,
    pipeline,
    quality_metrics,
    retrieving,
)

_PDF_PATH = "/Users/af/Development/thesis/context/src/tests/resources/documents/Economic Policy Thoughts for Today and Tomorrow.pdf"


def main():
    _query = "What is the relationship between economic freedom and prosperity according to Mises?"

    _loader = loading.PDFLoader(_PDF_PATH)
    _chunker = chunking.RecursiveChunker(
        pathlib.Path(_PDF_PATH),
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )

    _pipeline = pipeline.RAGPipeline(_loader, _chunker)

    _loaded_documents = _pipeline.load_documents()
    _chunked_documents = _pipeline.chunk_documents(_loaded_documents)
    _persister = persisting.ChromaStorage(
        langchain_huggingface.HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        ),
    )

    _pipeline.persister = _persister
    _vector_store = _pipeline.persist_documents(_chunked_documents)

    _retriever = retrieving.StandardRetriever(_vector_store)
    _pipeline.retriever = _retriever

    _lc_retriever = _pipeline.get_retriever()
    _context = _lc_retriever.invoke(_query)

    _generator = generating.LLAMAFileGenerator(_lc_retriever)
    _pipeline.generator = _generator

    _answer = _pipeline.generate_answer(_query)

    print(_answer)
