"""Pipeline's initializer."""

import logging
import pathlib

import dotenv
import langchain_huggingface
from langchain_openai import embeddings

from . import (
    chunking,
    generating,
    loading,
    persisting,
    pipeline,
    quality_metrics,
    retrieving,
)

dotenv.load_dotenv()

PDF_PATH = "/Users/af/Development/thesis/context/src/tests/resources/documents/Economic Policy Thoughts for Today and Tomorrow.pdf"
TXT_PATH = "/Users/af/Development/thesis/context/src/tests/resources/documents/Economic Policy Thoughts for Today and Tomorrow.txt"

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="logs/pipeline.log",
    filemode="w+",
    encoding="utf-8",
    level=logging.INFO,
)


def get_text_loader() -> loading.FileSystemLoader:
    """Helper function returning an instance of FileSystemLoader."""
    return loading.FileSystemLoader(
        "/Users/af/Development/thesis/context/src/tests/resources/data/",
        glob="*.txt",
        use_multithreading=True,
    )


def get_recursive_chunker() -> chunking.RecursiveChunker:
    """Helper function returning an instance of RecursiveChunker."""
    return chunking.RecursiveChunker(
        pathlib.Path(TXT_PATH),
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )


def get_semantic_chunker() -> chunking.SemanticChunker:
    """Helper function returning an instance of SemanticChunker."""
    return chunking.SemanticChunker(
        pathlib.Path(TXT_PATH),
        embedding_model=embeddings.OpenAIEmbeddings(),
    )


def get_quality_metrics(
    query: str,
    output: str,
    context: list[str],
) -> list[quality_metrics.BaseEvaluation]:
    """Return a list of instantiated quality metrics."""
    model = "gpt-4o-mini"
    return [
        quality_metrics.RAGAsEval(query, output, model=model),
        quality_metrics.LLMGraderEval(query, output, model=model),
        # quality_metrics.SelfCheckEval(query, output),
        # quality_metrics.LLMJudgeEval(query, output, examiner_model=model),
        # quality_metrics.ListwiseRerankingEval(
        #   query,
        #    context=context,
        #    output=output,
        #    model=model,
        # ),
    ]


def main() -> None:
    """Launch the pipeline."""
    queries = [
        "What is Chowder?",
        "Define PDI",
        "What should Site Reliability Engineer do?",
        "Can I work from a coffe shop?",
        "How many years of experience do I need to qualify for the title of Principal Programmer?",
        "What does Junior Designer need to be able to do with regard to web technologies?",
        "What is EOS?",
        "Tell me about Queenbee.",
        "Am I allowed to have a side gig?",
        "Can I bring my own devices to work?",
    ]

    loader = get_text_loader()
    chunker = get_recursive_chunker()

    _pipeline = pipeline.RAGPipeline(loader, chunker)

    loaded_documents = _pipeline.load_documents()
    chunked_documents = _pipeline.chunk_documents(loaded_documents)
    persister = persisting.ChromaStorage(
        langchain_huggingface.HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        ),
    )

    _pipeline.persister = persister
    vector_store = _pipeline.persist_documents(chunked_documents)

    retriever = retrieving.StandardRetriever(vector_store)
    _pipeline.retriever = retriever

    lc_retriever = _pipeline.get_retriever()

    generator = generating.LLAMAFileGenerator(lc_retriever)
    _pipeline.generator = generator

    for query in queries:
        context = lc_retriever.invoke(query)
        logger.info("Context received: %s", context)

        answer = _pipeline.generate_answer(query)

        _pipeline.eveluators = get_quality_metrics(
            query,
            answer,
            context=[context[i].page_content for i in range(len(context))],
        )
        quality = _pipeline.evaluate()

        print(answer)
        print(quality)
