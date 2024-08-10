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
        "/Users/af/Development/thesis/context/src/tests/resources/documents/",
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
        quality_metrics.SelfCheckEval(query, output),
        quality_metrics.LLMJudgeEval(query, output, examiner_model=model),
        quality_metrics.ListwiseRerankingEval(
            query,
            context=context,
            output=output,
            model=model,
        ),
    ]


def main() -> None:
    """Launch the pipeline."""
    queries = [
        "What role does private property play in promoting economic efficiency and resource allocation?",
        "Explain the merits of market institutions in contrast to the dangers of government intervention.",
        "How does Mises clarify the quantity theory of money? What implications does this theory have for inflation and monetary policy?",
        "According to Mises, why is socialism inherently flawed in terms of economic calculation?",
        "Explore Mises's views on interest rates and their impact on investment decisions.",
        "Discuss the concept of comparative advantage and its relevance to free trade.",
        "Analyze Mises's perspective on inflation.",
        "Compare and contrast fascism with other economic systems.",
        "What dangers does Mises highlight regarding industrial policy and central planning? How do these policies affect economic progress?",
        "Summarize Mises's core message about the relationship between liberty, private property, and prosperity.",
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
