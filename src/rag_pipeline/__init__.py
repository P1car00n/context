"""Pipeline's initializer."""

import logging
import pathlib

import dotenv
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

dotenv.load_dotenv()

_PDF_PATH = "/Users/af/Development/thesis/context/src/tests/resources/documents/Economic Policy Thoughts for Today and Tomorrow.pdf"

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="logs/pipeline.log",
    filemode="w+",
    encoding="utf-8",
    level=logging.INFO,
)


def _get_quality_metrics(
    query: str, output: str
) -> list[quality_metrics.BaseEvaluation]:
    _model = "gpt-4o-mini-2024-07-18"
    return [
        quality_metrics.RAGAsEval(query, output, model=_model),
        quality_metrics.LLMGraderEval(query, output, model=_model),
        # quality_metrics.SelfCheckEval(),
        quality_metrics.LLMJudgeEval(query, output, examiner_model=_model),
        # quality_metrics.ListwiseRerankingEval(query, output, model=_model),
    ]


def main():
    _queries = [
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

    _generator = generating.LLAMAFileGenerator(_lc_retriever)
    _pipeline.generator = _generator

    for query in _queries:
        _context = _lc_retriever.invoke(query)
        logger.info("Context received: %s", _context)

        _answer = _pipeline.generate_answer(query)

        _pipeline.eveluators = _get_quality_metrics(query, _answer)
        _quality = _pipeline.evaluate()

        print(_answer)
        print(_quality)
