"""The central module that orchestrates the entire RAG process."""

import typing

from langchain_community import vectorstores
from langchain_core import documents
from langchain_core import vectorstores as core_vectorstores

from . import (
    chunking,
    generating,
    loading,
    persisting,
    quality_metrics,
    retrieving,
)


class UnsetComponentError(Exception):
    """Component which must have been initialized by now is set to `None`."""

    @typing.override
    def __init__(self, component: str) -> None:
        """Initialize UnsetComponentError.

        Args:
            component: Name of the unset component.
        """
        super().__init__(f"{component} is set to `None`.")


class RAGPipeline:
    def __init__(
        self,
        loader: loading.BaseLoading | None = None,
        chunker: chunking.BaseChunker | None = None,
        persister: persisting.BaseStorage | None = None,
        retriever: retrieving.BaseRetriever | None = None,
        generator: generating.BaseGenerator | None = None,
        evaluators: typing.Iterable[quality_metrics.BaseEvaluation]
        | None = None,
    ) -> None:
        """Initialize the pipeline.

        The components set to `None` must be set to actual instances
        before the functionality they are responsible for is used.

        Args:
            loader: Defaults to None.
            chunker: Defaults to None.
            persister: Defaults to None.
            retriever: Defaults to None.
            generator: Defaults to None.
            evaluators: Defaults to an empty list.
        """
        self.loader = loader
        self.chunker = chunker
        self.persister = persister
        self.retriever = retriever
        self.generator = generator
        self.eveluators = [] if evaluators is None else evaluators

    def load_documents(self) -> list[documents.Document]:
        """Load the data."""
        if self.loader is None:
            raise UnsetComponentError("Loader")
        return self.loader.load()

    def chunk_documents(
        self,
        data: list[documents.Document],
    ) -> list[documents.Document]:
        """Chunk the data."""
        if self.chunker is None:
            raise UnsetComponentError("Chunker")
        return self.chunker.text_splitter.split_documents(data)

    def persist_documents(
        self,
        chunked_data: list[documents.Document],
    ) -> vectorstores.VectorStore:
        """Persist the data."""
        if self.persister is None:
            raise UnsetComponentError("Persister")
        return self.persister.store(chunked_data)

    def get_retriever(self) -> core_vectorstores.VectorStoreRetriever:
        """Retrieve the data."""
        if self.retriever is None:
            raise UnsetComponentError("Retriever")
        return self.retriever.get_retriever()

    def generate_answer(self, query: str) -> str:
        """Retrieve the data."""
        if self.generator is None:
            raise UnsetComponentError("Generator")
        return self.generator.generate(query)

    def evaluate(self) -> int:
        """Assess the pipeline's quality.

        Returns:
            An integer from 0 to 100 representing the quality score.
        """
        _evaluations = [evaluator.evaluate() for evaluator in self.eveluators]
        if not _evaluations:
            return 0
        return int(sum(_evaluations) / len(_evaluations) * 100)
