"""A common interface for all persisting strategies.

All adhere to the strategy pattern as currently implemented.
"""

import abc
import typing

import langchain_chroma
from langchain_core import documents, embeddings, vectorstores


class BaseStorage(abc.ABC):
    """Abstract base class defining common storage operations."""

    @abc.abstractmethod
    def get_vectorstore(self) -> vectorstores.VectorStore:
        """Return a vector DB with provided embeddings.

        Returns:
            A VectorStore implementation dependent on the exact class used.
        """


class ChromaStorage(BaseStorage):
    """Vector storage provided by the Chroma DB."""

    @typing.override
    def __init__(
        self,
        docs: list[documents.Document],
        embedding: embeddings.Embeddings,
        **kwargs: typing.Any,
    ) -> None:
        """Instantiate ChromaStorage.

        Args:
            docs: Textual data from which ti create embeddings for vector
                storage.
            embedding: Model to use to generate embeddings.
            kwargs: key-word arguments to pass to the underlying storage
                provider.
        """
        self.embedding = embedding
        self.vectorstore = langchain_chroma.Chroma.from_documents(
            docs,
            self.embedding,
            **kwargs,
        )

    @typing.override
    def get_vectorstore(self) -> langchain_chroma.Chroma:
        return self.vectorstore
