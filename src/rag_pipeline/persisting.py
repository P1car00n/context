"""A common interface for all persisting strategies.

All adhere to the strategy pattern as currently implemented.
"""

import abc
import typing

import langchain_chroma
from langchain_community import vectorstores
from langchain_core import documents, embeddings


class BaseStorage(abc.ABC):
    """Abstract base class defining common storage operations."""


class ChromaStorage(BaseStorage):
    """Vector storage provided by the Chroma DB.

    Attributes:
        vectorstore: The wrapped storage object.
            All of its methods are available to the end user to use.
    """

    @typing.override
    def __init__(
        self,
        docs: list[documents.Document],
        embedding: embeddings.Embeddings,
        **kwargs: typing.Any,
    ) -> None:
        """Instantiate ChromaStorage.

        Args:
            docs: Textual data from which to create embeddings for vector
                storage.
            embedding: Model to use to generate embeddings.
            kwargs: key-word arguments to pass to the underlying storage
                provider.
        """
        self._embedding = embedding
        self.vectorstore = langchain_chroma.Chroma.from_documents(
            docs,
            self._embedding,
            **kwargs,
        )


class LanceStorage(BaseStorage):
    """Vector storage provided by the Lance DB.

    Attributes:
        vectorstore: The wrapped storage object.
            All of its methods are available to the end user to use.
    """

    @typing.override
    def __init__(
        self,
        docs: list[documents.Document],
        embedding: embeddings.Embeddings,
        **kwargs: typing.Any,
    ) -> None:
        """Instantiate LanceStorage.

        Args:
            docs: Textual data from which to create embeddings for vector
                storage.
            embedding: Model to use to generate embeddings.
            kwargs: key-word arguments to pass to the underlying storage
                provider.
        """
        self._embedding = embedding
        self.vectorstore = vectorstores.LanceDB.from_documents(
            docs,
            self._embedding,
            **kwargs,
        )


class FAISSStorage(BaseStorage):
    """Vector storage provided by the FAISS DB.

    Attributes:
        vectorstore: The wrapped storage object.
            All of its methods are available to the end user to use.
    """

    @typing.override
    def __init__(
        self,
        docs: list[documents.Document],
        embedding: embeddings.Embeddings,
        **kwargs: typing.Any,
    ) -> None:
        """Instantiate FAISSStorage.

        Args:
            docs: Textual data from which to create embeddings for vector
                storage.
            embedding: Model to use to generate embeddings.
            kwargs: key-word arguments to pass to the underlying storage
                provider.
        """
        self._embedding = embedding
        self.vectorstore = vectorstores.FAISS.from_documents(
            docs,
            self._embedding,
            **kwargs,
        )
