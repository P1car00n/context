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

    @abc.abstractmethod
    @typing.override
    def __init__(
        self,
        embedding: embeddings.Embeddings,
    ) -> None:
        """Instantiate this vectorstore.

        Args:
            docs: Textual data from which to create embeddings for vector
                storage.
            embedding: Model to use to generate embeddings.
            kwargs: key-word arguments to pass to the underlying storage
                provider.
        """
        self._embedding = embedding
        self.vectorstore = None

    @abc.abstractmethod
    def store(
        self, docs: list[documents.Document], **kwargs: typing.Any
    ) -> vectorstores.VectorStore:
        """Store the docs in a vectorstore.

        Args:
            docs: Data to be sotred.
            kwargs: Key-word arguments to pass to the wrapped vectorstore.

        Returns:
            This vectorstore's instance.
        """


class ChromaStorage(BaseStorage):
    """Vector storage provided by the Chroma DB.

    Attributes:
        vectorstore: The wrapped storage object.
            All of its methods are available to the end user to use.
    """

    @typing.override
    def __init__(
        self,
        embedding: embeddings.Embeddings,
    ) -> None:
        super().__init__(embedding)

    @typing.override
    def store(
        self,
        docs: list[documents.Document],
        **kwargs: typing.Any,
    ) -> langchain_chroma.Chroma:
        self.vectorstore = langchain_chroma.Chroma.from_documents(
            docs,
            self._embedding,
            **kwargs,
        )
        return self.vectorstore


class LanceStorage(BaseStorage):
    """Vector storage provided by the Lance DB.

    Attributes:
        vectorstore: The wrapped storage object.
            All of its methods are available to the end user to use.
    """

    @typing.override
    def __init__(
        self,
        embedding: embeddings.Embeddings,
    ) -> None:
        super().__init__(embedding)

    @typing.override
    def store(
        self,
        docs: list[documents.Document],
        **kwargs: typing.Any,
    ) -> vectorstores.LanceDB:
        self.vectorstore = vectorstores.LanceDB.from_documents(
            docs,
            self._embedding,
            **kwargs,
        )
        return self.vectorstore


class FAISSStorage(BaseStorage):
    """Vector storage provided by the FAISS DB.

    Attributes:
        vectorstore: The wrapped storage object.
            All of its methods are available to the end user to use.
    """

    @typing.override
    def __init__(
        self,
        embedding: embeddings.Embeddings,
    ) -> None:
        super().__init__(embedding)

    @typing.override
    def store(
        self,
        docs: list[documents.Document],
        **kwargs: typing.Any,
    ) -> vectorstores.FAISS:
        self.vectorstore = vectorstores.FAISS.from_documents(
            docs,
            self._embedding,
            **kwargs,
        )
        return self.vectorstore
