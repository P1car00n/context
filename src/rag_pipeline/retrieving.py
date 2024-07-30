"""A common interface for all retrieving strategies.

All adhere to the strategy pattern as currently implemented.
"""

import abc
import typing

from langchain import retrievers
from langchain.retrievers import document_compressors, multi_query
from langchain_core import language_models, vectorstores
from langchain_core import retrievers as core_retrievers


class BaseRetriever(abc.ABC):
    """Abstract base class defining common retrieving operations."""

    @abc.abstractmethod
    @typing.override
    def __init__(self, storage: vectorstores.VectorStore) -> None:
        """Instantiate the class.

        Args:
            storage: The storage instance to turn into a retriever.
        """
        self._storage = storage

    @abc.abstractmethod
    def get_retriever(
        self,
        **kwargs: typing.Any,
    ) -> core_retrievers.BaseRetriever:
        """Return the storage upgraded to retriever.

        Returns:
            Vector stroe retriever.
        """


class StandardRetriever(BaseRetriever):
    """Retriever created from a persister."""

    @typing.override
    def __init__(self, storage: vectorstores.VectorStore) -> None:
        super().__init__(storage)

    @typing.override
    def get_retriever(
        self,
        **kwargs: typing.Any,
    ) -> vectorstores.VectorStoreRetriever:
        return self._storage.as_retriever(**kwargs)


class MultiQueryRetriever(BaseRetriever):
    """Retrieve using additional queries.

    The LLM will create more queries based on the original one
    in order to search more broadly.
    """

    @typing.override
    def __init__(
        self,
        storage: vectorstores.VectorStore,
        llm: language_models.BaseLanguageModel,
    ) -> None:
        super().__init__(storage)
        self._llm = llm

    @typing.override
    def get_retriever(
        self,
        **kwargs: typing.Any,
    ) -> multi_query.MultiQueryRetriever:
        return multi_query.MultiQueryRetriever.from_llm(
            retriever=self._storage.as_retriever(**kwargs),
            llm=self._llm,
        )


class ComressedContextRetriever(BaseRetriever):
    """Retrieve using filtering and compression.

    The LLM will filter the documents and make them more conscise.
    """

    @typing.override
    def __init__(
        self,
        storage: vectorstores.VectorStore,
        compressor_llm: language_models.BaseLanguageModel,
    ) -> None:
        super().__init__(storage)
        self._llm = compressor_llm

    @typing.override
    def get_retriever(
        self,
        **kwargs: typing.Any,
    ) -> retrievers.ContextualCompressionRetriever:
        _compressor = document_compressors.LLMChainExtractor.from_llm(
            self._llm,
        )
        return retrievers.ContextualCompressionRetriever(
            base_compressor=_compressor,
            base_retriever=self._storage.as_retriever(**kwargs),
        )
