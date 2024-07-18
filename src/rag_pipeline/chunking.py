"""A common interface for all chunkers and different chunking strategies.

All adhere to the strategy pattern as currently implemented.
"""

import abc
import enum
import pathlib
import typing

import langchain_huggingface
import langchain_text_splitters
from langchain_core import documents, embeddings
from langchain_experimental.text_splitter import (
    SemanticChunker as ExperimentalSemanticTextSplitter,
)


class BaseChunker(abc.ABC):
    """Abstract base class defining common chunking operations."""

    @abc.abstractmethod
    def __init__(self) -> None:
        """Instantiate the class.

        Args:
            doc_path: Path to the file to chunk.
            kwargs: Any arguments to pass to the wrapped object.
        """

    @abc.abstractmethod
    def chunk(self) -> list[documents.Document]:
        """Perform chunking on the data used to initialize the class.

        Returns:
            A list of Document objects representing the split data.
        """


class CharacterChunker(BaseChunker):
    """Split text based on a character sequence.

    It is a thin wrapper over `CharacterTextSplitter`.
    """

    @typing.override
    def __init__(self, doc_path: pathlib.Path, **kwargs: typing.Any) -> None:
        self.doc_path = doc_path
        self.text_splitter = langchain_text_splitters.CharacterTextSplitter(
            **kwargs,
        )

    @typing.override
    def chunk(self) -> list[documents.Document]:
        with self.doc_path.open() as file:
            text = file.read()
        return self.text_splitter.create_documents([text])


class RecursiveChunker(BaseChunker):
    """Recursively split text based on a character sequence.

    It is a thin wrapper over `RecursiveCharacterTextSplitter`.
    """

    @typing.override
    def __init__(self, doc_path: pathlib.Path, **kwargs: typing.Any) -> None:
        self.doc_path = doc_path
        self.text_splitter = (
            langchain_text_splitters.RecursiveCharacterTextSplitter(**kwargs)
        )

    @typing.override
    def chunk(self) -> list[documents.Document]:
        with self.doc_path.open() as file:
            text = file.read()
        return self.text_splitter.create_documents([text])


class HTMLChunkingMethod(enum.Enum):
    """Decide on the type of chunking to perform."""

    HEADER = enum.auto()
    SECTION = enum.auto()


class HTMLChunker(BaseChunker):
    """Split an HTML document.

    It is a thin wrapper over `HTMLHeaderTextSplitter`
    and `HTMLSectionSplitter`.
    """

    @typing.override
    def __init__(
        self,
        method: HTMLChunkingMethod,
        doc_path: pathlib.Path,
        **kwargs: typing.Any,
    ) -> None:
        """See base class.

        Args:
            method: Decide how to chunk.
        """
        self.doc_path = doc_path
        match method:
            case method.HEADER:
                self.text_splitter = (
                    langchain_text_splitters.HTMLHeaderTextSplitter(
                        **kwargs,
                    )
                )
            case method.SECTION:
                self.text_splitter = (
                    langchain_text_splitters.HTMLSectionSplitter(
                        **kwargs,
                    )
                )

    @typing.override
    def chunk(self) -> list[documents.Document]:
        return self.text_splitter.split_text_from_file(self.doc_path)


class MarkdownHeaderChunker(BaseChunker):
    """Split a Markdown document.

    It is a thin wrapper over `MarkdownHeaderTextSplitter`.
    """

    @typing.override
    def __init__(self, doc_path: pathlib.Path, **kwargs: typing.Any) -> None:
        super().__init__()
        self.doc_path = doc_path
        self.text_splitter = (
            langchain_text_splitters.MarkdownHeaderTextSplitter(
                **kwargs,
            )
        )

    @typing.override
    def chunk(self) -> list[documents.Document]:
        with self.doc_path.open() as file:
            text = file.read()
        return self.text_splitter.split_text(text)


class SemanticChunker(BaseChunker):
    """Semantically split text.

    It is a thin wrapper over langchain's `SemanticChunker`.
    """

    @typing.override
    def __init__(
        self,
        doc_path: pathlib.Path,
        embedding_model: embeddings.Embeddings | str,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__()
        if isinstance(embedding_model, str):
            embedding_model = langchain_huggingface.HuggingFaceEmbeddings(
                model_name=embedding_model,
            )
        self.doc_path = doc_path
        self.text_splitter = ExperimentalSemanticTextSplitter(
            embedding_model,
            **kwargs,
        )

    @typing.override
    def chunk(self) -> list[documents.Document]:
        with self.doc_path.open() as file:
            text = file.read()
        return self.text_splitter.create_documents([text])


class TokenChunker(BaseChunker):
    """Split text with a hard limit on the token size.

    It is a thin wrapper over `TokenTextSplitter`.
    """

    @typing.override
    def __init__(self, doc_path: pathlib.Path, **kwargs: typing.Any) -> None:
        super().__init__()
        self.doc_path = doc_path
        self.text_splitter = langchain_text_splitters.TokenTextSplitter(
            **kwargs,
        )

    @typing.override
    def chunk(self) -> list[documents.Document]:
        with self.doc_path.open() as file:
            text = file.read()
        return self.text_splitter.create_documents([text])
