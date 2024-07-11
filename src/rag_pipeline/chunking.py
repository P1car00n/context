"""A common interface for all chunkers and different chunking strategies.

All adhere to the strategy pattern as currently implemented.
"""

import abc
import pathlib
import typing

import langchain_text_splitters
from langchain_core import documents


class BaseChunker(abc.ABC):
    """Abstract base class defining common chunking operations."""

    @abc.abstractmethod
    def chunk(self) -> list[documents.Document]:
        """Perform chunking on the data used to initialize the class.

        Returns:
            A list of Document objects representing the split data.
        """


class CharacterChunker(BaseChunker):
    """Splitting text based on a character sequence."""

    @typing.override
    def __init__(self, doc_file: pathlib.Path, **kwargs: typing.Any) -> None:
        super().__init__()
        self.doc_file = doc_file
        self.text_splitter = langchain_text_splitters.CharacterTextSplitter(
            **kwargs,
        )

    @typing.override
    def chunk(self) -> list[documents.Document]:
        with self.doc_file.open() as file:
            text = file.read()
        return self.text_splitter.create_documents([text])


class RecursiveChunker(BaseChunker):
    """Recursively splitting text based on a character sequence."""

    @typing.override
    def __init__(self, doc_file: pathlib.Path, **kwargs: typing.Any) -> None:
        super().__init__()
        self.doc_file = doc_file
        self.text_splitter = (
            langchain_text_splitters.RecursiveCharacterTextSplitter(**kwargs)
        )

    @typing.override
    def chunk(self) -> list[documents.Document]:
        with self.doc_file.open() as file:
            text = file.read()
        return self.text_splitter.create_documents([text])
