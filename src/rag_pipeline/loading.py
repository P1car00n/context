"""A common interface for all loaders.

All adhere to the strategy pattern as currently implemented.
"""

import abc
import typing

from langchain_community import document_loaders
from langchain_core import documents


class BaseLoading(abc.ABC):
    """Abstract base class defining common loading operations."""

    @abc.abstractmethod
    def __init__(self, **kwargs: typing.Any) -> None:
        """Instantiate the class.

        Args:
            file_path: Path to the file to load.
            kwargs: Any arguments to pass to the wrapped object.
        """

    @abc.abstractmethod
    def load(self) -> list[documents.Document]:
        """Load the data.

        Returns:
            A list of Document objects representing the loaded data.
        """


class FileSystemLoader(BaseLoading):
    """Load documents from the file system.

    It is a thin wrapper over `DirectoryLoader`.
    """

    @typing.override
    def __init__(self, dir_path: str, **kwargs: typing.Any) -> None:
        """Instantiate the class.

        Args:
            dir_path: Path to a directory containing the files to load.
            kwargs: Any arguments to pass to the wrapped object.
        """
        self._loader = document_loaders.DirectoryLoader(dir_path, **kwargs)

    @typing.override
    def load(self) -> list[documents.Document]:
        return self._loader.load()


class CSVLoader(BaseLoading):
    """Load documents from a CSV file.

    It is a thin wrapper over LangChain's `CSVLoader`.
    """

    @typing.override
    def __init__(self, file_path: str, **kwargs: typing.Any) -> None:
        self._loader = document_loaders.CSVLoader(file_path, **kwargs)

    @typing.override
    def load(self) -> list[documents.Document]:
        return self._loader.load()


class JSONLoader(BaseLoading):
    """Load documents from a JSON file.

    It is a thin wrapper over LangChain's `JSONLoader`.
    """

    @typing.override
    def __init__(
        self, file_path: str, jq_schema: str, **kwargs: typing.Any
    ) -> None:
        """Instantiate the class.

        Args:
            file_path: Path to the file to load.
            jq_schema: The jq schema from which to extract text.
            kwargs: Any arguments to pass to the wrapped object.
        """
        self._loader = document_loaders.JSONLoader(
            file_path,
            jq_schema,
            **kwargs,
        )

    @typing.override
    def load(self) -> list[documents.Document]:
        return self._loader.load()


class MarkdownLoader(BaseLoading):
    """Load documents from a Markdown file.

    It is a thin wrapper over `UnstructuredMarkdownLoader`.
    """

    @typing.override
    def __init__(self, file_path: str, **kwargs: typing.Any) -> None:
        self._loader = document_loaders.UnstructuredMarkdownLoader(
            file_path,
            **kwargs,
        )

    @typing.override
    def load(self) -> list[documents.Document]:
        return self._loader.load()


class HTMLLoader(BaseLoading):
    """Load documents from an HTML file.

    It is a thin wrapper over `UnstructuredHTMLLoader`.
    """

    @typing.override
    def __init__(self, file_path: str, **kwargs: typing.Any) -> None:
        self._loader = document_loaders.UnstructuredHTMLLoader(
            file_path,
            **kwargs,
        )

    @typing.override
    def load(self) -> list[documents.Document]:
        return self._loader.load()


class PDFLoader(BaseLoading):
    """Load documents from a PDF file.

    It is a thin wrapper over `UnstructuredPDFLoader`.
    """

    @typing.override
    def __init__(self, file_path: str, **kwargs: typing.Any) -> None:
        self._loader = document_loaders.UnstructuredPDFLoader(
            file_path,
            **kwargs,
        )

    @typing.override
    def load(self) -> list[documents.Document]:
        return self._loader.load()
