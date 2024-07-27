"""Unit tests for loading.py."""

import typing
import unittest

from langchain_core import documents

from src.rag_pipeline import loading

PATH_TO_HTML = "src/tests/resources/documents/economic_policy.html"
PATH_TO_MARKDOWN = "src/tests/resources/documents/markdown_example.md"
PATH_TO_PDF = "src/tests/resources/documents/Economic Policy Thoughts for Today and Tomorrow.pdf"
PATH_TO_JSON = "src/tests/resources/documents/json_example.json"
PATH_TO_CSV = "src/tests/resources/documents/addresses.csv"
PATH_TO_DIR = "src/tests/resources/documents"


class TestMarkdownLoader(unittest.TestCase):
    """Tests for MarkdownLoader."""

    @classmethod
    @typing.override
    def setUpClass(cls) -> None:
        cls.loader = loading.MarkdownLoader(PATH_TO_MARKDOWN)

    def test_load(self) -> None:
        """A list of Document objects is successfully returned."""
        self.assertIsInstance(self.loader.load()[0], documents.Document)


class TestHTMLLoader(unittest.TestCase):
    """Tests for HTMLLoader."""

    @classmethod
    @typing.override
    def setUpClass(cls) -> None:
        cls.loader = loading.HTMLLoader(PATH_TO_HTML)

    def test_load(self) -> None:
        """A list of Document objects is successfully returned."""
        self.assertIsInstance(self.loader.load()[0], documents.Document)


class TestPDFLoader(unittest.TestCase):
    """Tests for PDFLoader."""

    @classmethod
    @typing.override
    def setUpClass(cls) -> None:
        cls.loader = loading.PDFLoader(PATH_TO_PDF)

    def test_load(self) -> None:
        """A list of Document objects is successfully returned."""
        self.assertIsInstance(self.loader.load()[0], documents.Document)


class TestJSONLoader(unittest.TestCase):
    """Tests for JSONLoader."""

    @classmethod
    @typing.override
    def setUpClass(cls) -> None:
        cls.loader = loading.JSONLoader(
            PATH_TO_JSON,
            jq_schema=".messages[].content",
            text_content=False,
        )

    def test_load(self) -> None:
        """A list of Document objects is successfully returned."""
        self.assertIsInstance(self.loader.load()[0], documents.Document)


class TestCSVLoader(unittest.TestCase):
    """Tests for CSVLoader."""

    @classmethod
    @typing.override
    def setUpClass(cls) -> None:
        cls.loader = loading.CSVLoader(PATH_TO_CSV)

    def test_load(self) -> None:
        """A list of Document objects is successfully returned."""
        self.assertIsInstance(self.loader.load()[0], documents.Document)


class TestFileSystemLoader(unittest.TestCase):
    """Tests for FileSystemLoader."""

    @classmethod
    @typing.override
    def setUpClass(cls) -> None:
        cls.loader = loading.FileSystemLoader(
            PATH_TO_DIR,
            silent_errors=True,
        )

    def test_load(self) -> None:
        """A list of Document objects is successfully returned."""
        self.assertIsInstance(self.loader.load()[0], documents.Document)
