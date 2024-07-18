"""Unit tests for chunking.py."""

import pathlib
import typing
import unittest

from langchain_core import documents

from src.rag_pipeline import chunking

PATH_TO_DOCUMENT = pathlib.Path(
    "src/tests/resources/documents/economic_policy.txt"
)
PATH_TO_HTML = pathlib.Path(
    "src/tests/resources/documents/economic_policy.html"
)
PATH_TO_MARKDOWN = pathlib.Path(
    "src/tests/resources/documents/markdown_example.md"
)


class TestCharacterChunker(unittest.TestCase):
    """Tests for CharacterChunker."""

    @classmethod
    @typing.override
    def setUpClass(cls) -> None:
        cls.chunker = chunking.CharacterChunker(PATH_TO_DOCUMENT)

    def test_chunk(self) -> None:
        """A list of Document objects is successfully returned."""
        self.assertIsInstance(self.chunker.chunk()[0], documents.Document)


class TestRecursiveChunker(unittest.TestCase):
    """Tests for RecursiveChunker."""

    @classmethod
    @typing.override
    def setUpClass(cls) -> None:
        cls.chunker = chunking.RecursiveChunker(PATH_TO_DOCUMENT)

    def test_chunk(self) -> None:
        """A list of Document objects is successfully returned."""
        self.assertIsInstance(self.chunker.chunk()[0], documents.Document)


class TestHTMLHeaderChunker(unittest.TestCase):
    """Tests for HTMLHeaderChunker."""

    @classmethod
    @typing.override
    def setUpClass(cls) -> None:
        headers_to_split_on = [
            ("h1", "Header 1"),
        ]
        cls.chunker_header = chunking.HTMLHeaderChunker(
            PATH_TO_HTML,
            headers_to_split_on=headers_to_split_on,
        )

    def test_chunk(self) -> None:
        """A list of Document objects is successfully returned."""
        self.assertIsInstance(
            self.chunker_header.chunk()[0],
            documents.Document,
        )


class TestHTMLSectionChunker(unittest.TestCase):
    """Tests for HTMLSectionChunker."""

    @classmethod
    @typing.override
    def setUpClass(cls) -> None:
        headers_to_split_on = [
            ("h1", "Header 1"),
        ]
        cls.chunker_section = chunking.HTMLSectionChunker(
            PATH_TO_HTML,
            headers_to_split_on=headers_to_split_on,
        )

    def test_chunk(self) -> None:
        """A list of Document objects is successfully returned."""
        self.assertIsInstance(
            self.chunker_section.chunk()[0],
            documents.Document,
        )


class TestMarkdownHeaderChunker(unittest.TestCase):
    """Tests for MarkdownHeaderChunker."""

    @classmethod
    @typing.override
    def setUpClass(cls) -> None:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]
        cls.chunker = chunking.MarkdownHeaderChunker(
            PATH_TO_MARKDOWN,
            headers_to_split_on=headers_to_split_on,
        )

    def test_chunk(self) -> None:
        """A list of Document objects is successfully returned."""
        self.assertIsInstance(self.chunker.chunk()[0], documents.Document)


class TestSemanticChunker(unittest.TestCase):
    """Tests for SemanticChunker."""

    @classmethod
    @typing.override
    def setUpClass(cls) -> None:
        cls.chunker = chunking.SemanticChunker(
            PATH_TO_DOCUMENT,
            "sentence-transformers/all-MiniLM-L6-v2",
        )

    def test_chunk(self) -> None:
        """A list of Document objects is successfully returned."""
        self.assertIsInstance(self.chunker.chunk()[0], documents.Document)


class TestTokenChunker(unittest.TestCase):
    """Tests for TokenChunker."""

    @classmethod
    @typing.override
    def setUpClass(cls) -> None:
        cls.chunker = chunking.TokenChunker(
            PATH_TO_DOCUMENT,
        )

    def test_chunk(self) -> None:
        """A list of Document objects is successfully returned."""
        self.assertIsInstance(self.chunker.chunk()[0], documents.Document)
