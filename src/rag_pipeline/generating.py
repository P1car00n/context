"""A common interface for all generators.

All adhere to the strategy pattern as currently implemented.
"""

import abc
import typing

import langchain_anthropic
import langchain_openai
from langchain import hub
from langchain_community.llms import gpt4all, llamafile
from langchain_core import (
    documents,
    language_models,
    output_parsers,
    retrievers,
    runnables,
)

PROMPT = hub.pull("rlm/rag-prompt")


def format_docs(docs: typing.Iterable[documents.Document]) -> str:
    """Combine all document objects into one string.

    Anything apart from the context is discarded.

    Args:
        docs: An iterable of docuemnts objects.

    Returns:
        A string combining the context of all documents.
    """
    return "\n\n".join(doc.page_content for doc in docs)


class BaseGenerator(abc.ABC):
    """Abstract base class defining common generation operations."""

    @abc.abstractmethod
    @typing.override
    def __init__(
        self,
        retriever: retrievers.BaseRetriever,
        llm: language_models.BaseLanguageModel,
    ) -> None:
        """Instantiate the class.

        Args:
            retriever: Object capable of retrieving document objects.
            llm: Language model to use for generation.
        """
        self._runnable_sequence = (
            {
                "context": retriever | format_docs,
                "question": runnables.RunnablePassthrough(),
            }
            | PROMPT
            | llm
            | output_parsers.StrOutputParser()
        )

    @abc.abstractmethod
    def generate(self, query: str) -> str:
        """Generate an answer.

        Args:
            query: The question to be answered by the LLM.

        Returns:
            A string containing the asnwer geenrated by the LLM.
        """
        return self._runnable_sequence.invoke(query)


class OpenAIGenerator(BaseGenerator):
    """OpenAPI generators."""

    @typing.override
    def __init__(
        self,
        retriever: retrievers.BaseRetriever,
        model_name: str = "gpt-4o-mini",
        **kwargs: typing.Any,
    ) -> None:
        """Instantiate the class.

        OPENAI_API_KEY must be set in the environemnt.

        Args:
            retriever: Object capable of retrieving document objects.
            model_name: The name of the model to use. Defaults to "gpt-4o-mini".
            kwargs: Key-word arguments to pass to the model.
        """
        super().__init__(
            retriever,
            langchain_openai.ChatOpenAI(model=model_name, **kwargs),
        )

    @typing.override
    def generate(self, query: str) -> str:
        return super().generate(query)


class AnthropicGenerator(BaseGenerator):
    """Anthropic generators."""

    @typing.override
    def __init__(
        self,
        retriever: retrievers.BaseRetriever,
        model_name: str = "claude-3-5-sonnet-20240620",
        **kwargs: typing.Any,
    ) -> None:
        """Instantiate the class.

        ANTHROPIC_API_KEY must be set in the environemnt.

        Args:
            retriever: Object capable of retrieving document objects.
            model_name: The name of the model to use.
                Defaults to "claude-3-5-sonnet-20240620".
            kwargs: Key-word arguments to pass to the model.
        """
        super().__init__(
            retriever,
            langchain_anthropic.ChatAnthropic(model=model_name, **kwargs),
        )

    @typing.override
    def generate(self, query: str) -> str:
        return super().generate(query)


class LLAMAFileGenerator(BaseGenerator):
    """LLAMAFile generators."""

    @typing.override
    def __init__(
        self,
        retriever: retrievers.BaseRetriever,
        **kwargs: typing.Any,
    ) -> None:
        """Instantiate the class.

        This model runs locally. Run the model first.

        Args:
            retriever: Object capable of retrieving document objects.
            kwargs: Key-word arguments to pass to the model.
        """
        super().__init__(
            retriever,
            llamafile.Llamafile(**kwargs),
        )

    @typing.override
    def generate(self, query: str) -> str:
        return super().generate(query)


class GPT4AllGenerator(BaseGenerator):
    """GPT4All generators."""

    @typing.override
    def __init__(
        self,
        retriever: retrievers.BaseRetriever,
        model_path: str,
        **kwargs: typing.Any,
    ) -> None:
        """Instantiate the class.

        This model runs locally. Install the model first.

        Args:
            retriever: Object capable of retrieving document objects.
            model_path: Path to the installed model.
            kwargs: Key-word arguments to pass to the model.
        """
        super().__init__(
            retriever,
            gpt4all.GPT4All(model=model_path, **kwargs),
        )

    @typing.override
    def generate(self, query: str) -> str:
        return super().generate(query)
