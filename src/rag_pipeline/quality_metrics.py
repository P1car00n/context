"""A common interface for all quality metrics.

All adhere to the strategy pattern as currently implemented.
"""

import abc
import typing

from parea.evals import general, rag
from parea.schemas import log


class BaseEvaluation(abc.ABC):
    """Abstract base class defining common evaluation operations."""

    @abc.abstractmethod
    @typing.override
    def __init__(self, query: str, output: str, **kwargs: typing.Any) -> None:
        """Instantiate the class.

        Args:
            kwargs: Key-word arguments to pass to Parea SDK.
            query: The question asked by the user.
            output: The answer provided by the LLM.
        """
        self._log = log.Log(inputs={"question": query}, output=output)

    @abc.abstractmethod
    def evaluate(self) -> float:
        """Return a grade representing the quality of the pipeline.

        Returns:
            A numerical representation of quality. From 0 to 1,
                where 1 is the best.
        """


class RAGAsEval(BaseEvaluation):
    """Evaluation based on answer_relevancy_factory."""

    @typing.override
    def __init__(self, query: str, output: str, **kwargs: typing.Any) -> None:
        self._llm_factory = general.answer_relevancy_factory(**kwargs)
        super().__init__(query, output)

    @typing.override
    def evaluate(self) -> float:
        return self._llm_factory(self._log)


class LLMGraderEval(BaseEvaluation):
    """Evaluation based on llm_grader_factory."""

    @typing.override
    def __init__(self, query: str, output: str, **kwargs: typing.Any) -> None:
        self._llm_factory = general.llm_grader_factory(**kwargs)
        super().__init__(query, output)

    @typing.override
    def evaluate(self) -> float:
        return self._llm_factory(self._log)


class SelfCheckEval(BaseEvaluation):
    """Evaluation based on self_check."""

    @typing.override
    def __init__(
        self,
        query: str,
        output: str,
    ) -> None:
        super().__init__(query, output)

    @typing.override
    def evaluate(self) -> float:
        _result = general.self_check(self._log)
        return _result if _result is not None else 0


class LLMJudgeEval(BaseEvaluation):
    """Evaluation based on lm_vs_lm_factuality_factory."""

    @typing.override
    def __init__(self, query: str, output: str, **kwargs: typing.Any) -> None:
        self._llm_factory = general.lm_vs_lm_factuality_factory(**kwargs)
        super().__init__(query, output)

    @typing.override
    def evaluate(self) -> float:
        return self._llm_factory(self._log)


class ListwiseRerankingEval(BaseEvaluation):
    """Evaluation based on context_ranking_listwise_factory."""

    @typing.override
    def __init__(self, query: str, output: str, **kwargs: typing.Any) -> None:
        self._llm_factory = rag.context_ranking_listwise_factory(**kwargs)
        super().__init__(query, output)

    @typing.override
    def evaluate(self) -> float:
        return self._llm_factory(self._log)
