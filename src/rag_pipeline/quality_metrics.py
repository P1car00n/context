"""A common interface for all quality metrics.

All adhere to the strategy pattern as currently implemented.
"""

import abc
import typing

from parea.evals import general
from parea.schemas import log

from . import mad_skillz


class BaseEvaluation(abc.ABC):
    """Abstract base class defining common evaluation operations."""

    @abc.abstractmethod
    @typing.override
    def __init__(
        self,
        query: str,
        context: list[str],
        output: str,
        **kwargs: typing.Any,
    ) -> None:
        """Instantiate the class.

        Args:
            kwargs: Key-word arguments to pass to Parea SDK.
            query: The question asked by the user.
            context: The context provided to the LLM.
            output: The answer provided by the LLM.
        """
        self._log = log.Log(
            inputs={"question": query, "context": context},
            output=output,
        )

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
        super().__init__(query, context="", output=output)

    @typing.override
    def evaluate(self) -> float:
        return self._llm_factory(self._log)


class LLMGraderEval(BaseEvaluation):
    """Evaluation based on llm_grader_factory."""

    @typing.override
    def __init__(self, query: str, output: str, **kwargs: typing.Any) -> None:
        self._llm_factory = general.llm_grader_factory(**kwargs)
        super().__init__(query, context="", output=output)

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
        use_local: bool,
    ) -> None:
        super().__init__(
            query,
            context="",
            output=output,
        )
        self.use_local = use_local

    @typing.override
    def evaluate(self) -> float:
        _result = mad_skillz.self_check.self_check(self._log, self.use_local)
        return _result if _result is not None else 0


class LLMJudgeEval(BaseEvaluation):
    """Evaluation based on lm_vs_lm_factuality_factory."""

    @typing.override
    def __init__(self, query: str, output: str, **kwargs: typing.Any) -> None:
        self._llm_factory = mad_skillz.lm_vs_lm.lm_vs_lm_factuality_factory(
            **kwargs,
        )
        super().__init__(query, context="", output=output)

    @typing.override
    def evaluate(self) -> float:
        return self._llm_factory(self._log)


class ListwiseRerankingEval(BaseEvaluation):
    """Evaluation based on context_ranking_listwise_factory."""

    @typing.override
    def __init__(
        self,
        query: str,
        context: list[str],
        output: str,
        **kwargs: typing.Any,
    ) -> None:
        self._llm_factory = mad_skillz.context_ranking_listwise.context_ranking_listwise_factory(
            context_fields=["context"],
            **kwargs,
        )
        super().__init__(query, context, output)

    @typing.override
    def evaluate(self) -> float:
        return self._llm_factory(self._log)
