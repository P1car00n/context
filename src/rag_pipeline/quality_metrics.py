"""A common interface for all quality metrics.

All adhere to the strategy pattern as currently implemented.
"""

import abc
import typing

from parea.evals import general
from parea.schemas import log


class BaseEvaluation(abc.ABC):
    """Abstract base class defining common evaluation operations."""

    @abc.abstractmethod
    def evaluate(self) -> float:
        """Return a grade representing the quality of the pipeline..

        Returns:
            A numerical representation of quality. From 1 to 10,
                where 10 is the best.
        """


class LLMGraderEval(BaseEvaluation):
    """Evaluation based on using an LLM as a judge."""

    @typing.override
    def __init__(self, model_name: str) -> None:
        # TODO: add docs.
        self._llm_factory = general.llm_grader_factory(model_name)
        # TODO: Figure out how to better set the log object.
        self._log = log.Log()

    @typing.override
    def evaluate(self) -> float:
        return self._llm_factory(self._log)
