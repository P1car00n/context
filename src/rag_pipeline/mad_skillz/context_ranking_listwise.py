"""A copy of context_ranking_listwise.py provideded by Parea AI.

Changes include not depending on the web UI, ability to communicate
with a local LLM and several other thigns that make it work
with the rest of the codebase better.
"""

from collections.abc import Callable

from parea.evals.utils import call_openai, ndcg
from parea.schemas.log import Log


def context_ranking_listwise_factory(
    question_field: str = "question",
    context_fields: list[str] | None = None,
    ranking_measurement: str = "ndcg",
    n_contexts_to_rank: int = 10,
    model: str | None = "gpt-3.5-turbo-16k",
    is_azure: bool | None = False,
) -> Callable[[Log], float]:
    """Copy of context_ranking_listwise_factory from Parea."""
    if n_contexts_to_rank < 1:
        raise ValueError("n_contexts_to_rank must be at least 1.")

    def listwise_reranking(query: str, contexts: list[str]) -> list[int]:
        """Uses a LLM to listwise rerank the contexts.

        Returns the indices of the contexts in the order of their
        relevance (most relevant to least relevant).
        """
        contexts_length = len(contexts)
        if contexts_length in (0, 1):
            return list(range(contexts_length))

        prompt = ""
        for i in range(len(contexts)):
            prompt += f"{i + 1} = {contexts[i]}\n"

        prompt += f"""Query = {query}
        Passages = [1, ..., {len(contexts)}]
        Sort the Passages by their relevance to the Query.
        Sorted Passages = ["""

        sorted_list = call_openai(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model=model,
            temperature=0.0,
            is_azure=is_azure,
        )

        s = sorted_list.strip("[] ").replace(" ", "")[16:]
        number_strings = s.split(",")
        return [int(num) for num in number_strings if num.isdigit()]

    def progressive_reranking(query: str, contexts: list[str]) -> list[int]:
        """Returns the indices of the contexts in the order of their relevance."""
        if len(contexts) <= n_contexts_to_rank:
            return listwise_reranking(query, contexts)

        window_size = n_contexts_to_rank
        window_step = n_contexts_to_rank // 2
        offset = len(contexts) - window_size

        indices = list(range(len(contexts)))

        while offset > 0:
            window_contexts = contexts[offset : offset + window_size]
            window_indices = indices[offset : offset + window_size]
            reranked_indices = listwise_reranking(query, window_contexts)
            contexts[offset : offset + window_size] = [
                window_contexts[i] for i in reranked_indices
            ]
            indices[offset : offset + window_size] = [
                window_indices[i] for i in reranked_indices
            ]

            offset -= window_step

        window_contexts = contexts[:window_size]
        window_indices = indices[:window_size]
        reranked_indices = listwise_reranking(query, window_contexts)
        contexts[:window_size] = [window_contexts[i] for i in reranked_indices]
        indices[:window_size] = [window_indices[i] for i in reranked_indices]

        return indices

    def context_ranking(log: Log) -> float:
        """Quantifies if the retrieved context is ranked by their relevancy."""
        question = log.inputs[question_field]
        contexts = log.inputs["context"]

        reranked_indices = progressive_reranking(question, contexts)
        print(reranked_indices)

        if ranking_measurement == "ndcg":
            return ndcg(reranked_indices, list(range(len(contexts))))
        else:
            raise NotImplementedError

    return context_ranking
