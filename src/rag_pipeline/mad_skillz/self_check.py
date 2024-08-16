"""A copy of self_check.py provideded by Parea AI.

Changes include not depending on the web UI, ability to communicate
with a local LLM and several other thigns that make it work
with the rest of the codebase better.
"""

from langchain_community.llms import llamafile
from parea.evals.utils import call_openai, sent_tokenize
from parea.schemas.log import Log


def self_check(log: Log) -> float | None:
    """Copy of self_check."""
    question = log.inputs["question"]

    n_sampled_outputs = 5
    sampled_outputs = []
    use_llamafile = True
    for _ in range(n_sampled_outputs):
        if use_llamafile:
            response = llamafile.Llamafile().invoke(f"Question: {question}")
        else:
            response = call_openai(
                messages=[
                    {
                        "role": "user",
                        "content": f"""
Question: {question}""",
                    },
                ],
                model="gpt-4o-mini",
                temperature=1.0,
            )
        sampled_outputs.append(response)

    sentences = sent_tokenize(log.output)

    if len(sentences) == 0:
        return 0.0

    sentences_scores = []
    for sentence in sentences:
        scores = []
        for sampled_output in sampled_outputs:
            response = call_openai(
                messages=[
                    {
                        "role": "user",
                        "content": f"""Context: {sampled_output}
Sentence: {sentence}
Is the sentence supported by the context above?
Answer Yes or No:""",
                    },
                ],
                model="gpt-4o-mini",
                temperature=0.0,
            )
            scores.append(float("yes" in response.lower()))
        sentences_scores.append(sum(scores) / len(scores))

    return sum(sentences_scores) / len(sentences_scores)
