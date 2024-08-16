"""Copy of lm_vs_lm.py provideded by Parea AI.

Changes include not depending on the web UI, ability to communicate
with a local LLM and several other thigns that make it work
with the rest of the codebase better.
"""

from collections.abc import Callable

from langchain_community.llms import llamafile
from parea.evals.utils import call_openai
from parea.schemas.log import Log


def lm_vs_lm_factuality_factory(
    examiner_model: str = "gpt-4",
    is_azure: bool | None = False,
) -> Callable[[Log], float]:
    """Copy of lm_vs_lm."""

    def lm_vs_lm_factuality(log: Log) -> float:
        output = log.output
        messages_examinee = []

        # ask examiner for follow-up questions
        setup_prompt = f"""Your goal is to try to verify the correctness of the following claim: "{output}", based on the background information you will gather. To gather this, You will provide short questions whose purpose will be to verify the correctness of the claim, and I will reply to you with the answers to these. Hopefully, with the help of the background questions and their answers, you will be able to reach a conclusion as to whether the claim is correct or possibly incorrect. Please keep asking questions as long as you’re yet to be sure regarding the true veracity of the claim. Please start with the first questions."""
        messages_examiner = [{"role": "user", "content": setup_prompt}]
        follow_up_questions = call_openai(
            model=examiner_model,
            messages=messages_examiner,
            temperature=0.0,
            is_azure=is_azure,
        )
        messages_examiner += [
            {"role": "assistant", "content": follow_up_questions},
        ]
        n_rounds_follow_up_questions = 1

        follow_up_prompt = """(i) Do you have any follow-up questions? Please answer with Yes or No.
    (ii) What are the follow-up questions?"""
        # ask examinee follow-up questions until they reach a conclusion
        while follow_up_questions is not None:
            messages_examinee += [
                {"role": "user", "content": follow_up_questions}
            ]
            follow_up_answers = llamafile.Llamafile().invoke(
                follow_up_questions,
            )

            messages_examiner.append(
                {"role": "assistant", "content": follow_up_answers},
            )

            if n_rounds_follow_up_questions > 3:
                break
            else:
                messages_examiner.append(
                    {"role": "user", "content": follow_up_prompt},
                )
                n_rounds_follow_up_questions += 1

            examiner_response = call_openai(
                model=examiner_model,
                messages=messages_examiner,
                temperature=0.0,
                is_azure=is_azure,
            )
            messages_examiner += [
                {"role": "assistant", "content": examiner_response},
            ]
            if "yes" in examiner_response.lower():
                follow_up_questions = examiner_response
                messages_examinee += (
                    [
                        {"role": "assistant", "content": follow_up_answers},
                    ],
                )
            else:
                follow_up_questions = None

        # ask examiner for their conclusion
        factuality_decision_prompt = """Based on the interviewee’s answers to your questions, what is your conclusion regarding the correctness of the claim? Do you think it is correct or incorrect?"""
        messages_examiner += [
            {"role": "user", "content": factuality_decision_prompt},
        ]
        examiner_response = call_openai(
            model=examiner_model,
            messages=messages_examiner,
            temperature=0.0,
            is_azure=is_azure,
        )
        return float("incorrect" not in examiner_response.lower())

    return lm_vs_lm_factuality
