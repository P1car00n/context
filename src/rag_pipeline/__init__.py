from . import pipeline


def main():
    print("Main")
    query = "What is the capital of France?"
    pipeline1 = pipeline.RAGPipeline()
    result = pipeline1.process_query(query)
    print(result)
