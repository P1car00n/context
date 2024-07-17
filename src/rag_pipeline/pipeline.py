"""The central module that orchestrates the entire RAG process."""

from . import (
    chunking,
    generating,
    loading,
    persisting,
    quality_metrics,
    retrieving,
)


class RAGPipeline:
    def __init__(
        self,
        loader: loading.BaseLoading,
        chunker: chunking.BaseChunker,
        persister: persisting.BaseStorage,
        retriever: retrieving.BaseRetriever,
        generator: generating.BaseGenerator,
        evaluator: quality_metrics.BaseEvaluation,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.persister = persister
        self.retriever = retriever
        self.generator = generator
        self.eveluator = evaluator

    def process_query(self, query):
        self.loader.load()
        documents = self.chunker.chunk()
        context = self.persister.get_vectorstore()
        self.retriever.retrieve()
        result = self.generator.generate()
        self.eveluator.evaluate()
        print("Aww, yeah")
