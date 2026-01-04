# debug_dp.py
import os
os.environ["CACHE_EMBEDDING_PROVIDER"] = "openai"
os.environ["ENABLE_SENTENCE_TRANSFORMERS"] = "0"
os.environ["ENABLE_MODEL_CACHE_PRELOAD"] = "0"

print("STEP0 import DocumentProcessor ...")
from graphrag_agent.pipelines.ingestion.document_processor import DocumentProcessor
print("STEP0 OK")

print("STEP1 new FileReader ...")
from graphrag_agent.pipelines.ingestion.file_reader import FileReader
fr = FileReader("files")
print("STEP1 OK")

print("STEP2 new ChineseTextChunker ...")
from graphrag_agent.pipelines.ingestion.text_chunker import ChineseTextChunker
ck = ChineseTextChunker(500, 100)
print("STEP2 OK")

print("STEP3 new GraphChunker factory ...")
from graphrag_agent.pipelines.ingestion.specialized_chunkers import create_graph_chunker, create_rag_chunker
print("STEP3 import OK")

print("STEP4 create_graph_chunker() ...")
g = create_graph_chunker()
print("STEP4 OK")

print("STEP5 instantiate DocumentProcessor(default) ...")
dp = DocumentProcessor("files", 500, 100, chunker_mode="default")
print("STEP5 OK")
