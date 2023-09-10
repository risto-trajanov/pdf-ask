# pylint: disable-all
import streamlit as st
from llama_index import download_loader, GPTVectorStoreIndex, GPTTreeIndex, GPTKeywordTableIndex
from llama_index.evaluation import DatasetGenerator, QueryResponseEvaluator
from llama_index import ServiceContext, Document, QuestionAnswerPrompt, PromptHelper
from llama_index import StorageContext, load_index_from_storage, load_graph_from_storage
from llama_index.tools import QueryEngineTool, ToolMetadata
from pathlib import Path
from llama_index import (
    GPTVectorStoreIndex,
)
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index import (
    GPTVectorStoreIndex, 
    GPTSimpleKeywordTableIndex, 
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext
)
import openai
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index import GPTListIndex, LLMPredictor
from langchain import OpenAI
from llama_index.composability import ComposableGraph
import os
from langchain.chat_models import ChatOpenAI
import logging
import sys
from llama_index.agent import OpenAIAgent

# create a class that will be consistent of functions that will help me work with llama index

class LlamaHelper():
    def __init__(self,cache, logger=None, chunk_size = 200) -> None:
        os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
        openai.api_key = os.environ["OPENAI_API_KEY"]
        if logger == 'Info':
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)
            logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
        elif logger == 'Debug':
            logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
            logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
        # define prompt helper
        # set maximum input size
        max_input_size = 4096
        # set number of output tokens
        num_output = 2048
        # set maximum chunk overlap
        max_chunk_overlap = 0.1
        self.chunk_size = chunk_size
        prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

        self.llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.2, 
                                                        max_tokens=num_output, 
                                                        verbose=True, 
                                                        model_name='gpt-3.5-turbo-0613'))
        UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)
        self.service_context = ServiceContext.from_defaults(chunk_size=chunk_size, 
                                                    llm_predictor=self.llm_predictor, 
                                                    prompt_helper=prompt_helper)
        self.loader = UnstructuredReader()
        PDFReader = download_loader("PDFReader")
        self.loader_pdf = PDFReader()
        self.cache = cache 

        
    def make_index_doc(self, doc, file_path):
        # create an index from the documents, chunking the index at 1024
        index = GPTVectorStoreIndex.from_documents(doc, service_context=self.service_context, chunk_size=self.chunk_size)
        # persist the index to a file
        index.storage_context.persist(f'./index/{file_path}')
        return f'./index/{file_path}'

    def make_index_pdf(self, file_path):
        doc = self.loader_pdf.load_data(file=Path(file_path))
        file_path = file_path.split('/')[-1].split('.')[0]
        return self.make_index_doc(doc, file_path)
    
    def load_index(self, file_path):
        # load the index
        print(file_path)
        storage_context = StorageContext.from_defaults(persist_dir=file_path)
        # load index
        index = load_index_from_storage(storage_context)
        return index

    def get_query_engine_for_index(self, index, QA_PROMPT_TMPL=None, response_mode='compact'):
        if QA_PROMPT_TMPL:
            QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
            # configure response synthesizer
            response_synthesizer = get_response_synthesizer(
                response_mode=response_mode,
                text_qa_template=QA_PROMPT
                )
        else:
            # configure response synthesizer
            response_synthesizer = get_response_synthesizer(
                response_mode=response_mode
            )
        # configure retriever
        retriever = VectorIndexRetriever(
            index=index, 
            similarity_top_k=4,
        )
        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        return query_engine
