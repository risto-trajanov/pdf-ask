# pylint: disable-all
import streamlit as st
from llama_index import download_loader, GPTVectorStoreIndex, GPTTreeIndex, GPTKeywordTableIndex
from llama_index.evaluation import DatasetGenerator, QueryResponseEvaluator
from llama_index import ServiceContext, Document, QuestionAnswerPrompt, PromptHelper
from llama_index import StorageContext, load_index_from_storage, load_graph_from_storage
from llama_index.tools import QueryEngineTool, ToolMetadata
from newspaper import Article
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


    def query_chatgpt(self, query):
        # check if query is in self.cache
        if query in self.cache:
            print("returning from cache")
            return self.cache[query]
        else:          
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages = [{"role": "user", "content": query}],
                temperature = 0)
            response = completion.choices[0].message.content
            self.cache[query] = response
        return response

    def extract_content_from_link(self, link):
        """
        Extracts the content from a link. It uses the Article library to download and parse the article.
        :param link: The link to extract the content from.
        :return: The extracted content, title and summary.
        """
        # Create an Article object
        article = Article(link)

        try:
            # Download and parse the article
            article.download()
            article.parse()
            return article.text, article.title, article.summary
        except:
            return None
        
    def make_index_doc(self, doc, file_path):
        # create an index from the documents, chunking the index at 1024
        index = GPTVectorStoreIndex.from_documents(doc, service_context=self.service_context, chunk_size=self.chunk_size)
        # persist the index to a file
        index.storage_context.persist(f'./index/{file_path}')
        return f'./index/{file_path}'

    def make_index_html(self, file_path):
        doc = self.loader.load_data(file=Path(file_path), split_documents=False)
        file_path = file_path.split('/')[-1].split('.')[0]
        return self.make_index_doc(doc, file_path)

    def make_index_text(self, text, file_path):
        return self.make_index_doc([Document(text=text)], file_path)

    def make_index_pdf(self, file_path):
        doc = self.loader_pdf.load_data(file=Path(file_path))
        file_path = file_path.split('/')[-1].split('.')[0]
        return self.make_index_doc(doc, file_path)
    
    def make_doc_text(self, text):
        return Document(text=text)

    def load_index(self, file_path):
        # load the index
        print(file_path)
        storage_context = StorageContext.from_defaults(persist_dir=file_path)
        # load index
        index = load_index_from_storage(storage_context)
        return index

    def process_text_input(self, text, _query_engine):
        # Process the submitted text
        response = _query_engine.query(text)
        return response

    def get_text_from_index(self, index, query):
        retriever = index.as_retriever(service_context=self.service_context)
        query_engine = RetrieverQueryEngine.from_args(retriever, response_mode='no_text')  
        response = query_engine.query(query)
        company_report = ''
        for node in response.source_nodes:
            company_report += node.node.text + '\n'
        return company_report

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
            similarity_top_k=5,
        )
        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        return query_engine

    def create_graph(self, bot_name, indexes, summaries):
        folder = bot_name
        # graph_storage_context = StorageContext.from_defaults(persist_dir=f'./graph/{folder}')
        print("creating graph")
        graph = ComposableGraph.from_indices(
            GPTListIndex,
            [val for key, val in indexes.items()],
            [val for key, val in summaries.items()],
            service_context=self.service_context,
        )
        # set the ID
        graph.root_index.set_index_id(bot_name)

        # persist to storage
        graph.root_index.storage_context.persist(persist_dir=f'./graph/{folder}')
        print(graph.all_indices)
        return graph
    
    def create_agent(self, query_engines, descriptions=None):
        query_engine_tools = []
        for company, query_engine in query_engines.items():
            year = company.split('_')[-1]
            company = company.split('_')[0]
            if descriptions:
                query_engine_tools.append(
                    QueryEngineTool(
                        query_engine=query_engine,
                        metadata=ToolMetadata(
                            name=company,
                            description=descriptions[company],
                        ),
                    )
                )
            else:
                query_engine_tools.append(
                    QueryEngineTool(
                        query_engine=query_engine,
                        metadata=ToolMetadata(
                            name=company,
                            description=f"Provides information about {company} financials for year {year}. "
                            "Use a detailed plain text question as input to the tool.",
                        ),
                    )
                )
        
        query_engine = OpenAIAgent.from_tools(
            query_engine_tools,
            llm=self.llm_predictor.llm,
            verbose=True,
        )
        return query_engine