import logging
import os
import sys
import re
import uuid

if sys.platform == 'linux':
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from apify_client import ApifyClient
import chromadb
from chromadb.config import Settings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import QianfanLLMEndpoint, Tongyi
from langchain_community.utilities import (BingSearchAPIWrapper,
                                           GoogleSearchAPIWrapper,
                                           GoogleSerperAPIWrapper)
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel

URL_EXCLUDE_LIST = [
    r'163\.com',
    r'ys80007\.com',
    r'wyzxwk\.com',
    r'bijie.gov.cn',
    r'yongxiu.gov.cn',
]

COMPANY_NAME = 'Theranos'

QUERY = f'''What is the negative news about this company? 
Summarize no more than 3 major ones, and itemizing each one in a seperate line.
'''

QA_TEMPLATE = '''
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use five sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:
'''

class QAInput(BaseModel):
    key_words: str
    query: str = QUERY
    num_results: int = 10
    search_engine_wrapper: str = 'Bing'
    llm_provider: str = 'Alibaba'


def qa_over_docs(qaio: QAInput):
    logging.info(f'Getting URLs from {qaio["search_engine_wrapper"]} search...')

    if qaio["search_engine_wrapper"] == 'Google' and qaio["num_results"] > 10:
        logging.error(
            'GoogleSearchAPI only supports up to 10 resutls, try other engines like Bing/GoogleSerper.')
        return

    if qaio["search_engine_wrapper"] == 'Google':
        search_engine = GoogleSearchAPIWrapper(k=qaio["num_results"])
    elif qaio["search_engine_wrapper"] == 'GoogleSerper':
        search_engine = GoogleSerperAPIWrapper(k=qaio["num_results"])
    elif qaio["search_engine_wrapper"] == 'Bing':
        search_engine = BingSearchAPIWrapper(k=qaio["num_results"])
    else:
        logging.error(f'Search engine {qaio["search_engine_wrapper"]} not supported.')
        return

    if qaio["llm_provider"] == 'Alibaba':
        llm = Tongyi(model_name='qwen-max', temperature=0)
    elif qaio["llm_provider"] == 'Baidu':
        llm = QianfanLLMEndpoint(model='ERNIE-Bot', temperature=0.01)
    elif qaio["llm_provider"] == 'OpenAI':
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    else:
        logging.error(f'LLM provider {qaio["llm_provider"]} not supported.')
        return

    search_results = search_engine.results(qaio["key_words"], num_results=qaio["num_results"])
    if qaio["search_engine_wrapper"] == 'GoogleSerper':
        urls = [item['link'] for item in search_results['organic']]
    else:
        urls = [item['link'] for item in search_results]

    urls_filtered_list = []
    for url in urls:
        include_flag = True
        for exclude_pattern in URL_EXCLUDE_LIST:
            if re.search(exclude_pattern, url):
                include_flag = False
                break
        if include_flag:
            urls_filtered_list.append(url)

    urls_filtered = [{'url': url} for url in urls_filtered_list]

    logging.info('Getting detailed web content from each URL...')

    apify_client = ApifyClient(os.getenv('APIFY_API_TOKEN'))
    actor_call = apify_client.actor('apify/website-content-crawler').call(
        run_input={
            'startUrls': urls_filtered,
            'crawlerType': 'cheerio',
            'maxCrawlDepth': 0,
            'maxSessionRotations': 0,
            'proxyConfiguration': {'useApifyProxy': True},
        })
    apify_dataset = apify_client.dataset(
        actor_call['defaultDatasetId']).list_items().items

    records = [rec for rec in apify_dataset if rec['crawl']['httpStatusCode'] < 300
               and len(rec['text']) >= 50]
    
    logging.info(f'Vectorizing documents into ChromaDB...')

    docs = []
    for rec in records:
        doc = Document(page_content=rec['text'],
                       metadata={'source': rec['url']})
        docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunked_docs = splitter.split_documents(docs)

    client = chromadb.PersistentClient(
        path='./chroma',
        settings=Settings(anonymized_telemetry=False))

    collection_name = uuid.uuid3(uuid.NAMESPACE_DNS, qaio["key_words"]).hex

    langchain_chroma = Chroma(
        collection_name,
        embedding_function=HuggingFaceEmbeddings(),
        client=client
    )

    ids = langchain_chroma.add_documents(chunked_docs)
    logging.info(
        f'{len(ids)} documents were added into collection {collection_name}')

    mmr_retriever = langchain_chroma.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 3, 'fetch_k': 5}
    )

    logging.info(f'Documents QA using LLM provied by {qaio["llm_provider"]}...')

    rag_prompt = PromptTemplate.from_template(QA_TEMPLATE)
    rag_chain = (
        {'context': mmr_retriever, 'question': RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    try:
        answer = rag_chain.invoke(qaio["query"])
        return {
            'query': qaio["query"],
            'answer': answer
        }
    except ValueError:  # Occurs when there is no relevant information
        return {
            'query': qaio["query"],
            'answer': 'I don\'t know.'
        }
    

if __name__ == '__main__':
    from fastapi import FastAPI
    import uvicorn
    from langchain.schema.runnable import RunnableLambda

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    fca_app = FastAPI(
        title='Financial Crime Analysis using LLM',
        version='1.0',
    )

    qa_runnable = RunnableLambda(qa_over_docs).with_types(
        input_type=QAInput
    )

    add_routes(
        fca_app,
        qa_runnable,
        path='/fca/qa',
    )

    uvicorn.run(fca_app, host='localhost', port=8000)
