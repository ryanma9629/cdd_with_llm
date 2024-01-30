import json
import logging
import os
import re
import sys
import uuid
from typing import Union, Optional, List, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

if sys.platform == 'linux':
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from apify_client import ApifyClient
from chromadb.config import Settings

from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import DashScopeEmbeddings, HuggingFaceEmbeddings
# from langchain_community.llms import QianfanLLMEndpoint, Tongyi
from langchain_community.utilities.bing_search import BingSearchAPIWrapper
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, AzureOpenAI, AzureChatOpenAI, AzureOpenAIEmbeddings

app = FastAPI(

)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_PERSISTENT_DIR = './chroma'

def sas_json_wrapper(data):
    return {
        'items': data,
        'start': 0,
        'limit': len(data),
        'count': len(data)
    }


def sas_json_wrapper2(data):
    return {
        'items': data,
        'start': 0,
        'limit': 1,
        'count': 1
    }


@app.get('/cdd_with_llm/test')
async def test():
    return sas_json_wrapper2(
        {
            "title": "SAS Intitute",
            "snippet": "When everything is data, discover what matters with SAS Viya",
            "url": "http://www.sas.com"
        },
    )


@app.get('/cdd_with_llm/fetch_and_store')
async def fetch_and_store(company_name: str,
                          search_suffix: Union[str, None] = None,
                        #   search_engine_wrapper: str = 'Bing',
                          num_results: int = 10,
                          mkt: str = 'zh-CN',
                          ):
    logging.info(f'Getting URLs from Bing search...')

    if mkt == 'zh-CN':
        if search_suffix is None:
            search_suffix = '负面新闻'
    elif mkt == 'en-US':
        if search_suffix is None:
            search_suffix = 'negative news'
    
    search_engine = BingSearchAPIWrapper(k=num_results, search_kwargs={'mkt': mkt})

    raw_search_results = search_engine.results(
        company_name + ' ' + search_suffix, num_results=num_results)

    search_results = [{'title': item['title'], 'snippet': item['snippet'],
                       'url': item['link']} for item in raw_search_results]

    logging.info('Getting detailed web content from each URL...')
    apify_client = ApifyClient(os.getenv('APIFY_API_TOKEN'))
    actor_call = apify_client.actor('apify/website-content-crawler').call(
        run_input={
            'startUrls': [{'url': item['url']} for item in search_results],
            'crawlerType': 'cheerio',
            'maxCrawlDepth': 0,
            'maxSessionRotations': 0,
            'maxRequestRetries': 1,
            'readableTextCharThreshold': 100,
            'proxyConfiguration': {'useApifyProxy': True},
        })
    apify_dataset = apify_client.dataset(
        actor_call['defaultDatasetId']).list_items().items

    records = [rec for rec in apify_dataset if rec['crawl']['httpStatusCode'] < 300]
    
    persistent_client = chromadb.PersistentClient(
        path=CHROMA_PERSISTENT_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    collection_name = uuid.uuid3(uuid.NAMESPACE_DNS, company_name).hex

    collection = persistent_client.get_or_create_collection(
        collection_name,
    )
    collection.upsert(
        ids=[item['url'] for item in records],
        documents=[item['text'] for item in records],
        metadatas=[{'source': item['url']} for item in records],
        # Fake embedding, only for raw doc storage
        embeddings=[[0]] * len(records)
    )

    return sas_json_wrapper(data=search_results)

class QAInput(BaseModel):
    company_name: str
    # qa_template: str = QA_TEMPLATE
    query: Union[str, None] = None
    lang: str = 'zh-CN'
    # llm_provider: str = 'Alibaba'


@app.post('/cdd_with_llm/qa_over_docs')
async def qa_over_docs(qa_input: QAInput):
    
    persistent_client = chromadb.PersistentClient(
        path=CHROMA_PERSISTENT_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    # huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    #     api_key=os.getenv('HUGGINGFACE_API_TOKEN'),
    #     model_name='sentence-transformers/all-MiniLM-L6-v2'
    # )
    collection_name = uuid.uuid3(uuid.NAMESPACE_DNS, qa_input.company_name).hex
    try:
        collection = persistent_client.get_collection(collection_name)
    except ValueError:
        logging.error('Collection does not exist.')
        return
    
    chroma_docs = collection.get()

    langchain_docs = []
    for i in range(len(chroma_docs['ids'])):
        langchain_doc = Document(
            page_content=chroma_docs['documents'][i], metadata=chroma_docs['metadatas'][i])
        langchain_docs.append(langchain_doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunked_docs = splitter.split_documents(langchain_docs)
    
    langchain_chroma = Chroma(
        collection_name='Ephemeral_Collection_for_QA',
        client_settings=Settings(anonymized_telemetry=False),
        # embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        embedding_function=AzureOpenAIEmbeddings(azure_deployment=os.getenv('AZURE_OPENAI_EMB_DEPLOY'))
    )

    langchain_chroma.add_documents(chunked_docs)

    mmr_retriever = langchain_chroma.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 3, 'fetch_k': 5}
    )

    logging.info(f'Documents QA using LLM provied by Azure OpenAI...')
    # if qa_input.llm_provider == 'Alibaba':
    #     llm = Tongyi(model_name='qwen-max', temperature=0)
    # elif qa_input.llm_provider == 'Baidu':
    #     llm = QianfanLLMEndpoint(model='ERNIE-Bot', temperature=0.01)
    # elif qa_input.llm_provider == 'OpenAI':
    #     llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    # else:
    #     logging.error(f'LLM provider {qa_input.llm_provider} is not supported.')
    #     return

    if qa_input.lang == 'zh-CN':
        qa_template = '''
利用下列信息回答后面的问题。如果你不知道答案就直接回答'不知道'，不要主观编造答案。
最多使用五句话，回答尽量简洁。

{context}

问题：{question}

有价值的回答：
'''
        if qa_input.query is None:
            qa_input.query = '''
这家公司有哪些负面新闻？总结不超过3条主要的，每条独立一行列出。
'''
    elif qa_input.lang == 'en-US':
        qa_template = '''
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use five sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:
''' 
        if qa_input.query is None:
            qa_input.query = '''
What is the negative news about this company? 
Summarize no more than 3 major ones, and itemizing each one in a seperate line.
'''
    else:
        logging.error(f'Language {qa_input.lang} is not supported.')
        return


    chat_llm = AzureChatOpenAI(
        azure_deployment=os.getenv('AZURE_OPENAI_LLM_DEPLOY'))
    # rag_prompt = PromptTemplate.from_template(qa_template)
    rag_prompt = ChatPromptTemplate.from_template(qa_template)
    rag_chain = (
        {'context': mmr_retriever, 'question': RunnablePassthrough()}
        | rag_prompt
        | chat_llm
        | StrOutputParser()
    )

    try:
        answer = rag_chain.invoke(qa_input.query)
        qa = {
            'query': qa_input.query,
            'answer': answer
        }
    except ValueError:  # Occurs when there is no relevant information
        qa = {
            'query': qa_input.query,
            'answer': 'I don\'t know'
        }

    if not os.path.exists('qa'):
        os.makedirs('qa')
    filename = './qa/' + collection_name + '.json'

    with open(filename, 'w') as f:
        json.dump(qa, f)

    return filename


@app.get('/cdd_with_llm/get_qa')
async def get_qa(company_name: str):
    collection_name = uuid.uuid3(uuid.NAMESPACE_DNS, company_name).hex
    filename = './qa/' + collection_name + '.json'
    if not os.path.exists(filename):
        logging.error('QA file does not exist.')
        return
    with open(filename, 'r') as f:
        qa = json.load(f)
    return sas_json_wrapper2(qa)
