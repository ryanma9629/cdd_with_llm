import logging
import os
import re
import sys
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

if sys.platform == 'linux':
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from apify_client import ApifyClient
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import QianfanLLMEndpoint, Tongyi
from langchain_community.utilities import (BingSearchAPIWrapper,
                                           GoogleSearchAPIWrapper,
                                           GoogleSerperAPIWrapper)
from langchain_community.utilities.apify import ApifyWrapper
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

app = FastAPI(

)

EXCLUDE_LIST = [
    r'163\.com',
    r'ys80007\.com',
    r'wyzxwk\.com',
    r'bijie.gov.cn',
    r'yongxiu.gov.cn'
]

CHROMA_PERSISTENT_DIR = './chroma'

QA_TEMPLATE = '''
利用下列信息回答后面的问题。如果你不知道答案就直接回答'不知道'，不要主观编造答案。
最多使用五句话，回答尽量简洁。

{context}

问题：{question}

有价值的回答：
'''

QUERY =f'''
这家公司有哪些负面新闻？总结不超过3条主要的，每条独立一行列出。
'''

SEARCH_SUFFIX = '负面新闻'

IDONTKNOW = 'I don\'t know.'

def sas_json_wrapper(data):
    return {
        'item': data,
        'start': 0,
        'limit': len(data),
        'count': len(data)
    }


@app.get('/cdd_with_llm/fetch_and_store')
async def fetch_and_store(company_name: str,
                          search_engine_wrapper: str = 'Bing',
                          num_results: int = 10,
                          ):
    logging.info(f'Getting URLs from {search_engine_wrapper} search...')

    if search_engine_wrapper == 'Google' and num_results > 10:
        logging.error(
            'GoogleSearchAPI only supports up to 10 resutls, try other engines like Bing/GoogleSerper.')
        return

    if search_engine_wrapper == 'Google':
        search_engine = GoogleSearchAPIWrapper(k=num_results)
    elif search_engine_wrapper == 'GoogleSerper':
        search_engine = GoogleSerperAPIWrapper(k=num_results)
    elif search_engine_wrapper == 'Bing':
        search_engine = BingSearchAPIWrapper(k=num_results)
    else:
        logging.error(f'Search engine {search_engine_wrapper} not supported.')
        return

    raw_search_results = search_engine.results(
        company_name + ' ' + SEARCH_SUFFIX, num_results=num_results)
    if search_engine_wrapper == 'GoogleSerper':
        raw_search_results = raw_search_results['organic']

    search_results = [{'title': item['title'], 'snippet': item['snippet'],
                       'url': item['link']} for item in raw_search_results]

    url_filtered_results = []
    for item in search_results:
        include_flag = True
        for exclude_pattern in EXCLUDE_LIST:
            if re.search(exclude_pattern, item['url']):
                include_flag = False
                break
        if include_flag:
            url_filtered_results.append(item)

    logging.info('Getting detailed web content from each URL...')
    apify_client = ApifyClient(os.getenv('APIFY_API_TOKEN'))
    actor_call = apify_client.actor('apify/website-content-crawler').call(
        run_input={
            'startUrls': [{'url': item['url']} for item in url_filtered_results],
            'crawlerType': 'cheerio',
            'maxCrawlDepth': 0,
            'maxSessionRotations': 0,
            'proxyConfiguration': {'useApifyProxy': True},
        })
    apify_dataset = apify_client.dataset(
        actor_call['defaultDatasetId']).list_items().items

    records = [rec for rec in apify_dataset if rec['crawl']['httpStatusCode'] < 300
               and len(rec['text']) >= 50]

    docs = []
    for rec in records:
        doc = Document(page_content=rec['text'],
                       metadata={'source': rec['url']})
        docs.append(doc)

    logging.info(f'Vectorizing documents into ChromaDB...')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunked_docs = splitter.split_documents(docs)

    client = chromadb.PersistentClient(
        path=CHROMA_PERSISTENT_DIR,
        settings=Settings(anonymized_telemetry=False))

    collection_name = uuid.uuid3(uuid.NAMESPACE_DNS, company_name).hex

    langchain_chroma = Chroma(
        collection_name,
        embedding_function=HuggingFaceEmbeddings(),
        client=client
    )

    ids = langchain_chroma.add_documents(chunked_docs)
    logging.info(
        f'{len(ids)} documents were added into collection {collection_name}')

    return sas_json_wrapper(data=url_filtered_results)

@app.get('/cdd_with_llm/qa_over_docs')
async def qa_over_docs(company_name: str,
                 qa_template: str = QA_TEMPLATE,
                 query: str = QUERY,
                 llm_provider: str = 'Alibaba'):
    
    collection_name = uuid.uuid3(uuid.NAMESPACE_DNS, company_name).hex

    client = chromadb.PersistentClient(
        path=CHROMA_PERSISTENT_DIR,
        settings=Settings(anonymized_telemetry=False))

    langchain_chroma = Chroma(
        collection_name,
        embedding_function=HuggingFaceEmbeddings(),
        client=client
    )

    mmr_retriever = langchain_chroma.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 3, 'fetch_k': 5}
    )

    logging.info(f'Documents QA using LLM provied by {llm_provider}...')
    if llm_provider == 'Alibaba':
        llm = Tongyi(model_name='qwen-max', temperature=0)
    elif llm_provider == 'Baidu':
        llm = QianfanLLMEndpoint(model='ERNIE-Bot', temperature=0.01)
    elif llm_provider == 'OpenAI':
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    else:
        logging.error(f'LLM provider {llm_provider} not supported.')
        return

    rag_prompt = PromptTemplate.from_template(qa_template)
    rag_chain = (
        {'context': mmr_retriever, 'question': RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    try:
        answer = rag_chain.invoke(query)
        return sas_json_wrapper(data=answer)
    except ValueError:  # Occurs when there is no relevant information
        return sas_json_wrapper(data=IDONTKNOW)