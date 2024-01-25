import logging
import os
import re


from apify_client import ApifyClient

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import (DashScopeEmbeddings,
                                            QianfanEmbeddingsEndpoint)
from langchain_community.llms import QianfanLLMEndpoint, Tongyi
from langchain_community.utilities import (BingSearchAPIWrapper,
                                           GoogleSearchAPIWrapper,
                                           GoogleSerperAPIWrapper)
from langchain_community.vectorstores import VectorStore
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel
from typing import Any

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

GENERAL_QA_TEMPLATE = '''
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use five sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:
'''

class QAInput(BaseModel):
    key_words: str = COMPANY_NAME
    query: str = QUERY
    num_results: int = 10
    min_text_length: int = 50
    search_engine_name: str = 'Bing'
    embedding_provider: str = 'Alibaba'
    llm_provider: str = 'Alibaba'


def qa_over_docs(qaio: QAInput):
    logging.info(f'Getting URLs from {qaio["search_engine_name"]} search...')

    if qaio["search_engine_name"] == 'Google' and qaio["num_results"] > 10:
        logging.error(
            'GoogleSearchAPI only supports up to 10 resutls, try other engines like Bing/GoogleSerper.')
        return

    if qaio["search_engine_name"] == 'Google':
        search_engine = GoogleSearchAPIWrapper(k=qaio["num_results"])
    elif qaio["search_engine_name"] == 'GoogleSerper':
        search_engine = GoogleSerperAPIWrapper(k=qaio["num_results"])
    elif qaio["search_engine_name"] == 'Bing':
        search_engine = BingSearchAPIWrapper(k=qaio["num_results"])
    else:
        logging.error(f'Search engine {qaio["search_engine_name"]} not supported.')
        return

    if qaio["embedding_provider"] == 'Alibaba':
        embedding = DashScopeEmbeddings()
    elif qaio["embedding_provider"] == 'Baidu':
        embedding = QianfanEmbeddingsEndpoint()
    elif qaio["embedding_provider"] == 'OpenAI':
        embedding = OpenAIEmbeddings()
    else:
        logging.error(
            f'Embedding provider {qaio["embedding_provider"]} not supported.')
        return

    # if qaio["vector_db_provider"] == 'FAISS':
    #     vector_db = FAISS
    # elif qaio["vector_db_provider"] == 'Chroma':
    #     vector_db = Chroma
    # else:
    #     logging.error(f'Vector DB {qaio["vector_db_provider"]} not supported.')
    #     return

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
    if qaio["search_engine_name"] == 'GoogleSerper':
        urls = [item['link'] for item in search_results['organic']]
    else:
        urls = [item['link'] for item in search_results]

    urls_filtered = []
    for u in urls:
        inc_flag = True
        for el in URL_EXCLUDE_LIST:
            if re.search(el, u):
                inc_flag = False
                break
        if inc_flag:
            urls_filtered.append(u)

    logging.info('Getting detailed web content from each URL...')
    apify_client = ApifyClient(os.getenv('APIFY_API_TOKEN'))
    actor_call = apify_client.actor('apify/website-content-crawler').call(
        run_input={
            'startUrls': [{'url': u} for u in urls_filtered],
            'crawlerType': 'cheerio',
            'maxCrawlDepth': 0,
            'maxSessionRotations': 0,
            'proxyConfiguration': {'useApifyProxy': True},
        })
    apify_dataset = apify_client.dataset(
        actor_call['defaultDatasetId']).list_items().items
    fetch_recs = []
    for rec in apify_dataset:
        if rec['crawl']['httpStatusCode'] < 300 and len(rec['text']) >= qaio["min_text_length"]:
            fetch_recs.append({'url': rec['url'], 'content': rec['text']})

    docs = []
    for item in fetch_recs:
        doc = Document(page_content=item['content'],
                       metadata={'source': item['url']})
        docs.append(doc)

    logging.info('Splitting documents into small chunks...')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunked_docs = splitter.split_documents(docs)

    logging.info(
        f'Vectorizing documents into FAISS using embedding method provided by {qaio["embedding_provider"]}')

    vector_store = FAISS.from_documents(
        documents=chunked_docs,
        embedding=embedding,
    )
    # vector_db.save_local(local_index_name)

    logging.info('Generating maximal-marginal-relevance retriever...')
    retriever = vector_store.as_retriever(
        search_type='mmr', search_kwargs={'k': 3, 'fetch_k': 5})

    logging.info(f'Documents QA using LLM provied by {qaio["llm_provider"]}...')

    rag_prompt = PromptTemplate.from_template(GENERAL_QA_TEMPLATE)
    rag_chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    try:
        answer = rag_chain.invoke(qaio["query"])
    except ValueError:  # Occurs when there is no relevant information
        answer = 'I don\'t know'

    qa = {
        'query': qaio["query"],
        'answer': answer,
        'urls': urls_filtered,
        'contents': fetch_recs,
    }

    return qa


if __name__ == '__main__':
    from fastapi import FastAPI
    import uvicorn
    from langchain.schema.runnable import RunnableLambda

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
    # contents = fetch_content(f'{COMPANY_NAME} negative news')
    # print(qa_over_docs(contents, query=QUERY))

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
