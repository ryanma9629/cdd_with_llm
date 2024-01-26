import logging
import os
import sys
import re
import uuid

if os.platform == 'linux':
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.config import Settings

from apify_client import ApifyClient

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


def fetch_urls(key_words: str,
               search_engine_wrapper: str = 'Bing',
               num_results: int = 10,
               exclude_list: list[str] = [
                   r'163\.com',
                   r'ys80007\.com',
                   r'wyzxwk\.com',
                   r'bijie.gov.cn',
                   r'yongxiu.gov.cn']):
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

    search_results = search_engine.results(key_words, num_results=num_results)
    if search_engine_wrapper == 'GoogleSerper':
        urls = [item['link'] for item in search_results['organic']]
    else:
        urls = [item['link'] for item in search_results]

    urls_filtered = []
    for url in urls:
        include_flag = True
        for exclude_pattern in exclude_list:
            if re.search(exclude_pattern, url):
                include_flag = False
                break
        if include_flag:
            urls_filtered.append(url)
    return [{'url': url} for url in urls_filtered]


# urls_filtered = fetch_urls('IBM', num_results=5)


def fetch_web_content(urls_filtered: list[str],
                      crawler_type: str = 'cheerio',
                      min_text_length: int = 50):
    logging.info('Getting detailed web content from each URL...')

    apify_client = ApifyClient(os.getenv('APIFY_API_TOKEN'))
    actor_call = apify_client.actor('apify/website-content-crawler').call(
        run_input={
            'startUrls': urls_filtered,
            'crawlerType': crawler_type,
            'maxCrawlDepth': 0,
            'maxSessionRotations': 0,
            'proxyConfiguration': {'useApifyProxy': True},
        })
    apify_dataset = apify_client.dataset(
        actor_call['defaultDatasetId']).list_items().items

    records = [rec for rec in apify_dataset if rec['crawl']['httpStatusCode'] < 300
               and len(rec['text']) >= min_text_length]
    return records


# records = fetch_web_content(urls_filtered)


def vectorization_store(records: list[dict],
                        key_words: str,
                        chunk_size: int = 1000,
                        chunk_overlap: int = 100,
                        persistent_dir = './chroma'):
    logging.info(f'Vectorizing documents into ChromaDB...')

    docs = []
    for rec in records:
        doc = Document(page_content=rec['text'],
                       metadata={'source': rec['url']})
        docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunked_docs = splitter.split_documents(docs)

    client = chromadb.PersistentClient(
        path=persistent_dir,
        settings=Settings(anonymized_telemetry=False))

    collection_name = uuid.uuid3(uuid.NAMESPACE_DNS, key_words).hex

    langchain_chroma = Chroma(
        collection_name,
        embedding_function=HuggingFaceEmbeddings(),
        client=client
    )

    ids = langchain_chroma.add_documents(chunked_docs)
    logging.info(
        f'{len(ids)} documents were added into collection {collection_name}')

    return collection_name


# collection_name = vectorization_store(records, 'IBM')


def qa_over_docs(collection_name: str,
                 general_qa_template: str,
                 query: str,
                 persistent_dir = './chroma',
                 llm_provider: str = 'Alibaba'):
    client = chromadb.PersistentClient(
        path=persistent_dir,
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

    rag_prompt = PromptTemplate.from_template(general_qa_template)
    rag_chain = (
        {'context': mmr_retriever, 'question': RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    try:
        answer = rag_chain.invoke(query)
        return {
            'query': query,
            'answer': answer
        }
    except ValueError:  # Occurs when there is no relevant information
        return {
            'query': query,
            'answer': 'I don\'t know.'
        }


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    # Proj settings
    # COMPANY_NAME = 'Rothenberg Ventures Management Company, LLC.'
    COMPANY_NAME = '红岭创投'
    # COMPANY_NAME = '恒大财富'
    # COMPANY_NAME = '鸿博股份'
    # COMPANY_NAME = '平安银行'
    # COMPANY_NAME = 'Theranos'
    # COMPANY_NAME = 'BridgeWater Fund'
    # COMPANY_NAME = 'SAS Institute'
    # COMPANY_NAME = 'Apple Inc.'

    N_NEWS = 10
    LANG = 'zh'  # {'zh', 'en'}
    SEARCH_ENGINE = 'Bing'  # {'Bing', 'Google', 'GoogleSerper'}
    LLM_PROVIDER = 'Alibaba'  # {'Alibaba', 'Baidu', 'OpenAI', 'AzureOpenAI'}

    if LANG == 'en':
        SEARCH_SUFFIX = 'negative news'
        QA_TEMPLATE = '''
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use five sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:
'''
        QUERY = f'''
What is the negative news about {COMPANY_NAME}? 
Summarize no more than 3 major ones, and itemizing each one in a seperate line.
'''
    elif LANG == 'zh':
        SEARCH_SUFFIX = '负面新闻'
        QA_TEMPLATE = '''
利用下列信息回答后面的问题。如果你不知道答案就直接回答'不知道'，不要主观编造答案。
最多使用五句话，回答尽量简洁。

{context}

问题：{question}

有价值的回答：
'''
        QUERY = f'''
{COMPANY_NAME}有哪些负面新闻？总结不超过3条主要的，每条独立一行列出。
'''

    urls = fetch_urls(f'{COMPANY_NAME} {SEARCH_SUFFIX}',
                      search_engine_wrapper=SEARCH_ENGINE, num_results=N_NEWS)

    records = fetch_web_content(urls)

    collection_name = vectorization_store(
        records, f'{COMPANY_NAME} {SEARCH_SUFFIX}')

    qa = qa_over_docs(collection_name, QA_TEMPLATE, QUERY)

    print('-' * 80)
    print(qa['query'])
    print('-' * 80)
    print(qa['answer'])
