import logging
import os
import sys
if sys.platform == 'linux':
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import uuid
from typing import Union, List, Dict


import chromadb
from chromadb.config import Settings

from apify_client import ApifyClient

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


def web_search(company_name: str,
               search_suffix: Union[str, None] = None,
               search_engine: str = 'Bing',  # 'Bing', 'Google'
               num_results: int = 10,
               lang: str = 'zh-CN',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US'
               ):
    logging.info(f'Getting URLs from {search_engine} search...')

    if search_engine == 'Google':
        if lang == 'zh-CN':
            langchain_se = GoogleSerperAPIWrapper(
                k=num_results, gl='cn', hl='zh-cn')
        elif lang == 'zh-HK':
            langchain_se = GoogleSerperAPIWrapper(
                k=num_results, gl='hk', hl='zh-tw')
        elif lang == 'zh-TW':
            langchain_se = GoogleSerperAPIWrapper(
                k=num_results, gl='tw', hl='zh-tw')
        else:
            langchain_se = GoogleSerperAPIWrapper(k=num_results)
    elif search_engine == 'Bing':
        langchain_se = BingSearchAPIWrapper(
            k=num_results, search_kwargs={'mkt': lang})
    else:
        logging.error(
            f'Search engine {search_engine} is not supported.')
        return

    if search_suffix is None:
        if lang == 'zh-CN':
            search_suffix = '负面新闻'
        elif lang == 'zh-TW' or lang == 'zh-HK':
            search_suffix = '負面新聞'
        else:
            search_suffix = 'negative news'

    raw_search_results = langchain_se.results(company_name + ' ' + search_suffix,
                                              num_results=num_results)
    if search_engine == 'Google':
        raw_search_results = raw_search_results['organic']

    search_results = [{'title': item['title'], 'snippet': item['snippet'],
                       'url': item['link']} for item in raw_search_results]

    return search_results


# search_results = web_search('Bridge Water', num_results=15)


def fetch_web_content(urls: List[str],
                      min_text_length: int = 100):
    logging.info('Getting detailed web content from each URL...')

    apify_client = ApifyClient(os.getenv('APIFY_API_TOKEN'))
    actor_call = apify_client.actor('apify/website-content-crawler').call(
        run_input={
            'startUrls': [{'url': url} for url in urls],
            'crawlerType': 'cheerio',
            'maxCrawlDepth': 0,
            'maxSessionRotations': 0,
            'maxRequestRetries': 1,
            'readableTextCharThreshold': min_text_length,
            'proxyConfiguration': {'useApifyProxy': True},
        })
    apify_dataset = apify_client.dataset(
        actor_call['defaultDatasetId']).list_items().items

    records = [rec for rec in apify_dataset if rec['crawl']
               ['httpStatusCode'] < 300]
    return records


# records = fetch_web_content([item['url'] for item in search_results])


def doc_store(records: List[dict],
              company_name: str):

    logging.info('Storing fetched documents into Chroma...')
    persistent_client = chromadb.PersistentClient(
        path='./chroma',
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
        # Fake embeddings, only for raw doc storage
        embeddings=[[0]] * len(records)
    )

    return collection_name


# doc_store(records, company_name='Bridge Water')


def qa_over_docs(
        collection_name: str,
        query: str,
        qa_template: str,
        # llm_provider: str = 'Azure OpenAI'
):
    persistent_client = chromadb.PersistentClient(
        path='./chroma',
        settings=Settings(anonymized_telemetry=False)
    )
    # huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    #     api_key=os.getenv('HUGGINGFACE_API_TOKEN'),
    #     model_name='sentence-transformers/all-MiniLM-L6-v2'
    # )

    try:
        collection = persistent_client.get_collection(collection_name)
    except ValueError:
        logging.error(f'Collection {collection_name} does not exist.')

    chroma_docs = collection.get()

    langchain_docs = []
    for i in range(len(chroma_docs['ids'])):
        langchain_doc = Document(
            page_content=chroma_docs['documents'][i], metadata=chroma_docs['metadatas'][i])
        langchain_docs.append(langchain_doc)

    logging.info('Splitting documents into small chunks...')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunked_docs = splitter.split_documents(langchain_docs)

    langchain_chroma = Chroma(
        collection_name='Ephemeral_Collection_for_QA',
        client_settings=Settings(anonymized_telemetry=False),
        # embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # embedding_function=DashScopeEmbeddings()
        embedding_function=AzureOpenAIEmbeddings(
            azure_deployment=os.getenv('AZURE_OPENAI_EMB_DEPLOY'))
    )

    langchain_chroma.add_documents(chunked_docs)

    mmr_retriever = langchain_chroma.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 3, 'fetch_k': 5}
    )

    logging.info(f'Documents QA using LLM provied by Azure OpenAI...')
    # if llm_provider == 'Alibaba':
    #     llm = Tongyi(model_name='qwen-max', temperature=0)
    # elif llm_provider == 'Baidu':
    #     llm = QianfanLLMEndpoint(model='ERNIE-Bot', temperature=0.01)
    # elif llm_provider == 'OpenAI':
    #     llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    # else:
    #     logging.error(f'LLM provider {llm_provider} is not supported.')
    #     return

    # llm = AzureOpenAI(azure_deployment='gpt4')
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
    # COMPANY_NAME = '红岭创投'
    COMPANY_NAME = '东方甄选'
    # COMPANY_NAME = '恒大财富'
    # COMPANY_NAME = '鸿博股份'
    # COMPANY_NAME = '平安银行'
    # COMPANY_NAME = 'Theranos'
    # COMPANY_NAME = 'Bridge Water'
    # COMPANY_NAME = 'SAS Institute'
    # COMPANY_NAME = 'Apple Inc.'

    N_NEWS = 10
    LANG = 'zh'  # {'zh', 'en'}
    SEARCH_ENGINE = 'Bing'  # {'Bing', 'Google', 'GoogleSerper'}
    # LLM_PROVIDER = 'Alibaba'  # {'Alibaba', 'Baidu', 'OpenAI', 'AzureOpenAI'}

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

    search_results = web_search(
        COMPANY_NAME, search_engine=SEARCH_ENGINE, num_results=N_NEWS)

    records = fetch_web_content([item['url'] for item in search_results])

    collection_name = doc_store(records, COMPANY_NAME)

    qa = qa_over_docs(collection_name, QUERY, QA_TEMPLATE)

    print('-' * 80)
    print(qa['query'])
    print('-' * 80)
    print(qa['answer'])
