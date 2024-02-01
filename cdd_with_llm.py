import logging
import os
import sys
if sys.platform == 'linux':
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import uuid
import pprint
from typing import Optional, List, Dict

import redis
from chromadb.config import Settings

from apify_client import ApifyClient

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings, HuggingFaceEmbeddings, QianfanEmbeddingsEndpoint
from langchain_community.llms import QianfanLLMEndpoint, Tongyi
from langchain_community.utilities.bing_search import BingSearchAPIWrapper
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings

logger = logging.getLogger()
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def web_search(company_name: str,
               search_suffix: Optional[str] = None,
               search_engine: str = 'Bing',  # 'Bing', 'Google'
               num_results: int = 10,
               lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US'
               ):
    logger.info(f'Getting urls from {search_engine} search...')

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
        logger.error(
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

# company_name = '红岭创投'
# # company_name = 'Theranos'
# search_results = web_search(company_name, lang='en-US')


def fetch_web_content(company_name: str,
                      urls: List[str],
                      min_text_length: int = 100,
                      lang: str = 'en-US',
                      save_to_redis: bool = True,
                      ):
    logger.info('Getting detailed web content from each url...')

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

    web_content = [rec for rec in apify_dataset if rec['crawl']
                   ['httpStatusCode'] < 300]

    if save_to_redis:
        logger.info('Save fetched web contents to redis...')
        encoded_name = uuid.uuid3(uuid.NAMESPACE_DNS, company_name).hex
        r = redis.Redis(host=os.getenv('REDIS_HOST'),
                        port=os.getenv('REDIS_PORT'),
                        username=os.getenv('REDIS_USER'),
                        password=os.getenv('REDIS_PASSWORD'))
        hname = 'cdd_with_llm:web_content:' + encoded_name + ':' + lang
        for item in web_content:
            r.hset(hname, item['url'], item['text'])

        r.close()

    return web_content


# web_content = fetch_web_content(company_name, [item['url'] for item in search_results])


def template_by_lang(company_name: str,
                     lang: str):
    if lang == 'zh-CN':
        query = f'''{company_name}有哪些负面新闻？总结不超过3条主要的，每条独立一行列出，并给出信息出处的URL。
'''
        qa_template = '''利用下列信息回答后面的问题。如果你不知道答案就直接回答'不知道'，不要主观编造答案。
最多使用5句话，回答尽量简洁。

{context}

问题：{question}

有价值的回答：
'''
        no_info = '没有足够的信息回答该问题'
    elif lang == 'zh-HK' or lang == 'zh-TW':
        query = f'''{company_name}有哪些負面新聞？總結不超過3條主要的，每條獨立一行列出，並給出資訊出處的URL。
'''
        qa_template = '''利用下列資訊回答後面的問題。如果你不知道答案就直接回答'不知道'，不要主觀編造答案。
最多使用5句話，回答儘量簡潔。

{context}

問題：{question}

有價值的回答：
'''
        no_info = '沒有足夠的資訊回答該問題'
    else:
        query = f'''What is the negative news about {company_name}? 
Summarize no more than 3 major ones, list each on a separate line, and give the URL where the information came from.
'''
        qa_template = '''Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use 5 sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:
'''
        no_info = 'No enough information to answer this.'

    return {'query': query, 'qa_template': qa_template, 'no_info': no_info}


def qa_over_docs(
        company_name: str,
        web_content: List[Dict],
        query: Optional[str] = None,
        lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US',
        # 'Alibaba', 'Baidu', 'HuggingFace', 'OpenAI', 'AzureOpenAI'
        embedding_provider: str = 'AzureOpenAI',
        llm_provider: str = 'AzureOpenAI',  # 'Alibaba', 'Baidu', 'OpenAI', 'AzureOpenAI'
        with_redis_data: bool = True,
):
    qa_template = template_by_lang(company_name, lang)['qa_template']
    no_info = template_by_lang(company_name, lang)['no_info']
    if query is None:
        query = template_by_lang(company_name, lang)['query']

    langchain_docs = []
    for item in web_content:
        langchain_doc = Document(
            page_content=item['text'],
            metadata={'source': item['url']}
        )
        langchain_docs.append(langchain_doc)

    if with_redis_data:
        logger.info('Getting historical data from redis...')
        encoded_name = uuid.uuid3(uuid.NAMESPACE_DNS, company_name).hex
        r = redis.Redis(host=os.getenv('REDIS_HOST'),
                        port=os.getenv('REDIS_PORT'),
                        username=os.getenv('REDIS_USER'),
                        password=os.getenv('REDIS_PASSWORD'))
        hname = 'cdd_with_llm:web_content:' + encoded_name + ':' + lang
        redis_data = r.hgetall(hname)
        r.close()

        for key in redis_data:
            langchain_doc = Document(
                page_content=redis_data[key].decode('UTF-8'),
                metadata={'source': key.decode('UTF-8')}
            )
            langchain_docs.append(langchain_doc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                              chunk_overlap=100,
                                              )
    chunked_docs = splitter.split_documents(langchain_docs)

    logger.info(f'Documents embedding with provider {embedding_provider}')
    if embedding_provider == 'Alibaba':
        embedding_fc = DashScopeEmbeddings(
            dashscope_api_key=os.getenv('DASHSCOPE_API_KEY'))
    elif embedding_provider == 'Baidu':
        embedding_fc = QianfanEmbeddingsEndpoint()
    elif embedding_provider == 'HuggingFace':
        embedding_fc = HuggingFaceEmbeddings()
    elif embedding_provider == 'OpenAI':
        embedding_fc = OpenAIEmbeddings()
    elif embedding_provider == 'AzureOpenAI':
        embedding_fc = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv('AZURE_OPENAI_EMB_DEPLOY'))
    else:
        logger.error(
            f'Embedding provider {embedding_provider} is not supported.')
        return

    langchain_chroma = Chroma(
        collection_name='Ephemeral_Collection_for_QA',
        client_settings=Settings(anonymized_telemetry=False),
        embedding_function=embedding_fc
    )

    langchain_chroma.add_documents(chunked_docs)

    mmr_retriever = langchain_chroma.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 3, 'fetch_k': 5}
    )

    logger.info(f'Documents QA with provider {llm_provider}...')
    if llm_provider == 'Alibaba':
        chat_llm = Tongyi(model_name='qwen-max', temperature=0)
    elif llm_provider == 'Baidu':
        chat_llm = QianfanLLMEndpoint(model='ERNIE-Bot', temperature=0.01)
    elif llm_provider == 'OpenAI':
        chat_llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    elif llm_provider == 'AzureOpenAI':
        chat_llm = AzureChatOpenAI(azure_deployment=os.getenv(
            'AZURE_OPENAI_LLM_DEPLOY'), temperature=0)
    else:
        logger.error(f'LLM provider {llm_provider} is not supported.')
        return

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
    except ValueError:  # Occurs when retriever returns nothing
        return {
            'query': query,
            'answer': no_info
        }


# qa = qa_over_docs('Theranos', lang='zh-CN', llm_provider='AzureOpenAI')


if __name__ == '__main__':
    # Proj settings
    # COMPANY_NAME = 'Rothenberg Ventures Management Company, LLC.'
    # COMPANY_NAME = '红岭创投'
    # COMPANY_NAME = '东方甄选'
    # COMPANY_NAME = '恒大财富'
    # COMPANY_NAME = '鸿博股份'
    # COMPANY_NAME = '平安银行'
    # COMPANY_NAME = 'Theranos'
    # COMPANY_NAME = 'Bridge Water'
    COMPANY_NAME = 'SAS Institute'
    # COMPANY_NAME = 'Apple Inc.

    LANG = 'en-US'

    search_results = web_search(COMPANY_NAME, lang=LANG, num_results=1)
    web_content = fetch_web_content(
        COMPANY_NAME, [item['url'] for item in search_results], save_to_redis=True, lang=LANG)
    qa = qa_over_docs(COMPANY_NAME, web_content,
                      lang=LANG, with_redis_data=True)
    if qa:
        pprint.pprint(qa, compact=True)
