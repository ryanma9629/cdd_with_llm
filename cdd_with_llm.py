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

from langchain import hub
from langchain.chains import create_tagging_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
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
            'maxRequestRetries': 0,
            # 'readableTextCharThreshold': min_text_length,
            'proxyConfiguration': {'useApifyProxy': True},
        })
    apify_dataset = apify_client.dataset(
        actor_call['defaultDatasetId']).list_items().items

    web_content = [item for item in apify_dataset if item['crawl']
                   ['httpStatusCode'] < 300 and len(item['text']) >= min_text_length]

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


def template_by_lang(company_name: str,
                     lang: str):
    if lang == 'zh-CN':
        query = f'''{company_name}有哪些负面新闻？总结不超过3条主要的，每条独立一行列出，并给出信息出处的URL。
'''
        no_info = '没有足够的信息回答该问题'
    elif lang == 'zh-HK' or lang == 'zh-TW':
        query = f'''{company_name}有哪些負面新聞？總結不超過3條主要的，每條獨立一行列出，並給出資訊出處的URL。
'''
        no_info = '沒有足夠的資訊回答該問題'
    else:
        query = f'''What is the negative news about {company_name}? 
Summarize no more than 3 major ones, list each on a separate line, and give the URL where the information came from.
'''
        no_info = 'No enough information to answer this.'

    return {'query': query, 'no_info': no_info}


def qa_over_docs(
        company_name: str,
        web_content: List[Dict],
        query: Optional[str] = None,
        lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US',
        embedding_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
        llm_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
        with_redis_data: bool = False,
):
    no_info = template_by_lang(company_name, lang)['no_info']
    query = query or template_by_lang(company_name, lang)['query']

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

    logger.info(f'Documents embedding with provider {embedding_provider}...')
    if embedding_provider == 'Alibaba':
        embedding_fc = DashScopeEmbeddings(
            dashscope_api_key=os.getenv('DASHSCOPE_API_KEY'))
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
        search_kwargs={'k': min(3, len(chunked_docs)),
                       'fetch_k': min(5, len(chunked_docs))}
    )

    logger.info(f'Documents QA with provider {llm_provider}...')
    if llm_provider == 'Alibaba':
        llm = Tongyi(model_name='qwen-max', temperature=0)
    elif llm_provider == 'OpenAI':
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    elif llm_provider == 'AzureOpenAI':
        llm = AzureChatOpenAI(azure_deployment=os.getenv(
            'AZURE_OPENAI_LLM_DEPLOY'), temperature=0)
    else:
        logger.error(f'LLM provider {llm_provider} is not supported.')
        return

    rag_prompt = hub.pull("rlm/rag-prompt")
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
    except ValueError:  # Occurs when retriever returns nothing
        return {
            'query': query,
            'answer': no_info
        }


def tagging_over_docs(company_name: str,
                      web_content: List[Dict],
                      lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US',
                      llm_provider: str = 'AzureOpenAI'
                      ):
    if lang == 'en-US':
        schema = {
            'properties': {
                'suspected of financial crimes': {
                    'type': 'string',
                    'enum': ['suspect', 'unsuspect'],
                    'description': f'determine whether the company {company_name} is suspected of financial crimes, This refers specifically to financial crimes and not to other types of crime',
                },
                'types of suspected financial crimes': {
                    'type': 'string',
                    'enum': ['Not Suspected', 'Financial Fraud', 'Counterfeiting Currency/Financial Instruments', 'Illegal Absorption of Public Deposits', 'Money Laundering', 'Insider Trading', 'Manipulation of Securities Markets'],
                    'description': f'Describes the specific type of financial crime {company_name} is suspected of committing, or returns the type "not suspected" if not suspected',
                },
                'probability': {
                    'type': 'string',
                    'enum': ['low', 'medium', 'high'],
                    'description': f'describes the probability that the company {company_name} is suspected of financial crimes, This refers specifically to financial crimes and not to other types of crime',
                },
            },
            'required': ['suspected of financial crimes', 'types of suspected financial crimes', 'probability'],
        }
    elif lang == 'zh-CN':
        schema = {
            'properties': {
                '是否涉嫌金融犯罪': {
                    'type': 'string',
                    'enum': ['涉嫌', '不涉嫌'],
                    'description': f'判断{company_name}这家公司是否涉嫌金融犯罪，这里特指金融犯罪而不是其它的犯罪类型',
                },
                '涉嫌金融犯罪类型': {
                    'type': 'string',
                    'enum': ['不涉嫌', '金融诈骗', '伪造货币/金融票据', '非法吸收公众存款', '洗钱', '内幕交易', '操纵证券市场'],
                    'description': f'描述{company_name}涉嫌的金融犯罪具体类型，如果不涉嫌则返回类型"不涉嫌"',
                },
                '概率': {
                    'type': 'string',
                    'enum': ['低', '中', '高'],
                    'description': f'描述{company_name}涉嫌金融犯罪的概率，这里特指金融犯罪而不是其它的犯罪类型',
                },
            },
            'required': ['是否涉嫌金融犯罪', '涉嫌金融犯罪类型', '概率'],
        }

    if llm_provider == 'Alibaba':
        llm = Tongyi(model_name='qwen-max', temperature=0)
    elif llm_provider == 'OpenAI':
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    elif llm_provider == 'AzureOpenAI':
        llm = AzureChatOpenAI(azure_deployment=os.getenv(
            'AZURE_OPENAI_LLM_DEPLOY'), temperature=0)
    else:
        logger.error(f'LLM provider {llm_provider} is not supported.')
        return

    chain = create_tagging_chain(schema, llm)
    tags = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000,
                                              chunk_overlap=0,
                                              )
    for doc in [item['text'] for item in web_content]:
        chunked_docs = splitter.split_text(doc)
        # Use only the first chunk to save llm calls
        tags.append(chain.invoke(chunked_docs[0]))

    return tags


if __name__ == '__main__':
    # Proj settings
    # company_name = 'Rothenberg Ventures Management Company, LLC.'
    # company_name = '红岭创投'
    # company_name = '东方甄选'
    # company_name = '恒大财富'
    company_name = '鸿博股份'
    # company_name = '平安银行'
    # company_name = 'Theranos'
    # company_name = 'Bridge Water'
    # company_name = 'SAS Institute'
    # company_name = 'Apple Inc.'
    search_engine = 'Bing'
    search_suffix = '负面新闻'
    # search_suffix = 'negative news'
    lang = 'zh-CN'
    # lang = 'en-US'
    llm_provider = 'AzureOpenAI'
    embedding_provider = 'AzureOpenAI'
    num_results = 5

    search_results = web_search(company_name=company_name,
                                search_suffix=search_suffix,
                                search_engine=search_engine,
                                num_results=num_results,
                                lang=lang)

    web_content = fetch_web_content(company_name=company_name,
                                    urls=[item['url']
                                          for item in search_results],
                                    lang=lang,
                                    save_to_redis=False)

    qa = qa_over_docs(company_name=company_name,
                      web_content=web_content,
                      lang=lang,
                      llm_provider=llm_provider,
                      embedding_provider=embedding_provider,
                      with_redis_data=False)
    if qa:
        pprint.pprint(qa, compact=True)

    tags = tagging_over_docs(company_name,
                             web_content=web_content,
                             lang=lang,  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US',
                             llm_provider='AzureOpenAI'
                             )

    if tags:
        pprint.pprint([item['text'] for item in tags])
