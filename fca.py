import logging
import os
import re

from apify_client import ApifyClient

# from langchain_community.document_loaders.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import (DashScopeEmbeddings,
                                            QianfanEmbeddingsEndpoint)
from langchain_community.llms import QianfanLLMEndpoint, Tongyi
from langchain_community.utilities import (ApifyWrapper, BingSearchAPIWrapper,
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


def fetch_urls(key_words: str,
               search_engine_wrapper: str = 'Bing',
               num_results: int = 10,
               exclude_list: list[str] = [
                   r'163\.com',
                   r'ys80007\.com',
                   r'wyzxwk\.com',
                   r'bijie.gov.cn',
                   r'yongxiu.gov.cn',
               ]):
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


def fetch_web_content(urls_filtered: list[str],
                      crawler_type: str = 'cheerio',
                      min_text_length: int = 50):
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
            and len(rec['text']) >= min_text_length]
    return records


def vectorization_store(records: list[dict], 
                        chunk_size: int = 1000, 
                        chunk_overlap: int = 100, 
                        embedding_provider: str = 'Alibaba'):
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
    return splitter.split_documents(docs)


def vectorization(chunked_docs: list[Document],
                  embedding_provider: str = 'Alibaba',
                  vector_db_provider: str = 'FAISS'):
    logging.info(
        f'Vectorizing documents into ChromaDB using embedding method provided by {embedding_provider}')
    if embedding_provider == 'Alibaba':
        embedding = DashScopeEmbeddings()
    elif embedding_provider == 'Baidu':
        embedding = QianfanEmbeddingsEndpoint()
    elif embedding_provider == 'OpenAI':
        embedding = OpenAIEmbeddings()
    else:
        logging.error(
            f'Embedding provider {embedding_provider} not supported.')
        return

    if vector_db_provider == 'FAISS':
        vector_db = FAISS
    elif vector_db_provider == 'Chroma':
        vector_db = Chroma

    vector_store = vector_db.from_documents(
        documents=chunked_docs,
        embedding=embedding,
    )
    # vector_db.save_local(local_index_name)
    return vector_store


def retriever(vector_db: VectorStore,
              search_type: str = 'mmr',
              search_args: dict[str, float] = {'k': 3, 'fetch_k': 5}):
    logging.info('Generating maximal-marginal-relevance retriever...')
    return vector_db.as_retriever(search_type=search_type, search_kwargs=search_args)


def llm_qa(retriever: VectorStoreRetriever,
           general_qa_template: str,
           query: str,
           llm_provider: str = 'Alibaba'):
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
        {'context': retriever, 'question': RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    try:
        answer = rag_chain.invoke(query)
        return answer
    except ValueError:  # Occurs when there is no relevant information
        return


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    # Proj settings
    # COMPANY_NAME = 'Rothenberg Ventures Management Company, LLC.'
    # COMPANY_NAME = '红岭创投'
    # COMPANY_NAME = '恒大财富'
    COMPANY_NAME = '鸿博股份'
    # COMPANY_NAME = '平安银行'
    # COMPANY_NAME = 'Theranos'
    # COMPANY_NAME = 'BridgeWater Fund'
    # COMPANY_NAME = 'SAS Institute'
    # COMPANY_NAME = 'Apple Inc.'

    N_NEWS = 10
    LANG = 'zh'  # {'zh', 'en'}
    SEARCH_ENGINE = 'Bing'  # {'Bing', 'Google', 'GoogleSerper'}
    LLM_PROVIDER = 'Alibaba'  # {'Alibaba', 'Baidu', 'OpenAI', 'AzureOpenAI'}
    # {'Alibaba', 'Baidu', 'OpenAI', 'AzureOpenAI'}
    EMBEDDING_PROVIDER = 'Alibaba'

    if LANG == 'en':
        SEARCH_SUFFIX = 'negative news'
        TEMPLATE = '''
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use five sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:
'''
        # QUERY = '''
        # Whether this company is suspected of financial crimes?
        # If it is suspected, What is the evidence of the crime in question?
        # '''
        QUERY = f'''
What is the negative news about {COMPANY_NAME}? 
Summarize no more than 3 major ones, and itemizing each one in a seperate line.
'''
        NOINFO = 'No relevant information'
    elif LANG == 'zh':
        SEARCH_SUFFIX = '负面新闻'
        TEMPLATE = '''
利用下列信息回答后面的问题。如果你不知道答案就直接回答'不知道'，不要主观编造答案。
最多使用五句话，回答尽量简洁。

{context}

问题：{question}

有价值的回答：
'''
        # QUERY = '''
        # 这家公司是否涉嫌金融犯罪？如果涉嫌，相关犯罪证据是什么？
        # '''
        QUERY = f'''
{COMPANY_NAME}有哪些负面新闻？总结不超过3条主要的，每条独立一行列出。
        '''
        NOINFO = '没有相关信息'

    urls = fetch_urls(f'{COMPANY_NAME} {SEARCH_SUFFIX}',
                      search_engine_wrapper=SEARCH_ENGINE, num_results=N_NEWS)
    urls_filtered = crawlable_url_filter(urls)
    docs = fetch_web_content(urls_filtered)
    chunked_docs = split_docs(docs)
    vector_db = vectorization(
        chunked_docs, embedding_provider=EMBEDDING_PROVIDER)
    mmr_retriever = retriever(vector_db)
    answer = llm_qa(mmr_retriever, general_qa_template=TEMPLATE,
                    query=QUERY, llm_provider=LLM_PROVIDER)
    print('-' * 80)
    print(QUERY)
    print('-' * 80)
    if answer is None:
        print(NOINFO)
    else:
        print(answer)
