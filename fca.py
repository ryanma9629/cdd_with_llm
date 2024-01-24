import logging
import os
import re

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


def search_engine_fetch_urls(kw: str,
                             search_engine_name: str = 'Bing',
                             k: int = 10):
    logging.info(f'Getting URLs from {search_engine_name} search...')

    if search_engine_name == 'Google' and k > 10:
        logging.error(
            'GoogleSearchAPI only supports up to 10 resutls, try other engines like Bing/GoogleSerper.')
        return

    if search_engine_name == 'Google':
        search_engine = GoogleSearchAPIWrapper(k=k)
    elif search_engine_name == 'GoogleSerper':
        search_engine = GoogleSerperAPIWrapper(k=k)
    elif search_engine_name == 'Bing':
        search_engine = BingSearchAPIWrapper(k=k)
    else:
        logging.error(f'Search engine {search_engine_name} not supported.')
        return

    search_results = search_engine.results(kw, num_results=k)
    if search_engine_name == 'GoogleSerper':
        urls = [item['link'] for item in search_results['organic']]
    else:
        urls = [item['link'] for item in search_results]
    return urls


def crawlable_url_filter(urls: list[str],
                         exclude_list: list[str] = [
                             r'163\.com',
                             r'ys80007\.com',
                             r'wyzxwk\.com',
                             r'bijie.gov.cn',
                             r'yongxiu.gov.cn',
]):
    logging.info('Removing URLs that cannot be crawled...')
    urls_filterd = []
    for u in urls:
        inc_flag = True
        for el in exclude_list:
            if re.search(el, u):
                inc_flag = False
                break
        if inc_flag:
            urls_filterd.append(u)
    return urls_filterd


def apify_fetch_web_context(urls_filterd: list[str],
                            crawler_type: str = 'cheerio',
                            min_text_length: int = 50):
    logging.info('Getting detailed web content from each URL...')

    apify = ApifyWrapper()
    loader = apify.call_actor(
        actor_id='apify/website-content-crawler',
        run_input={'startUrls': [{'url': u} for u in urls_filterd],
                   'crawlerType': crawler_type,
                   'maxCrawlDepth': 0,
                   'maxSessionRotations': 0,
                   'proxyConfiguration': {'useApifyProxy': True},
                   },
        dataset_mapping_function=lambda item: Document(
            page_content=item['text'] or '', metadata={'source': item['url']}
        ),
    )
    docs = loader.load()
    docs_filtered = [d for d in docs if len(d.page_content) >= min_text_length]
    return docs_filtered


def split_docs(docs: list[Document],
               chunk_size: int = 1000,
               chunk_overlap: int = 100):
    logging.info('Splitting documents into small chunks...')

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def vectorization(chunked_docs: list[Document],
                  embedding_provider: str = 'Alibaba',
                  vector_db_provider: str = 'FAISS'):
    logging.info(
        f'Vectorizing documents into FAISS using embedding method provided by {embedding_provider}')
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

    def format_docs(docs: list[Document]):
        return '\n\n'.join(doc.page_content for doc in docs)

    rag_prompt = PromptTemplate.from_template(general_qa_template)
    rag_chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
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

    urls = search_engine_fetch_urls(f'{COMPANY_NAME} {SEARCH_SUFFIX}',
                                    search_engine_name=SEARCH_ENGINE, k=N_NEWS)
    urls_filterd = crawlable_url_filter(urls)
    docs = apify_fetch_web_context(urls_filterd)
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
