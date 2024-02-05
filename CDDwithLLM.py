import logging
import os
import sys
if sys.platform == "linux":
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import json
import uuid
import pprint
from typing import Optional, List, Dict

import redis
import chromadb
from chromadb.config import Settings

from apify_client import ApifyClient

# from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.chains import create_tagging_chain, load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain_community.utilities.bing_search import BingSearchAPIWrapper
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.callbacks import get_openai_callback
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class CDDwithLLM:
    def template_by_lang(self):
        if self.lang == "zh-CN":
            self.default_search_suffix = "负面新闻"
            self.qa_template = """利用下面的上下文回答后面的问题。
如果不知道答案，就说不知道，不要试图编造答案。
答案尽量简洁。

{context}

问题：{question}

有用的答案："""
            self.qa_default_query = f"""{self.company_name}有哪些负面新闻？总结不超过3条主要的，每条独立一行列出，并给出信息出处的URL。"""
            self.qa_no_info = "没有足够的信息回答该问题"
            self.summary_map = """请写出以下内容的简明摘要：

"{text}"

简明摘要："""
            self.summary_combine = """简要概括三个反引号之间文字，以要点形式返回你的回复，每个要点独立一行，涵盖文本的要点：

```{text}```

要点："""
        elif self.lang == "zh-HK" or self.lang == "zh-TW":
            self.default_search_suffix = "負面新聞"
            self.qa_template = """利用下面的上下文回答後面的問題。
如果不知道答案，就說不知道，不要試圖編造答案。
答案儘量簡潔。

{context}

問題：{question}

有用的答案："""
            self.qa_default_query = f"""{self.company_name}有哪些負面新聞？總結不超過3條主要的，每條獨立一行列出，並給出資訊出處的URL。"""
            self.qa_no_info = "沒有足夠的資訊回答該問題"
            self.summary_map = """請寫出以下內容的簡明摘要：

"{text}"

簡明摘要："""
            self.summary_combine = """簡要概括三個反引號之間文字，以要點形式返回你的回復，每個要點獨立一行，涵蓋文本的要點：

```{text}```

要點："""
        else:
            self.default_search_suffix = "negative news"
            self.qa_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:"""
            self.qa_default_query = f"""What is the negative news about {self.company_name}? Summarize no more than 3 major ones, list each on a separate line, and give the URL where the information came from."""
            self.qa_no_info = "No enough information to answer"
            self.summary_map = """Write a concise summary of the following:

"{text}"

CONCISE SUMMARY:"""
            self.summary_combine = """Write a concise summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.

```{text}```

BULLET POINT SUMMARY:"""
        self.default_tagging_schema = {
            "properties": {
                "suspected of financial crimes": {
                    "type": "string",
                    "enum": ["suspect", "unsuspect"],
                    "description": f"determine whether the company {self.company_name} is suspected of financial crimes, This refers specifically to financial crimes and not to other types of crime",
                },
                "types of suspected financial crimes": {
                    "type": "string",
                    "enum": [
                        "Not Suspected", "Financial Fraud", "Counterfeiting Currency/Financial Instruments", "Illegal Absorption of Public Deposits", "Illegal Granting of Loans", "Money Laundering", "Insider Trading", "Manipulation of Securities Markets"],
                    "description": f"Describes the specific type of financial crime {self.company_name} is suspected of committing, or returns the type 'Not Suspected' if not suspected",
                },
                "probability": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": f"describes the probability that the company {self.company_name} is suspected of financial crimes, This refers specifically to financial crimes and not to other types of crime",
                },
            },
            "required": ["suspected of financial crimes", "types of suspected financial crimes", "probability"],
        }

    def __init__(
        self,
        company_name: str,
        lang: str = "en-US",
    ) -> None:
        self.company_name = company_name
        self.lang = lang
        self.search_results = []
        self.web_contents = []
        self.encoded_name = uuid.uuid3(uuid.NAMESPACE_DNS, company_name).hex
        # self.redis_client = redis.Redis(host=os.getenv("REDIS_HOST"),
        #                                 port=os.getenv("REDIS_PORT"),
        #                                 username=os.getenv("REDIS_USER"),
        #                                 password=os.getenv("REDIS_PASSWORD"),
        #                                 )
        self.template_by_lang()
        


    def web_search(self,
                   search_suffix: Optional[str] = None,
                   search_engine: str = "Bing",
                   num_results: int = 10) -> None:
        if search_engine == "Google":
            if self.lang == "zh-CN":
                langchain_se = GoogleSerperAPIWrapper(
                    k=num_results, gl="cn", hl="zh-cn")
            elif self.lang == "zh-HK":
                langchain_se = GoogleSerperAPIWrapper(
                    k=num_results, gl="hk", hl="zh-tw")
            elif self.lang == "zh-TW":
                langchain_se = GoogleSerperAPIWrapper(
                    k=num_results, gl="tw", hl="zh-tw")
            else:  # en-US
                langchain_se = GoogleSerperAPIWrapper(k=num_results)
        elif search_engine == "Bing":
            langchain_se = BingSearchAPIWrapper(
                k=num_results, search_kwargs={"mkt": self.lang})
        else:
            raise ValueError(
                f"Search engine {search_engine} is not supported.")

        logger.info(f"Getting urls from {search_engine} search...")
        search_suffix = search_suffix or self.default_search_suffix

        raw_search_results = langchain_se.results(
            self.company_name + " " + search_suffix, num_results=num_results)
        if search_engine == "Google":
            raw_search_results = raw_search_results["organic"]
        self.search_results = [{"title": item["title"], "snippet": item["snippet"],
                                "url": item["link"]} for item in raw_search_results]

    def search_to_file(self,
                       path: Optional[str] = "./store/",
                       base_name: Optional[str] = None) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        base_name = base_name or self.encoded_name + "_search_results.json"
        logger.info(f"Saving search results to {base_name}...")
        with open(os.path.join(path, base_name), 'w') as f:
            json.dump(self.search_results, f)

    def search_from_file(self,
                         path: Optional[str] = "./store/",
                         base_name: Optional[str] = None) -> None:
        base_name = base_name or self.encoded_name + "_search_results.json"
        logger.info(f"Loading search results from {base_name}...")
        with open(os.path.join(path, base_name), 'r') as f:
            self.search_results = json.load(f)

    def contents_from_crawler(self,
                              min_text_length: int = 100) -> None:
        logger.info("Getting detailed web contents from each url...")
        apify_client = ApifyClient(os.getenv("APIFY_API_TOKEN"))
        actor_call = apify_client.actor("apify/website-content-crawler").call(
            run_input={
                "startUrls": [{"url": item["url"]} for item in self.search_results],
                "crawlerType": "cheerio",
                "maxCrawlDepth": 0,
                "maxSessionRotations": 0,
                "maxRequestRetries": 0,
                "proxyConfiguration": {"useApifyProxy": True},
            }
        )
        apify_dataset = (apify_client.dataset(
            actor_call["defaultDatasetId"]).list_items().items)
        self.web_contents = [{"url": item['url'], "text": item['text']} for item in apify_dataset if item["crawl"]
                             ["httpStatusCode"] < 300 and len(item["text"]) >= min_text_length]

    def contents_from_file(self,
                           path: Optional[str] = "./store/",
                           base_name: Optional[str] = None) -> None:
        base_name = base_name or self.encoded_name + "_web_contents.json"
        logger.info(f"Loading web contents from {base_name}...")
        with open(os.path.join(path, base_name), 'r') as f:
            self.web_contents = json.load(f)

    def contents_to_file(self,
                         path: Optional[str] = "./store/",
                         base_name: Optional[str] = None) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        base_name = base_name or self.encoded_name + "_web_contents.json"
        logger.info(f"Saving web contents to {base_name}...")
        with open(os.path.join(path, base_name), 'w') as f:
            json.dump(self.web_contents, f)

    def contents_from_redis(self, field_name: Optional[str] = None) -> None:
        # field_name = field_name or "cdd_with_llm:web_contents:" + \
        #     self.encoded_name + ":" + self.lang
        field_name = field_name or "cdd_with_llm:web_contents:" + \
            self.encoded_name
        logger.info(
            f"Loading web contents from redis with field name {self.encoded_name}...")
        redis_client = redis.Redis.from_url(os.getenv("REDIS_URI"))
        redis_data = redis_client.hgetall(field_name)
        redis_client.close()
        for key in redis_data:
            self.web_contents.append(
                {"url": key.decode("UTF-8"), "text": redis_data[key].decode("UTF-8")})

    def contents_to_redis(self, field_name: Optional[str] = None) -> None:
        # field_name = field_name or "cdd_with_llm:web_contents:" + \
        #     self.encoded_name + ":" + self.lang
        field_name = field_name or "cdd_with_llm:web_contents:" + \
            self.encoded_name
        logger.info(
            f"Saving web contents to redis with field name {self.encoded_name}...")
        redis_client = redis.Redis.from_url(os.getenv("REDIS_URI"))
        for item in self.web_contents:
            redis_client.hset(field_name, item["url"], item["text"])
        redis_client.close()

    def fca_tagging(self,
                    tagging_schema: Optional[Dict] = None,
                    llm_provider: str = "AzureOpenAI") -> List[Dict]:
        if llm_provider == "Alibaba":
            llm = Tongyi(model_name="qwen-max", temperature=0)
        elif llm_provider == "OpenAI":
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        elif llm_provider == "AzureOpenAI":
            llm = AzureChatOpenAI(azure_deployment=os.getenv(
                "AZURE_OPENAI_LLM_DEPLOY"), temperature=0)
        else:
            raise ValueError(f"LLM provider {llm_provider} is not supported.")

        logger.info(f"Documents tagging with LLM provider {llm_provider}...")
        tagging_schema = tagging_schema or self.default_tagging_schema
        tagging_chain = create_tagging_chain(tagging_schema, llm)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, chunk_overlap=0)
        fca_tags = []

        with get_openai_callback() as cb:
            for doc in [item["text"] for item in self.web_contents]:
                chunked_docs = splitter.split_text(doc)
                # Use only the first chunk to save llm calls
                fca_tags.append(tagging_chain.invoke(chunked_docs[0]))
            logger.info(f"{cb.total_tokens} tokens used")

        return [item['text'] for item in fca_tags]
        # return fca_tags

    def summarization(self,
                    #   with_his_data: bool = False,
                      llm_provider: str = "AzureOpenAI") -> str:
        if llm_provider == "Alibaba":
            llm = Tongyi(model_name="qwen-max", temperature=0)
        elif llm_provider == "OpenAI":
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        elif llm_provider == "AzureOpenAI":
            llm = AzureChatOpenAI(azure_deployment=os.getenv(
                "AZURE_OPENAI_LLM_DEPLOY"), temperature=0)
        else:
            raise ValueError(f"LLM provider {llm_provider} is not supported.")

        logger.info(
            f"Documents summarrization with LLM provider {llm_provider}...")
        langchain_docs = []
        for item in self.web_contents:
            langchain_docs.append(
                Document(page_content=item["text"], metadata={"source": item["url"]}))
        # if with_his_data:
        #     field_name = "cdd_with_llm:web_contents:" + self.encoded_name
        #     historical_data = self.redis_client.hgetall(field_name)
        #     self.redis_client.close()
        #     for key in historical_data:
        #         langchain_docs.append(Document(page_content=historical_data[key].decode(
        #             "UTF-8"), metadata={"source": key.decode("UTF-8")}))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, chunk_overlap=0)
        chunked_docs = splitter.split_documents(langchain_docs)

        # map_prompt = hub.pull("rlm/map-prompt")
        # reduce_prompt = hub.pull("rlm/reduce-prompt")
        map_prompt_template = PromptTemplate(
            template=self.summary_map, input_variables=["text"])
        combine_prompt_template = PromptTemplate(
            template=self.summary_combine, input_variables=["text"])
        summarize_chain = load_summarize_chain(llm,
                                               chain_type="map_reduce",
                                               map_prompt=map_prompt_template,
                                               combine_prompt=combine_prompt_template,
                                               )
        try:
            with get_openai_callback() as cb:
                summary = summarize_chain.invoke(chunked_docs)['output_text']
                logger.info(f"{cb.total_tokens} tokens used")
            return summary
        except ValueError:
            return "I can\'t make a summary"

    def qa(self,
           query: Optional[str] = None,
           with_his_data: bool = False,
           embedding_provider: str = "AzureOpenAI",
           llm_provider: str = "AzureOpenAI") -> Dict[str, str]:
        if embedding_provider == "Alibaba":
            embedding = DashScopeEmbeddings(
                dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))
        elif embedding_provider == "OpenAI":
            embedding = OpenAIEmbeddings()
        elif embedding_provider == "AzureOpenAI":
            embedding = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AZURE_OPENAI_EMB_DEPLOY"))
        else:
            raise ValueError(
                f"Embedding provider {embedding_provider} is not supported.")

        if llm_provider == "Alibaba":
            llm = Tongyi(model_name="qwen-max", temperature=0)
        elif llm_provider == "OpenAI":
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        elif llm_provider == "AzureOpenAI":
            llm = AzureChatOpenAI(azure_deployment=os.getenv(
                "AZURE_OPENAI_LLM_DEPLOY"), temperature=0)
        else:
            raise ValueError(f"LLM provider {llm_provider} is not supported.")

        query = query or self.qa_default_query
        langchain_docs = []
        for item in self.web_contents:
            langchain_docs.append(
                Document(page_content=item["text"], metadata={"source": item["url"]}))
        if with_his_data:
            field_name = "cdd_with_llm:web_contents:" + self.encoded_name
            redis_client = redis.Redis.from_url(os.getenv("REDIS_URI"))
            historical_data = redis_client.hgetall(field_name)
            redis_client.close()
            for key in historical_data:
                langchain_docs.append(Document(page_content=historical_data[key].decode(
                    "UTF-8"), metadata={"source": key.decode("UTF-8")}))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200)
        chunked_docs = splitter.split_documents(langchain_docs)
        # print(f"chunked docs count: {len(chunked_docs)}")

        logger.info(
            f"Documents embedding with provider {embedding_provider}...")

        chroma_client = chromadb.EphemeralClient(
            Settings(anonymized_telemetry=False, allow_reset=True))
        chroma_client.reset()
        langchain_chroma = Chroma(client=chroma_client,
                                  embedding_function=embedding)
        langchain_chroma.add_documents(chunked_docs)
        # print(f"chroma collection count: {langchain_chroma._collection.count()}")

        mmr_retriever = langchain_chroma.as_retriever(search_type="mmr",
                                                      search_kwargs={"k": min(3, len(chunked_docs)),
                                                                     "fetch_k": min(5, len(chunked_docs))})

        logger.info(f"Documents QA with LLM provider {llm_provider}...")
        # rag_prompt = hub.pull("rlm/rag-prompt")
        rag_prompt = PromptTemplate.from_template(self.qa_template)
        rag_chain = (
            {"context": mmr_retriever, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        try:
            with get_openai_callback() as cb:
                answer = rag_chain.invoke(query)
                logger.info(f"{cb.total_tokens} tokens used")
                return {"query": query, "answer": answer}

        except ValueError:  # Occurs when retriever returns nothing
            return {"query": query, "answer": self.no_info}


if __name__ == "__main__":
    # cdd = CDDwithLLM("金融壹账通", lang="zh-CN")
    # cdd = CDDwithLLM("红岭创投", lang="zh-CN")
    # cdd = CDDwithLLM("鸿博股份", lang="zh-CN")
    # cdd = CDDwithLLM("Theranos", lang="en-US")
    # cdd = CDDwithLLM("BridgeWater", lang="en-US")
    cdd = CDDwithLLM("SAS Institute", lang="en-US")
    cdd.web_search(num_results=5, search_engine="Bing")
    cdd.contents_from_crawler()
    cdd.contents_to_redis()

    cdd.contents_from_redis()
    tags = cdd.fca_tagging()
    pprint.pprint(tags)
    summary = cdd.summarization()
    pprint.pprint(summary)
    qa = cdd.qa()
    pprint.pprint(qa)
