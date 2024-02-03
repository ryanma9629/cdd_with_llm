import logging
import os
import sys

if sys.platform == "linux":
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
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
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class CDDwithLLM:
    def template_by_lang(self):
        if self.lang == "zh-CN":
            self.default_search_suffix = "负面新闻"
            self.default_query = f"""{self.company_name}有哪些负面新闻？总结不超过3条主要的，每条独立一行列出，并给出信息出处的URL。"""
            self.qa_no_info = "没有足够的信息回答该问题"
        elif self.lang == "zh-HK" or self.lang == "zh-TW":
            self.default_search_suffix = "負面新聞"
            self.default_query = f"""{self.company_name}有哪些負面新聞？總結不超過3條主要的，每條獨立一行列出，並給出資訊出處的URL。"""
            self.qa_no_info = "沒有足夠的資訊回答該問題"
        else:
            self.default_search_suffix = "negative news"
            self.default_query = f"""What is the negative news about {self.company_name}? Summarize no more than 3 major ones, list each on a separate line, and give the URL where the information came from."""
            self.qa_no_info = "No enough information to answer"
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
                        "Not Suspected",
                        "Financial Fraud",
                        "Counterfeiting Currency/Financial Instruments",
                        "Illegal Absorption of Public Deposits",
                        "Illegal Granting of Loans",
                        "Money Laundering",
                        "Insider Trading",
                        "Manipulation of Securities Markets",
                    ],
                    "description": f"Describes the specific type of financial crime {self.company_name} is suspected of committing, or returns the type 'Not Suspected' if not suspected",
                },
                "probability": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": f"describes the probability that the company {self.company_name} is suspected of financial crimes, This refers specifically to financial crimes and not to other types of crime",
                },
            },
            "required": [
                "suspected of financial crimes",
                "types of suspected financial crimes",
                "probability",
            ],
        }

    def __init__(
        self,
        company_name: str,
        lang: str = "en-US",
        embedding_provider: str = "AzureOpenAI",
        llm_provider: str = "AzureOpenAI",
    ) -> None:
        self.company_name = company_name
        self.lang = lang
        self.search_results = []
        self.web_contents = []
        self.encoded_name = uuid.uuid3(uuid.NAMESPACE_DNS, company_name).hex
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=os.getenv("REDIS_PORT"),
            username=os.getenv("REDIS_USER"),
            password=os.getenv("REDIS_PASSWORD"),
        )
        self.template_by_lang()

        self.embedding_provider = embedding_provider
        if embedding_provider == "Alibaba":
            self.embedding = DashScopeEmbeddings(
                dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
            )
        elif embedding_provider == "OpenAI":
            self.embedding = OpenAIEmbeddings()
        elif embedding_provider == "AzureOpenAI":
            self.embedding = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AZURE_OPENAI_EMB_DEPLOY")
            )
        else:
            raise ValueError(
                f"Embedding provider {embedding_provider} is not supported."
            )

        self.llm_provider = llm_provider
        if llm_provider == "Alibaba":
            self.llm = Tongyi(model_name="qwen-max", temperature=0)
        elif llm_provider == "OpenAI":
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        elif llm_provider == "AzureOpenAI":
            self.llm = AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_LLM_DEPLOY"), temperature=0
            )
        else:
            raise ValueError(f"LLM provider {llm_provider} is not supported.")

    # def __del__(self):
    #     self.redis_client.close()

    def web_search(
        self,
        search_suffix: Optional[str] = None,
        search_engine: str = "Bing",
        num_results: int = 10,
    ):
        if search_engine == "Google":
            if self.lang == "zh-CN":
                langchain_se = GoogleSerperAPIWrapper(
                    k=num_results, gl="cn", hl="zh-cn"
                )
            elif self.lang == "zh-HK":
                langchain_se = GoogleSerperAPIWrapper(
                    k=num_results, gl="hk", hl="zh-tw"
                )
            elif self.lang == "zh-TW":
                langchain_se = GoogleSerperAPIWrapper(
                    k=num_results, gl="tw", hl="zh-tw"
                )
            else:  # en-US
                langchain_se = GoogleSerperAPIWrapper(k=num_results)
        elif search_engine == "Bing":
            langchain_se = BingSearchAPIWrapper(
                k=num_results, search_kwargs={"mkt": self.lang}
            )
        else:
            raise ValueError(f"Search engine {search_engine} is not supported.")

        search_suffix = search_suffix or self.default_search_suffix

        raw_search_results = langchain_se.results(
            self.company_name + " " + search_suffix, num_results=num_results
        )
        if search_engine == "Google":
            raw_search_results = raw_search_results["organic"]
        self.search_results = [
            {"title": item["title"], "snippet": item["snippet"], "url": item["link"]}
            for item in raw_search_results
        ]

    def fetch_web_content(self, min_text_length: int = 100, save_to_redis: bool = True):
        logger.info("Getting detailed web content from each url...")
        apify_client = ApifyClient(os.getenv("APIFY_API_TOKEN"))
        actor_call = apify_client.actor("apify/website-content-crawler").call(
            run_input={
                "startUrls": [{"url": item["url"]} for item in self.search_results],
                "crawlerType": "cheerio",
                "maxCrawlDepth": 0,
                "maxSessionRotations": 0,
                "maxRequestRetries": 0,
                # "readableTextCharThreshold": min_text_length,
                "proxyConfiguration": {"useApifyProxy": True},
            }
        )
        apify_dataset = (
            apify_client.dataset(actor_call["defaultDatasetId"]).list_items().items
        )
        web_content = [
            item
            for item in apify_dataset
            if item["crawl"]["httpStatusCode"] < 300
            and len(item["text"]) >= min_text_length
        ]
        if save_to_redis:
            logger.info("Save fetched web contents to redis...")
            hname = "cdd_with_llm:web_content:" + self.encoded_name + ":" + self.lang
            for item in web_content:
                self.redis_client.hset(hname, item["url"], item["text"])
            self.redis_client.close()

    def fca_tagging(self, tagging_schema: Optional[Dict] = None):
        tagging_schema = tagging_schema or self.default_tagging_schema
        tagging_chain = create_tagging_chain(tagging_schema, self.llm)
        splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
        fca_tags = []
        for doc in [item["text"] for item in self.web_contents]:
            chunked_docs = splitter.split_text(doc)
            # Use only the first chunk to save llm calls
            fca_tags.append(tagging_chain.invoke(chunked_docs[0]))

        return fca_tags

    def qa(self, query: Optional[str] = None, with_historial_data: bool = False):
        query = query or self.default_query
        langchain_docs = []
        for item in self.web_content:
            langchain_docs.append(
                Document(page_content=item["text"], metadata={"source": item["url"]})
            )
        if with_historial_data:
            hname = "cdd_with_llm:web_content:" + self.encoded_name + ":" + self.lang
            historical_data = self.redis_client.hgetall(hname)
            self.redis_client.close()
            for key in historical_data:
                langchain_docs.append(
                    Document(
                        page_content=historical_data[key].decode("UTF-8"),
                        metadata={"source": key.decode("UTF-8")},
                    )
                )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        chunked_docs = splitter.split_documents(langchain_docs)

        logger.info(f"Documents embedding with provider {self.embedding_provider}...")

        langchain_chroma = Chroma(
            collection_name="Ephemeral_Collection_for_QA",
            client_settings=Settings(anonymized_telemetry=False),
            embedding_function=self.embedding,
        )
        langchain_chroma.add_documents(chunked_docs)
        mmr_retriever = langchain_chroma.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": min(3, len(chunked_docs)),
                "fetch_k": min(5, len(chunked_docs)),
            },
        )

        logger.info(f"Documents QA with provider {self.llm_provider}...")
        rag_prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
            {"context": mmr_retriever, "question": RunnablePassthrough()}
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )
        try:
            answer = rag_chain.invoke(query)
            return {"query": query, "answer": answer}
        except ValueError:  # Occurs when retriever returns nothing
            return {"query": query, "answer": self.no_info}
