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
from datetime import datetime, timedelta
from operator import itemgetter

import pymongo
import chromadb
from chromadb.config import Settings
from apify_client import ApifyClient

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
from langchain_community.document_transformers import EmbeddingsClusteringFilter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings

logger = logging.getLogger()
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class CDDwithLLM:
    def __init__(
        self,
        company_name: str,
        lang: str = "en-US",
    ) -> None:
        self.company_name = company_name
        self.lang = lang
        self.search_results = []
        self.web_contents = []
        # self.encoded_name = uuid.uuid3(uuid.NAMESPACE_DNS, company_name).hex
        # self.template_by_lang()

        if lang == "zh-CN":
            self.default_search_suffix = "负面新闻"
            self.language = "Simplified Chinese"
        elif lang == "zh-HK" or lang == "zh-TW":
            self.default_search_suffix = "負面新聞"
            self.language = "Traditional Chinese"
        elif lang == "ja-JP":
            self.default_search_suffix = "悪い知らせ"
            self.language = "Japanese"
        else:
            self.default_search_suffix = "negative news"
            self.language = "English"

        self.qa_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible. Make your response in {language}.

{context}

Question: {question}

Helpful Answer:"""
        self.qa_default_query = f"""What is the negative news about {self.company_name}? Summarize no more than 3 major ones, list each on a separate line, and give the URL where the information came from. Make your response in {self.language}"""
        self.summary_map = "Write a concise summary about " + self.company_name + """ using the following text delimited by triple backquotes:

```{text}```

CONCISE SUMMARY:"""
        self.summary_combine = "Write a concise summary about " + self.company_name + """ using the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text, with no more than {max_words} words. Make your respopnse in {language}.

```{text}```

BULLET POINT SUMMARY:"""
        self.tagging_schema = {
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [
                        "Not suspected",
                        "Financial Fraud",
                        "Counterfeiting Currency/Financial Instruments",
                        "Illegal Absorption of Public Deposits",
                        "Illegal Granting of Loans",
                        "Money Laundering",
                        "Insider Trading",
                        "Manipulation of Securities Markets"],
                    "description": f"Describes the specific type of financial crime {self.company_name} is suspected of committing, or returns the type 'Not suspected' if not suspected",
                },
                "probability": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": f"describes the probability that the company {self.company_name} is suspected of financial crimes, This refers specifically to financial crimes and not to other types of crime",
                },
            },
            "required": ["types of suspected financial crimes", "probability"],
        }

    def web_search(self,
                   search_suffix: Optional[str] = None,
                   search_engine: str = "Bing",
                   num_results: int = 10,
                   ) -> None:
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
            elif self.lang == "ja-JP":
                langchain_se = GoogleSerperAPIWrapper(
                    k=num_results, gl="jp", hl="ja")
            elif self.lang == "en-US":
                langchain_se = GoogleSerperAPIWrapper(
                    k=num_results, gl="us", hl="en")
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
        self.search_results = [
            {"url": item["link"], "title": item["title"]} for item in raw_search_results]

    def search_to_mongo(self,
                        userid: Optional[str] = None,
                        tmp_collection: str = "tmp_search",
                        ) -> None:
        logging.info("Saving search results to MongoDB...")
        client = pymongo.MongoClient(os.getenv("MONGODB_URI"))
        col = client.cdd_with_llm[tmp_collection]

        col.delete_many({"company_name": self.company_name,
                         "lang": self.lang,
                         "userid": userid})

        for item in self.search_results:
            col.insert_one({
                "company_name": self.company_name,
                "lang": self.lang,
                "userid": userid,
                "url": item["url"],
                "title": item["title"]
            })

        client.close()

    def search_from_mongo(self,
                          userid: Optional[str] = None,
                          tmp_collection: str = "tmp_search",
                          ) -> None:
        logging.info("Loading search results from MongoDB...")
        client = pymongo.MongoClient(os.getenv("MONGODB_URI"))
        col = client.cdd_with_llm[tmp_collection]
        cursor = col.find({
            "company_name": self.company_name,
            "lang": self.lang,
            "userid": userid,
        }, {"url": 1, "title": 1, "_id": 0})
        # self.web_contents = list(cursor.sort({"url": 1}))
        self.search_results = list(cursor)
        client.close()

    def contents_from_mongo(self,
                            userid: Optional[str] = None,
                            tmp_collection: str = "tmp_contents",
                            ) -> None:
        logging.info("Loading web contents from MongoDB...")
        client = pymongo.MongoClient(os.getenv("MONGODB_URI"))
        col = client.cdd_with_llm[tmp_collection]

        cursor = col.find({
            "company_name": self.company_name,
            "lang": self.lang,
            "userid": userid,
        }, {"url": 1, "text": 1, "_id": 0})
        # self.web_contents = list(cursor.sort({"url": 1}))
        self.web_contents = list(cursor)
        client.close()

    def contents_to_mongo(self,
                          userid: Optional[str] = None,
                          tmp_collection: str = "tmp_contents",
                          ) -> None:
        logging.info("Saving web contents to MongoDB...")
        client = pymongo.MongoClient(os.getenv("MONGODB_URI"))
        col = client.cdd_with_llm[tmp_collection]

        col.delete_many({"company_name": self.company_name,
                         "lang": self.lang,
                         "userid": userid})

        for item in self.web_contents:
            col.insert_one({
                "company_name": self.company_name,
                "lang": self.lang,
                "userid": userid,
                "url": item["url"],
                "text": item["text"],
            })

        client.close()

    def contents_from_crawler(self,
                              min_text_length: int = 100,
                              persist_collection = "web_contents",
                              contents_save: bool = True,
                              contents_load: bool = True,) -> None:
        logger.info("Getting detailed web contents from each url...")

        client = pymongo.MongoClient(os.getenv("MONGODB_URI"))

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
                             ["httpStatusCode"] == 200 and len(item["text"]) >= min_text_length]



    def fc_tagging(self,
                   strategy: str = "all",  # "first", "all"
                   chunk_size: int = 2000,
                   chunk_overlap: int = 100,
                   llm_model: str = "GPT4") -> List[Dict]:
        if llm_model == "GPT4":
            llm = AzureChatOpenAI(azure_deployment=os.getenv(
                "AZURE_OPENAI_LLM_DEPLOY_GPT4"), temperature=0)
        elif llm_model == "GPT35":
            llm = AzureChatOpenAI(azure_deployment=os.getenv(
                "AZURE_OPENAI_LLM_DEPLOY_GPT35"), temperature=0)
        elif llm_model == "GPT4-32k":
            llm = AzureChatOpenAI(azure_deployment=os.getenv(
                "AZURE_OPENAI_LLM_DEPLOY_GPT4_32K"), temperature=0)
        else:
            raise ValueError(f"LLM model {llm_model} is not supported.")

        logger.info(f"Documents tagging with LLM model {llm_model}...")
        tagging_chain = create_tagging_chain(self.tagging_schema, llm)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        fca_tags = []

        with get_openai_callback() as cb:
            # for doc in [item["text"] for item in self.web_contents]:
            for item in self.web_contents:
                url = item["url"]
                doc = item["text"]
                chunked_docs = splitter.split_text(doc)
                if strategy == "first":
                    tag = tagging_chain.invoke(chunked_docs[0])["text"]
                    fca_tags.append({"url": url,
                                     "type": tag["type"],
                                     "probability": tag["probability"]})
                else:  # strategy == "all":
                    p_tag_medium = {}
                    p_tag_high = {}
                    for piece in chunked_docs:
                        p_tag = tagging_chain.invoke(piece)["text"]
                        if "probability" in p_tag.keys():
                            if p_tag["probability"] == "medium":
                                if not p_tag_medium:
                                    p_tag_medium = p_tag
                            elif p_tag["probability"] == "high":
                                p_tag_high = p_tag
                                break
                    if p_tag_high:
                        fca_tags.append({"url": url,
                                         "type": p_tag_high["type"],
                                         "probability": p_tag_high["probability"]})
                    elif p_tag_medium:
                        fca_tags.append({"url": url,
                                         "type": p_tag_medium["type"],
                                         "probability": p_tag_medium["probability"]})
                    else:
                        fca_tags.append({"url": url,
                                         "type": "Not suspected",
                                         "probability": "low"})
            logger.info(f"{cb.total_tokens} tokens used")

        return fca_tags

    def summary(self,
                max_words: int = 300,
                chunk_size: int = 2000,
                chunk_overlap: int = 100,
                clus_docs: bool = True,
                num_clus: int = 5,
                llm_model: str = "GPT4") -> str:
        if llm_model == "GPT4":
            llm = AzureChatOpenAI(azure_deployment=os.getenv(
                "AZURE_OPENAI_LLM_DEPLOY_GPT4"), temperature=0)
        elif llm_model == "GPT35":
            llm = AzureChatOpenAI(azure_deployment=os.getenv(
                "AZURE_OPENAI_LLM_DEPLOY_GPT35"), temperature=0)
        elif llm_model == "GPT4-32k":
            llm = AzureChatOpenAI(azure_deployment=os.getenv(
                "AZURE_OPENAI_LLM_DEPLOY_GPT4_32K"), temperature=0)
        else:
            raise ValueError(f"LLM model {llm_model} is not supported.")

        embedding = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMB_DEPLOY"))

        logger.info(
            f"Documents summarization with LLM model {llm_model}...")
        langchain_docs = []
        for item in self.web_contents:
            langchain_docs.append(
                Document(page_content=item["text"], metadata={"source": item["url"]}))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_docs = splitter.split_documents(langchain_docs)

        # clustering docs to save llm calls
        if clus_docs:
            num_clus = min(num_clus, len(chunked_docs))
            repr_docs = EmbeddingsClusteringFilter(embeddings=embedding,
                                                   num_clusters=num_clus).transform_documents(chunked_docs)
        else:
            repr_docs = chunked_docs

        map_prompt = PromptTemplate(
            template=self.summary_map, input_variables=["text"])
        combine_prompt = PromptTemplate(
            template=self.summary_combine, input_variables=["text", "max_words", "language"])
        summarize_chain = load_summarize_chain(llm,
                                               chain_type="map_reduce",
                                               map_prompt=map_prompt,
                                               combine_prompt=combine_prompt,
                                               )
        try:
            with get_openai_callback() as cb:
                summ = summarize_chain.invoke(
                    {"input_documents": repr_docs,
                     "max_words": max_words,
                     "language": self.language})['output_text']
                logger.info(f"{cb.total_tokens} tokens used")
            return summ
        except ValueError:
            return "I can\'t make a summary"

    def qa(self,
           query: Optional[str] = None,
           with_his_data: bool = False,
           data_within_days: int = 90,
           chunk_size: int = 1000,
           chunk_overlap: int = 100,
           #    embedding_model: str = "AzureOpenAI",
           llm_model: str = "GPT4") -> str:

        if llm_model == "GPT4":
            llm = AzureChatOpenAI(azure_deployment=os.getenv(
                "AZURE_OPENAI_LLM_DEPLOY_GPT4"), temperature=0)
        elif llm_model == "GPT35":
            llm = AzureChatOpenAI(azure_deployment=os.getenv(
                "AZURE_OPENAI_LLM_DEPLOY_GPT35"), temperature=0)
        elif llm_model == "GPT4-32k":
            llm = AzureChatOpenAI(azure_deployment=os.getenv(
                "AZURE_OPENAI_LLM_DEPLOY_GPT4_32K"), temperature=0)
        else:
            raise ValueError(f"LLM model {llm_model} is not supported.")

        embedding = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMB_DEPLOY"))

        query = query or self.qa_default_query
        langchain_docs = []
        for item in self.web_contents:
            langchain_docs.append(
                Document(page_content=item["text"], metadata={"source": item["url"]}))
        if with_his_data:
            client = pymongo.MongoClient(os.getenv("MONGODB_URI"))
            collection = client.cdd_with_llm["web_contents"]
            within_date = datetime.combine(
                datetime.today(), datetime.min.time()) - timedelta(data_within_days)
            cursor = collection.find({
                "company_name": self.company_name,
                "lang": self.lang,
                "modified_date": {"$gte": within_date}
            }, {"url": 1, "text": 1, "_id": 0})
            for doc in cursor:
                langchain_docs.append(Document(
                    page_content=doc["text"],
                    metadata={"source": doc["url"]}
                ))
            client.close()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_docs = splitter.split_documents(langchain_docs)

        chroma_client = chromadb.EphemeralClient(
            Settings(anonymized_telemetry=False, allow_reset=True))
        chroma_client.reset()
        langchain_chroma = Chroma(client=chroma_client,
                                  embedding_function=embedding)
        langchain_chroma.add_documents(chunked_docs)

        mmr_retriever = langchain_chroma.as_retriever(search_type="mmr",
                                                      search_kwargs={"k": min(3, len(chunked_docs)),
                                                                     "fetch_k": min(5, len(chunked_docs))})

        logger.info(f"Documents QA with LLM model {llm_model}...")
        rag_prompt = PromptTemplate.from_template(self.qa_template)

        rag_chain = (
            {"context": itemgetter("question") | mmr_retriever,
             "question": itemgetter("question"),
             "language": itemgetter("language"),
             }
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        try:
            with get_openai_callback() as cb:
                answer = rag_chain.invoke(
                    {"question": query, "language": self.language})
                logger.info(f"{cb.total_tokens} tokens used")
                return answer
        except ValueError:  # Occurs when retriever returns nothing
            return "I can\'t make an answer"


if __name__ == "__main__":
    # cdd = CDDwithLLM("金融壹账通", lang="zh-CN")
    cdd = CDDwithLLM("红岭创投", lang="zh-CN")
    cdd = CDDwithLLM("红岭创投", lang="ja-JP")
    # cdd = CDDwithLLM("鸿博股份", lang="zh-CN")
    cdd = CDDwithLLM("Theranos", lang="en-US")
    # cdd = CDDwithLLM("BridgeWater", lang="en-US")
    # cdd = CDDwithLLM("SAS Institute", lang="en-US")
    cdd.web_search(num_results=5, search_engine="Bing")
    cdd.search_to_mongo()
    cdd.search_from_mongo()

    cdd.contents_from_crawler()
    cdd.contents_to_mongo()
    cdd.contents_from_mongo()

    tags = cdd.fc_tagging(strategy="first-sus", chunk_size=2000)
    pprint.pprint(tags)
    summary = cdd.summary()
    pprint.pprint(summary)
    qa = cdd.qa()
    pprint.pprint(qa)
