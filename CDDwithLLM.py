import logging
import os
import sys

if sys.platform == "linux":
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import pprint
from datetime import datetime, timedelta
from operator import itemgetter
from typing import Dict, List, Optional
from dotenv import load_dotenv

import chromadb
import pymongo
from apify_client import ApifyClient
from chromadb.config import Settings
from langchain.chains import create_tagging_chain, load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_transformers import EmbeddingsClusteringFilter
from langchain_community.utilities.bing_search import BingSearchAPIWrapper
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

logger = logging.getLogger()
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

load_dotenv()


class CDDwithLLM:
    def __init__(
        self,
        company_name: str,
        lang: str = "en-US",
    ) -> None:
        self.company_name = company_name
        self.lang = lang
        self.search_results = None
        self.web_contents = None

        if lang == "zh-CN":
            self.default_search_suffix = "负面新闻"
            self.language = "Simplified Chinese"
        elif lang == "zh-HK" or lang == "zh-TW":
            self.default_search_suffix = "負面新聞"
            self.language = "Traditional Chinese"
        elif lang == "ja-JP":
            self.default_search_suffix = "悪い知らせ"
            self.language = "Japanese"
        elif lang == "en-US":
            self.default_search_suffix = "negative news"
            self.language = "English"

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

        logger.info(f"Getting urls from {search_engine} search...")
        search_suffix = search_suffix or self.default_search_suffix

        raw_search_results = langchain_se.results(
            self.company_name + " " + search_suffix, num_results=num_results)
        if search_engine == "Google":
            raw_search_results = raw_search_results["organic"]
        self.search_results = [
            {"url": item["link"], "title": item["title"]} for item in raw_search_results]

    def contents_from_mongo(self,
                            urls: Optional[List] = None,
                            data_within_days: int = 0,
                            collection: str = "web_contents",
                            ) -> List[Dict]:
        logger.info("Loading existing web contents from MongoDB...")
        client = pymongo.MongoClient(os.getenv("MONGO_URI"))
        col = client.cdd_with_llm[collection]

        within_date = datetime.combine(
            datetime.today(), datetime.min.time()) - timedelta(data_within_days)

        if urls:
            if data_within_days:
                cursor = col.find({
                    "company_name": self.company_name,
                    "lang": self.lang,
                    "url": {"$in": urls},
                    "modified_date": {"$gte": within_date},
                }, {"url": 1, "text": 1, "_id": 0})
            else:
                cursor = col.find({
                    "company_name": self.company_name,
                    "lang": self.lang,
                    "url": {"$in": urls},
                }, {"url": 1, "text": 1, "_id": 0})
        else:
            if data_within_days:
                cursor = col.find({
                    "company_name": self.company_name,
                    "lang": self.lang,
                    "modified_date": {"$gte": within_date},
                }, {"url": 1, "text": 1, "_id": 0})
            else:
                cursor = col.find({
                    "company_name": self.company_name,
                    "lang": self.lang,
                }, {"url": 1, "text": 1, "_id": 0})

        web_contents = list(cursor)
        client.close()
        logger.info(
            f"{len(web_contents)} existing web contents is/are loaded from MongoDB.")
        return web_contents

    def contents_to_mongo(self,
                          web_contents: List[Dict],
                          collection: str = "web_contents",
                          ) -> None:
        logger.info("Saving web contents to MongoDB...")
        client = pymongo.MongoClient(os.getenv("MONGO_URI"))
        col = client.cdd_with_llm[collection]

        for item in web_contents:
            col.update_one(
                {"company_name": self.company_name,
                 "lang": self.lang, "url": item["url"]},
                {
                    "$currentDate": {
                        "modified_date": {"$type": "date"}
                    },
                    "$set": {
                        "company_name": self.company_name,
                        "lang": self.lang,
                        "url": item["url"],
                        "text": item["text"]}},
                upsert=True
            )
        client.close()

    def contents_from_crawler(self,
                              min_content_length: int = 100,
                              contents_load: bool = True,
                              contents_save: bool = True,) -> None:
        logger.info(
            f"Grabbing web contents with Apify/website-content-crawler...")

        urls = [item["url"] for item in self.search_results]
        web_contents = []
        if contents_load:
            contents_loaded = self.contents_from_mongo(urls)
            if contents_loaded:
                urls_loaded = [item["url"] for item in contents_loaded]
                urls_tofetch = list(set(urls) - set(urls_loaded))
            else:
                urls_tofetch = urls
        else:
            urls_tofetch = urls

        logger.info(
            f"Grabbing web contents from {len(urls_tofetch)} url(s)...")

        if urls_tofetch:
            apify_client = ApifyClient(os.getenv("APIFY_API_TOKEN"))
            actor_call = apify_client.actor("apify/website-content-crawler").call(
                run_input={
                    "startUrls": [{"url": url} for url in urls_tofetch],
                    "crawlerType": "cheerio",
                    "maxCrawlDepth": 0,
                    "maxSessionRotations": 0,
                    "maxRequestRetries": 0,
                    "proxyConfiguration": {"useApifyProxy": True},
                }
            )
            apify_dataset = (apify_client.dataset(
                actor_call["defaultDatasetId"]).list_items().items)
            web_contents = [{"url": item['url'], "text": item['text']}
                            for item in apify_dataset if item["crawl"]
                            ["httpStatusCode"] == 200 and len(item["text"]) >= min_content_length]
            if contents_save:
                self.contents_to_mongo(web_contents)

        if contents_load:
            web_contents.extend(contents_loaded)

        self.web_contents = web_contents

    def tags_from_mongo(self,
                        urls: List[Dict],
                        strategy: str,
                        chunk_size: int,
                        llm_model: str,
                        data_within_days: int = 0,
                        collection: str = "tags",
                        ) -> List[Dict]:
        logger.info("Loading existing tags from MongoDB...")
        client = pymongo.MongoClient(os.getenv("MONGO_URI"))
        col = client.cdd_with_llm[collection]

        within_date = datetime.combine(
            datetime.today(), datetime.min.time()) - timedelta(data_within_days)

        if data_within_days:
            cursor = col.find({
                "company_name": self.company_name,
                "lang": self.lang,
                "strategy": strategy,
                "chunk_size": chunk_size,
                "llm_model": llm_model,
                "url": {"$in": urls},
                "modified_date": {"$gte": within_date},
            }, {"url": 1, "type": 1, "probability": 1, "_id": 0})
        else:
            cursor = col.find({
                "company_name": self.company_name,
                "lang": self.lang,
                "strategy": strategy,
                "chunk_size": chunk_size,
                "llm_model": llm_model,
                "url": {"$in": urls},
            }, {"url": 1, "type": 1, "probability": 1, "_id": 0})

        tags = list(cursor)
        client.close()
        logger.info(f"{len(tags)} existing tags is/are loaded from MongoDB.")
        return tags

    def tags_to_mongo(self,
                      tags: List[Dict],
                      strategy: str,
                      chunk_size: int,
                      llm_model: str,
                      collection: str = "tags",
                      ) -> None:
        logger.info("Saving tags to MongoDB...")
        client = pymongo.MongoClient(os.getenv("MONGO_URI"))
        col = client.cdd_with_llm[collection]

        for item in tags:
            col.update_one(
                {"company_name": self.company_name,
                 "lang": self.lang,
                 "strategy": strategy,
                 "chunk_size": chunk_size,
                 "llm_model": llm_model,
                 "url": item["url"]},
                {
                    "$currentDate": {
                        "modified_date": {"$type": "date"}
                    },
                    "$set": {
                        "company_name": self.company_name,
                        "lang": self.lang,
                        "strategy": strategy,
                        "chunk_size": chunk_size,
                        "llm_model": llm_model,
                        "url": item["url"],
                        "type": item["type"],
                        "probability": item["probability"]}},
                upsert=True
            )
        client.close()

    def fc_tagging(self,
                   strategy: str = "all",
                   chunk_size: int = 2000,
                   chunk_overlap: int = 100,
                   llm_model: str = "GPT4",
                   tags_load: bool = True,
                   tags_save: bool = True) -> List[Dict]:
        if llm_model == "GPT4":
            llm = AzureChatOpenAI(azure_deployment=os.getenv(
                "AZURE_OPENAI_LLM_DEPLOY_GPT4"), temperature=0)
        elif llm_model == "GPT35":
            llm = AzureChatOpenAI(azure_deployment=os.getenv(
                "AZURE_OPENAI_LLM_DEPLOY_GPT35"), temperature=0)
        elif llm_model == "GPT4-32k":
            llm = AzureChatOpenAI(azure_deployment=os.getenv(
                "AZURE_OPENAI_LLM_DEPLOY_GPT4_32K"), temperature=0)

        logger.info(f"Documents tagging with LLM model {llm_model}...")
        tagging_schema = {
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
                    "description": f"Describes the specific type of financial crime {self.company_name} \
                    is suspected of committing, or returns the type 'Not suspected' if not suspected",
                },
                "probability": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": f"describes the probability that the company {self.company_name} \
                    is suspected of financial crimes, This refers specifically to financial crimes \
                    and not to other types of crime",
                },
            },
            "required": ["types of suspected financial crimes", "probability"],
        }
        tagging_chain = create_tagging_chain(tagging_schema, llm)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        urls = [item["url"] for item in self.search_results]
        if tags_load:
            tags_loaded = self.tags_from_mongo(
                urls, strategy, chunk_size, llm_model)
            if tags_loaded:
                urls_loaded = [item["url"] for item in tags_loaded]
                urls_totagging = list(set(urls) - set(urls_loaded))
            else:
                urls_totagging = urls
        else:
            urls_totagging = urls

        logger.info(
            f"Generating tags for {len(urls_totagging)} document(s)...")
        tags = []
        if urls_totagging:
            contents_totag = []
            for item in self.web_contents:
                if item["url"] in urls_totagging:
                    contents_totag.append(
                        {"url": item["url"], "text": item["text"]})

            with get_openai_callback() as cb:
                for item in contents_totag:
                    url = item["url"]
                    doc = item["text"]
                    chunked_docs = splitter.split_text(doc)
                    tag_unsuspect = {"url": url,
                                     "type": "Not suspected",
                                     "probability": "low"}
                    try:
                        if strategy == "first":
                            tag = tagging_chain.invoke(chunked_docs[0])["text"]
                            if tag:
                                tags.append({"url": url,
                                            "type": tag["type"],
                                             "probability": tag["probability"]})
                        else:  # strategy == "all":
                            p_tag_medium = {}
                            p_tag_high = {}
                            for piece in chunked_docs:
                                p_tag = tagging_chain.invoke(piece)["text"]
                                if p_tag:
                                    if p_tag["probability"] == "medium":
                                        if not p_tag_medium:
                                            p_tag_medium = p_tag
                                    elif p_tag["probability"] == "high":
                                        p_tag_high = p_tag
                                        break
                            if p_tag_high:
                                tags.append({"url": url,
                                            "type": p_tag_high["type"],
                                             "probability": p_tag_high["probability"]})
                            elif p_tag_medium:
                                tags.append({"url": url,
                                            "type": p_tag_medium["type"],
                                             "probability": p_tag_medium["probability"]})
                            else:
                                tags.append(tag_unsuspect)
                    except ValueError:
                        tags.append(tag_unsuspect)

                logger.info(f"{cb.total_tokens} tokens used")

            if tags_save and tags:
                self.tags_to_mongo(tags, strategy, chunk_size, llm_model)
        if tags_load:
            tags.extend(tags_loaded)

        return tags

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

        summary_map = f"""Write a concise summary about {self.company_name} using the following text \
delimited by triple backquotes:""" + """

```{text}```

CONCISE SUMMARY:"""
        summary_combine = f"""Write a concise summary about {self.company_name} using the following \
text delimited by triple backquotes. Return your response in bullet points which covers the key \
points of the text. Make your respopnse in {self.language} with no more than {max_words} words.""" + """

```{text}```

BULLET POINT SUMMARY:"""

        map_prompt = PromptTemplate(
            template=summary_map, input_variables=["company_name", "text"])
        combine_prompt = PromptTemplate(
            template=summary_combine, input_variables=["company_name", "language", "max_words", "text"])
        summarize_chain = load_summarize_chain(llm,
                                               chain_type="map_reduce",
                                               map_prompt=map_prompt,
                                               combine_prompt=combine_prompt,
                                               )
        try:
            with get_openai_callback() as cb:
                summ = summarize_chain.invoke({"input_documents": repr_docs})[
                    'output_text']
                logger.info(f"{cb.total_tokens} tokens used")
            return summ
        except ValueError:
            return "I can\'t make a summary"

    def qa(self,
           query: Optional[str] = None,
           with_his_data: bool = False,
           data_within_days: int = 90,
           chunk_size: int = 2000,
           chunk_overlap: int = 100,
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

        embedding = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMB_DEPLOY"))

        logger.info(f"Documents QA with LLM model {llm_model}...")
        qa_template = f"""Use the following pieces of context to answer the question at the end. \
If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep \
the answer as concise as possible. Make your response in {self.language}.""" + """

{context}

Question: {question}

Helpful Answer:"""
        qa_default_query = f"""What is the negative news about {self.company_name}? Summarize \
no more than 3 major ones, list each on a separate line, and give the URL where the information \
came from. Make your response in {self.language}"""

        query = query or qa_default_query
        langchain_docs = []
        for item in self.web_contents:
            langchain_docs.append(
                Document(page_content=item["text"], metadata={"source": item["url"]}))
        if with_his_data:
            his_docs = self.contents_from_mongo(
                data_within_days=data_within_days)
            for his_doc in his_docs:
                urls_current = [item.metadata["source"]
                                for item in langchain_docs]
                if his_doc["url"] not in urls_current:
                    langchain_docs.append(Document(
                        page_content=his_doc["text"],
                        metadata={"source": his_doc["url"]}
                    ))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_docs = splitter.split_documents(langchain_docs)

        chroma_client = chromadb.EphemeralClient(
            Settings(anonymized_telemetry=False, allow_reset=True))
        chroma_client.reset()
        vectordb = Chroma(client=chroma_client,
                          embedding_function=embedding)
        vectordb.add_documents(chunked_docs)

        mmr_retriever = vectordb.as_retriever(search_type="mmr",
                                              search_kwargs={"k": min(3, len(chunked_docs)),
                                                             "fetch_k": min(5, len(chunked_docs))})

        rag_prompt = PromptTemplate.from_template(qa_template)

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
                    {"question": query,
                     "language": self.language})
                logger.info(f"{cb.total_tokens} tokens used")
                return answer
        except ValueError:
            return "I can\'t make an answer"


if __name__ == "__main__":

    cdd = CDDwithLLM("红岭创投", lang="zh-CN")
    # cdd = CDDwithLLM("鸿博股份", lang="zh-CN")
    # cdd = CDDwithLLM("金融壹账通", lang="zh-CN")
    # cdd = CDDwithLLM("Theranos", lang="en-US")
    # cdd = CDDwithLLM("BridgeWater", lang="en-US")
    # cdd = CDDwithLLM("SAS Institute", lang="en-US")
    # cdd = CDDwithLLM("红岭创投", lang="ja-JP")

    cdd.web_search()
    cdd.contents_from_crawler()

    tags = cdd.fc_tagging()
    pprint.pprint(tags)

    summary = cdd.summary()
    pprint.pprint(summary)

    qa = cdd.qa()
    pprint.pprint(qa)
