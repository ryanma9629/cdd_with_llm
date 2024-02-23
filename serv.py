import json
import os
import pandas as pd
from typing import Optional, List, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from CDDwithLLM import CDDwithLLM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get('/cdd_with_llm/web_search')
async def web_search(company_name: str,
                     lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US'
                     search_engine: str = 'Bing',  # 'Bing', 'Google'
                     num_results: int = 5,
                     ):
    cdd = CDDwithLLM(company_name, lang)
    cdd.web_search(search_engine=search_engine, num_results=num_results)
    # cdd.contents_from_crawler(min_text_length=0)
    # cdd.contents_to_mongo(collection="tmp", truncate_before_insert=True)
    cdd.contents_from_mongo(collection="tmp")
    return json.dumps(cdd.search_results)


# @app.get('/cdd_with_llm/contents_from_crawler')
# def contents_from_crawler(company_name: str,
#                           lang: str = 'en-US',
#                           min_text_length: int = 100,
#                           save_to_redis: bool = True,
#                           ):
#     cdd = CDDwithLLM(company_name, lang)
#     cdd.search_from_file()
#     cdd.contents_from_crawler(min_text_length)
#     if save_to_redis:
#         cdd.contents_to_redis()
#     cdd.contents_to_file()
#     return sas_json_wrapper(cdd.web_contents)


@app.get('/cdd_with_llm/fca_tagging')
async def fca_tagging(company_name: str,
                      lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US',
                      llm_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
                      ):
    cdd = CDDwithLLM(company_name, lang)
    cdd.contents_from_mongo(collection="tmp")
    tags = cdd.fca_tagging(llm_provider=llm_provider)
    return json.dumps(tags)


@app.get('/cdd_with_llm/summarization')
async def summarization(company_name: str,
                        lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US',
                        llm_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
                        ):

    cdd = CDDwithLLM(company_name, lang)
    cdd.contents_from_mongo(collection="tmp")
    summary = cdd.summarization(llm_provider=llm_provider)
    return json.dumps(summary)


@app.get('/cdd_with_llm/qa')
async def qa(company_name: str,
             lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US',
             query: Optional[str] = None,
             embedding_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
             llm_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
             ):
    cdd = CDDwithLLM(company_name, lang)
    cdd.contents_from_mongo(collection="tmp")
    answer = cdd.qa(
        query=query, embedding_provider=embedding_provider, llm_provider=llm_provider)
    return json.dumps(answer)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)
