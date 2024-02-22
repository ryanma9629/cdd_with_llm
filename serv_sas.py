import json
import os
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


def sas_json_wrapper(data: List):
    if not isinstance(data, list):
        data = [data]
    return {
        'items': data,
        'start': 0,
        'limit': len(data),
        'count': len(data)
    }


@app.get('/cdd_with_llm/test')
async def test():
    return sas_json_wrapper(
        [{
            "title": "SAS Intitute",
            "snippet": "When everything is data, discover what matters with SAS Viya",
            "url": "http://www.sas.com"
        }],
    )


@app.get('/cdd_with_llm/web_search')
async def web_search(company_name: str,
                     search_suffix: Optional[str] = None,
                     search_engine: str = 'Bing',  # 'Bing', 'Google'
                     num_results: int = 10,
                     lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US'
                     ):
    cdd = CDDwithLLM(company_name, lang)
    cdd.web_search(search_suffix, search_engine, num_results)
    cdd.search_to_file()
    return sas_json_wrapper(cdd.search_results)


@app.get('/cdd_with_llm/contents_from_crawler')
def contents_from_crawler(company_name: str,
                          lang: str = 'en-US',
                          min_text_length: int = 100,
                          save_to_redis: bool = True,
                          ):
    cdd = CDDwithLLM(company_name, lang)
    cdd.search_from_file()
    cdd.contents_from_crawler(min_text_length)
    if save_to_redis:
        cdd.contents_to_redis()
    cdd.contents_to_file()
    return sas_json_wrapper(cdd.web_contents)


@app.get('/cdd_with_llm/fca_tagging')
async def fca_tagging(company_name: str,
                      lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US',
                      llm_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
                      ):
    cdd = CDDwithLLM(company_name, lang)
    cdd.contents_from_file()
    tags = cdd.fca_tagging(llm_provider=llm_provider)
    return sas_json_wrapper(tags)


@app.get('/cdd_with_llm/summarization')
async def summarization(company_name: str,
                        lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US',
                        llm_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
                        ):

    cdd = CDDwithLLM(company_name, lang)
    cdd.contents_from_file()
    summary = cdd.summarization(llm_provider=llm_provider)
    return sas_json_wrapper(summary)


@app.get('/cdd_with_llm/qa')
async def qa(company_name: str,
             lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US',
             query: Optional[str] = None,
             embedding_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
             llm_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
             with_his_data: bool = False
             ):
    cdd = CDDwithLLM(company_name, lang)
    cdd.contents_from_file()
    qa = cdd.qa(query=query, with_his_data=with_his_data,
                embedding_provider=embedding_provider, llm_provider=llm_provider)
    return sas_json_wrapper(qa)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)
