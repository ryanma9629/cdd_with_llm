import json
import os
import pandas as pd
from typing import Optional

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
                     lang: str = 'en-US',
                     search_engine: str = 'Bing', 
                     num_results: int = 5,
                     ):
    cdd = CDDwithLLM(company_name, lang)
    cdd.web_search(search_engine=search_engine, num_results=num_results)
    cdd.contents_from_crawler(min_text_length=0)
    cdd.contents_to_mongo(collection="tmp", truncate_before_insert=True)
    cdd.contents_from_mongo(collection="tmp")
    return cdd.search_results


@app.get('/cdd_with_llm/fca_tagging')
async def fca_tagging(company_name: str,
                      lang: str = 'en-US', 
                      llm_provider: str = 'AzureOpenAI',
                      ):
    cdd = CDDwithLLM(company_name, lang)
    cdd.contents_from_mongo(collection="tmp")
    tags = cdd.fca_tagging(llm_provider=llm_provider)
    return tags


@app.get('/cdd_with_llm/summarization')
async def summarization(company_name: str,
                        lang: str = 'en-US',
                        llm_provider: str = 'AzureOpenAI',
                        ):
    cdd = CDDwithLLM(company_name, lang)
    cdd.contents_from_mongo(collection="tmp")
    summary = cdd.summarization(llm_provider=llm_provider)
    return summary


@app.get('/cdd_with_llm/qa')
async def qa(company_name: str,
             lang: str = 'en-US',
             query: Optional[str] = None,
             embedding_provider: str = 'AzureOpenAI',
             llm_provider: str = 'AzureOpenAI',
             ):
    cdd = CDDwithLLM(company_name, lang)
    cdd.contents_from_mongo(collection="tmp")
    answer = cdd.qa(
        query=query, embedding_provider=embedding_provider, llm_provider=llm_provider)
    return answer


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)
