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
    search_results = cdd.search_results

    if not os.path.exists('db'):
        os.makedirs('db')
    filename = './db/' + cdd.encoded_name + '_web_search.json'
    with open(filename, 'w') as f:
        json.dump(search_results, f)

    del cdd
    return sas_json_wrapper(search_results)


@app.get('/cdd_with_llm/fetch_web_content')
def fetch_web_contents(company_name: str,
                       lang: str = 'en-US',
                       min_text_length: int = 100,
                       save_to_redis: bool = True,
                       ):
    cdd = CDDwithLLM(company_name, lang)

    filename = './db/' + cdd.encoded_name + '_web_search.json'
    with open(filename, 'r') as f:
        cdd.search_results = json.load(f)

    cdd.contents_from_crawler(min_text_length)
    if save_to_redis:
        cdd.contents_to_redis()

    web_contents = cdd.web_contents

    filename = './db/' + cdd.encoded_name + '_web_contents.json'
    with open(filename, 'w') as f:
        json.dump(web_contents, f)

    del cdd
    return sas_json_wrapper(web_contents)


@app.get('/cdd_with_llm/tagging_over_docs')
async def fca_tagging(company_name: str,
                      lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US',
                      llm_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
                      ):
    cdd = CDDwithLLM(company_name, lang)

    filename = './db/' + cdd.encoded_name + '_web_contents.json'
    with open(filename, 'r') as f:
        cdd.web_contents = json.load(f)

    tags = cdd.fca_tagging(llm_provider=llm_provider)

    filename = './db/' + cdd.encoded_name + '_tags.json'
    with open(filename, 'w') as f:
        json.dump(tags, f)

    del cdd
    return sas_json_wrapper(tags)


@app.get('/cdd_with_llm/sum_over_docs')
async def summarization(company_name: str,
                        lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US',
                        llm_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
                        with_redis_data: bool = False
                        ):

    cdd = CDDwithLLM(company_name, lang)
    filename = './db/' + cdd.encoded_name + '_web_contents.json'
    with open(filename, 'r') as f:
        cdd.web_contents = json.load(f)

    summary = cdd.summarization(
        with_historial_data=with_redis_data, llm_provider=llm_provider)

    filename = './db/' + cdd.encoded_name + '_summary.json'
    with open(filename, 'w') as f:
        json.dump(summary, f)

    del cdd
    return sas_json_wrapper(summary)


@app.get('/cdd_with_llm/qa_over_docs')
async def qa(company_name: str,
             lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US',
             query: Optional[str] = None,
             embedding_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
             llm_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
             with_redis_data: bool = False
             ):

    cdd = CDDwithLLM(company_name, lang)
    filename = './db/' + cdd.encoded_name + '_web_contents.json'
    with open(filename, 'r') as f:
        cdd.web_contents = json.load(f)

    qa = cdd.qa(query=query, with_historial_data=with_redis_data,
                embedding_provider=embedding_provider, llm_provider=llm_provider)

    filename = './db/' + cdd.encoded_name + '_qa.json'
    with open(filename, 'w') as f:
        json.dump(qa, f)

    del cdd
    return sas_json_wrapper(qa)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)
