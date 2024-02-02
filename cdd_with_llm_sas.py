import json
import os
import uuid
from typing import Optional, List, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cdd_with_llm import web_search, fetch_web_content, qa_over_docs, tagging_over_docs


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
async def web_search_sas(company_name: str,
                         search_suffix: Optional[str] = None,
                         search_engine: str = 'Bing',  # 'Bing', 'Google'
                         num_results: int = 10,
                         lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US'
                         ):
    search_results = web_search(
        company_name,
        search_suffix,
        search_engine,
        num_results,
        lang)

    encoded_name = uuid.uuid3(uuid.NAMESPACE_DNS, company_name).hex

    if not os.path.exists('db'):
        os.makedirs('db')
    filename = './db/' + encoded_name + '_web_search.json'
    with open(filename, 'w') as f:
        json.dump(search_results, f)

    return sas_json_wrapper(search_results)


@app.get('/cdd_with_llm/fetch_web_content')
def fetch_web_content_sas(company_name: str,
                          min_text_length: int = 100,
                          lang: str = 'en-US',
                          save_to_redis: bool = True,
                          ):
    encoded_name = uuid.uuid3(uuid.NAMESPACE_DNS, company_name).hex
    filename = './db/' + encoded_name + '_web_search.json'

    with open(filename, 'r') as f:
        search_results = json.load(f)

    web_content = fetch_web_content(company_name,
                                    ([item['url'] for item in search_results]),
                                    min_text_length,
                                    lang,
                                    save_to_redis)

    if not os.path.exists('db'):
        os.makedirs('db')
    filename = './db/' + encoded_name + '_web_content.json'
    with open(filename, 'w') as f:
        json.dump(web_content, f)

    return sas_json_wrapper(web_content)


# class QAInput(BaseModel):
#     company_name: str
#     query: Optional[str] = None
#     lang: str = 'zh-CN'
#     embedding_provider: str = 'AzureOpenAI'
#     llm_provider: str = 'AzureOpenAI'
#     with_redis_data: bool = False


# @app.post('/cdd_with_llm/qa_over_docs')
# async def qa_over_docs_sas(qa_input: QAInput):
#     encoded_name = uuid.uuid3(uuid.NAMESPACE_DNS, qa_input.company_name).hex
#     filename = './db/' + encoded_name + '_web_content.json'
#     with open(filename, 'r') as f:
#         web_content = json.load(f)

#     qa = qa_over_docs(qa_input.company_name, web_content, qa_input.query,
#                       qa_input.lang, qa_input.embedding_provider, qa_input.llm_provider, qa_input.with_redis_data)

#     if not os.path.exists('db'):
#         os.makedirs('db')
#     filename = './db/' + encoded_name + '_qa.json'

#     with open(filename, 'w') as f:
#         json.dump(qa, f)

#     return sas_json_wrapper([qa])

@app.get('/cdd_with_llm/tagging_over_docs')
async def tagging_over_docs_sas(company_name: str,
                                lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US',
                                llm_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
                                ):
    encoded_name = uuid.uuid3(uuid.NAMESPACE_DNS, company_name).hex
    filename = './db/' + encoded_name + '_web_content.json'
    with open(filename, 'r') as f:
        web_content = json.load(f)

    res = tagging_over_docs(company_name, web_content, lang, llm_provider)
    tags = [item['text'] for item in res]

    if not os.path.exists('db'):
        os.makedirs('db')
    filename = './db/' + encoded_name + '_tag.json'

    with open(filename, 'w') as f:
        json.dump(tags, f)

    return sas_json_wrapper(tags)


@app.get('/cdd_with_llm/qa_over_docs')
async def qa_over_docs_sas(company_name: str,
                           query: Optional[str] = None,
                           lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US',
                           # 'Alibaba', 'Baidu', 'HuggingFace', 'OpenAI', 'AzureOpenAI'
                           embedding_provider: str = 'AzureOpenAI',
                           llm_provider: str = 'AzureOpenAI',  # 'Alibaba', 'Baidu', 'OpenAI', 'AzureOpenAI'
                           with_redis_data: bool = False
                           ):
    encoded_name = uuid.uuid3(uuid.NAMESPACE_DNS, company_name).hex
    filename = './db/' + encoded_name + '_web_content.json'
    with open(filename, 'r') as f:
        web_content = json.load(f)

    qa = qa_over_docs(company_name,
                      web_content,
                      query,
                      lang,
                      embedding_provider,
                      llm_provider,
                      with_redis_data)

    if not os.path.exists('db'):
        os.makedirs('db')
    filename = './db/' + encoded_name + '_qa.json'

    with open(filename, 'w') as f:
        json.dump(qa, f)

    return sas_json_wrapper([qa])

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)
