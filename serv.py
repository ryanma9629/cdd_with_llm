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
    df_search_results = pd.DataFrame(cdd.search_results).set_index("url")
    df_web_contents = pd.DataFrame(cdd.web_contents).set_index("url")
    res = pd.concat([df_search_results, df_web_contents], axis=1)
    res.reset_index(inplace=True)
    return res.to_json(orient="records", force_ascii=False)



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


# @app.get('/cdd_with_llm/fca_tagging')
# async def fca_tagging(company_name: str,
#                       lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US',
#                       llm_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
#                       ):
#     cdd = CDDwithLLM(company_name, lang)
#     cdd.contents_from_file()
#     tags = cdd.fca_tagging(llm_provider=llm_provider)
#     return sas_json_wrapper(tags)


# @app.get('/cdd_with_llm/summarization')
# async def summarization(company_name: str,
#                         lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US',
#                         llm_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
#                         ):

#     cdd = CDDwithLLM(company_name, lang)
#     cdd.contents_from_file()
#     summary = cdd.summarization(llm_provider=llm_provider)
#     return sas_json_wrapper(summary)


# @app.get('/cdd_with_llm/qa')
# async def qa(company_name: str,
#              lang: str = 'en-US',  # 'zh-CN', 'zh-HK', 'zh-TW', 'en-US',
#              query: Optional[str] = None,
#              embedding_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
#              llm_provider: str = 'AzureOpenAI',  # 'Alibaba', 'OpenAI', 'AzureOpenAI'
#              with_his_data: bool = False
#              ):
#     cdd = CDDwithLLM(company_name, lang)
#     cdd.contents_from_file()
#     qa = cdd.qa(query=query, with_his_data=with_his_data,
#                 embedding_provider=embedding_provider, llm_provider=llm_provider)
#     return sas_json_wrapper(qa)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=7980)
