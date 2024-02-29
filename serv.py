import json
import pandas as pd
from typing import Optional, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from CDDwithLLM import CDDwithLLM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/cdd_with_llm/web_search")
async def web_search(company_name: str,
                     lang: str = "en-US",
                     search_suffix: Optional[str] = None,
                     search_engine: str = "Bing",
                     num_results: int = 5):
    cdd = CDDwithLLM(company_name, lang)
    cdd.web_search(search_suffix, search_engine, num_results)
    cdd.search_to_mongo()
    # cdd.search_from_mongo()
    return pd.DataFrame(cdd.search_results).sort_values(by="url").to_html(table_id="tbl_search_results", render_links=True, index=False)


@app.get("/cdd_with_llm/contents_from_crawler")
async def contents_from_crawler(company_name: str,
                                lang: str = "en-US",
                                min_text_length: int = 0):
    cdd = CDDwithLLM(company_name, lang)
    cdd.search_from_mongo()
    cdd.contents_from_crawler(min_text_length)
    cdd.contents_to_mongo()
    # cdd.search_from_mongo()
    # cdd.contents_from_mongo()
    df_search_results = pd.DataFrame(cdd.search_results)
    df_contents = pd.DataFrame(cdd.web_contents)
    df_merged = pd.merge(df_search_results, df_contents, how="left", on="url")
    df_merged["Content"] = df_merged["text"].notnull()
    df_merged.drop("text", axis=1, inplace=True)
    return df_merged.sort_values(by="url").to_html(table_id="tbl_search_results", render_links=True, index=False)


@app.get("/cdd_with_llm/fca_tagging")
async def fca_tagging(company_name: str,
                      lang: str = "en-US",
                      strategy: str = "first-sus",  # "first", "first-sus"
                      chunk_size: int = 2000,
                      chunk_overlap: int = 100,
                      llm_provider: str = "AzureOpenAI",
                      ):
    cdd = CDDwithLLM(company_name, lang)
    cdd.search_from_mongo()
    cdd.contents_from_mongo()
    tags = cdd.fca_tagging(strategy, chunk_size, chunk_overlap, llm_provider)
    df_search_results = pd.DataFrame(cdd.search_results)
    df_contents = pd.DataFrame(cdd.web_contents)
    df_merged = pd.merge(df_search_results, df_contents, how="left", on="url")
    df_merged["Content"] = df_merged["text"].notnull()
    df_merged.drop("text", axis=1, inplace=True)
    df_tags = pd.DataFrame(tags)
    df_merged2 = pd.merge(df_merged, df_tags, how="left", on="url")
    return df_merged2.sort_values(by="url").to_html(table_id="tbl_search_results", render_links=True, index=False)


@app.get("/cdd_with_llm/summarization")
async def summarization(company_name: str,
                        lang: str = "en-US",
                        chunk_size: int = 2000,
                        chunk_overlap: int = 100,
                        llm_provider: str = "AzureOpenAI",
                        ):
    cdd = CDDwithLLM(company_name, lang)
    cdd.search_from_mongo()
    cdd.contents_from_mongo()
    summary = cdd.summarization(chunk_size, chunk_overlap, llm_provider)
    return summary


@app.get("/cdd_with_llm/qa")
async def qa(company_name: str,
             lang: str = "en-US",
             query: Optional[str] = None,
             with_his_data: bool = False,
             data_within_days: int = 90,
             chunk_size: int = 1000,
             chunk_overlap: int = 100,
             embedding_provider: str = "AzureOpenAI",
             llm_provider: str = "AzureOpenAI",
             ):
    cdd = CDDwithLLM(company_name, lang)
    cdd.search_from_mongo()
    cdd.contents_from_mongo()
    answer = cdd.qa(query, with_his_data, data_within_days,
                    chunk_size, chunk_overlap, embedding_provider, llm_provider)
    return answer


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serv:app", host="localhost", port=8000, reload=True)
    # uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="tf02+1-key.pem", ssl_certfile="tf02+1.pem")
