import os
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, Literal
from uuid import uuid4

import jsonpickle
import pandas as pd
import pymongo
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from CDDwithLLM import CDDwithLLM

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["http://127.0.0.1:5500",
                                  "https://tenant01.viyahost.site"],
                   allow_methods=["GET", "POST"],
                   allow_credentials=True)
app.add_middleware(SessionMiddleware,
                   secret_key="192b9bdd22ab9ed4dc15f71bbf5dc987d54727823bcbf",
                   same_site="none",
                   https_only=True)


def mongo_save(session_id: str,
               data: Any,
               session_collection: str = "sessions",
               ttl_days: int = 14) -> None:
    client = pymongo.MongoClient(os.getenv("MONGO_URI"))
    col = client.fastapi_session[session_collection]
    col.update_one({"session_id": session_id},
                   {"$set": {"session_id": session_id,
                             "data": jsonpickle.encode(data),
                             "expiration": datetime.today() + timedelta(days=ttl_days)}},
                   upsert=True)
    client.close()


def mongo_load(session_id: str,
               session_collection: str = "sessions") -> Any:
    client = pymongo.MongoClient(os.getenv("MONGO_URI"))
    col = client.fastapi_session[session_collection]
    cursor = col.find({"session_id": session_id},
                      {"_id": 0, "data": 1})
    res = list(cursor)
    client.close()
    if res:
        return jsonpickle.decode(res[0]["data"])


@app.get("/cdd_with_llm/web_search")
def web_search(request: Request,
               company_name: str,
               lang: Literal["en-US", "zh-CN", "zh-TW",
                             "zh-HK", "ja-JP"] = "en-US",
               search_engine: Literal["Bing", "Google"] = "Bing",
               search_suffix: Literal["negative", "crime", "everything"] = "negative",
               num_results: Annotated[int, Query(ge=1, le=50)] = 5
               ) -> str:
    cdd = CDDwithLLM(company_name, lang)
    if search_suffix == "negative":
        if lang == "zh-CN":
            suffix = "负面新闻"
        elif lang == "zh-HK" or lang == "zh-TW":
            suffix = "負面新聞"
        elif lang == "ja-JP":
            suffix = "悪い知らせ"
        else:
            suffix = "negative news"
    elif search_suffix == "crime":
        if lang == "zh-CN":
            suffix = "涉嫌犯罪"
        elif lang == "zh-HK" or lang == "zh-TW":
            suffix = "涉嫌犯罪"
        elif lang == "ja-JP":
            suffix = "犯罪の疑いがある"
        else:
            suffix = "criminal suspect"
    elif search_suffix == "everything":
        suffix = ""
    
    cdd.web_search(search_suffix=suffix, search_engine=search_engine, num_results=num_results)
    session_id = uuid4().hex
    request.session["id"] = session_id
    mongo_save(session_id, cdd)

    df_search = pd.DataFrame(cdd.search_results).sort_values(by="url")
    return df_search.to_html(table_id="tbl_search_res",
                             render_links=True,
                             index=False,
                             justify="left",
                             na_rep="NA")


@app.get("/cdd_with_llm/contents_crawler")
def contents_crawler(request: Request,
                     min_content_length: Annotated[int, Query(ge=0)] = 100,
                     contents_load: bool = False,
                     contents_save: bool = False
                     ) -> str:
    session_id = request.session["id"]
    cdd = mongo_load(session_id)
    cdd.contents_crawler(min_content_length, contents_load, contents_save)
    mongo_save(session_id, cdd)

    df_search_results = pd.DataFrame(cdd.search_results)
    df_contents = pd.DataFrame(cdd.web_contents)
    df_merged = pd.merge(df_search_results, df_contents, how="left", on="url")
    df_merged["content"] = df_merged["text"].notnull()
    df_merged.drop("text", axis=1, inplace=True)
    df_merged.sort_values(by="url", inplace=True)

    return df_merged.to_html(table_id="tbl_search_res",
                             render_links=True,
                             index=False,
                             justify="left",
                             na_rep="NA")


@app.get("/cdd_with_llm/fc_tagging")
def fc_tagging(request: Request,
               tagging_strategy: Literal["first", "all"] = "all",
               tagging_chunk_size: Annotated[int, Query(gt=0)] = 2000,
               tagging_llm_model: Literal["GPT35",
                                          "GPT4", "GPT4-32k", "ERNIE35", "ERNIE4"] = "GPT4",
               tags_load: bool = False,
               tags_save: bool = False,
               ) -> str:
    session_id = request.session["id"]
    cdd = mongo_load(session_id)

    tags = cdd.fc_tagging(strategy=tagging_strategy,
                          chunk_size=tagging_chunk_size,
                          llm_model=tagging_llm_model,
                          tags_load=tags_load,
                          tags_save=tags_save)

    df_search_results = pd.DataFrame(cdd.search_results)
    df_contents = pd.DataFrame(cdd.web_contents)
    df_merged = pd.merge(df_search_results, df_contents, how="left", on="url")
    df_merged["content"] = df_merged["text"].notnull()
    df_merged.drop("text", axis=1, inplace=True)
    df_tags = pd.DataFrame(tags)
    df_merged2 = pd.merge(df_merged, df_tags, how="left", on="url")
    df_merged2.sort_values(by="url", inplace=True)

    return df_merged2.to_html(table_id="tbl_search_res",
                              render_links=True,
                              index=False,
                              justify="left",
                              na_rep="NA")


@app.get("/cdd_with_llm/summary")
def summary(request: Request,
            summary_max_words: Annotated[int, Query(gt=0)] = 300,
            summary_clus_docs: bool = False,
            summary_chunk_size: Annotated[int, Query(gt=0)] = 2000,
            summary_llm_model: Literal["GPT35", "GPT4", "GPT4-32k", "ERNIE35", "ERNIE4"] = "GPT4",
            summary_num_clus: Annotated[int, Query(ge=2)] = 2
            ) -> str:
    session_id = request.session["id"]
    cdd = mongo_load(session_id)

    summary = cdd.summary(max_words=summary_max_words,
                          clus_docs=summary_clus_docs,
                          num_clus=summary_num_clus,
                          chunk_size=summary_chunk_size,
                          llm_model=summary_llm_model)

    return summary


@app.get("/cdd_with_llm/qa")
def qa(request: Request,
       ta_qa_query: Annotated[str, Query(min_length=1)],
       with_his_data: bool = False,
       data_within_days: Annotated[int, Query(ge=0)] = 90,
       qa_chunk_size: Annotated[int, Query(gt=0)] = 2000,
       qa_llm_model: Literal["GPT35", "GPT4", "GPT4-32k", "ERNIE35", "ERNIE4"] = "GPT4"
       ) -> str:
    session_id = request.session["id"]
    cdd = mongo_load(session_id)

    answer = cdd.qa(query=ta_qa_query,
                    with_his_data=with_his_data,
                    data_within_days=data_within_days,
                    chunk_size=qa_chunk_size,
                    llm_model=qa_llm_model)

    return answer


if __name__ == '__main__':
    VI_DEPLOY = False

    if VI_DEPLOY:
        uvicorn.run('serv_fastapi:app', host="0.0.0.0", port=8000,
                    ssl_keyfile="tf02+1-key.pem",
                    ssl_certfile="tf02+1.pem")
    else:
        uvicorn.run('serv_fastapi:app', port=8000, reload=True)
