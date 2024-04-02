import os
from typing import Dict, Optional
from uuid import uuid4

import jsonpickle
import pandas as pd
import pymongo
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
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


def mongo_save(session_id: str, cdd: Dict):
    client = pymongo.MongoClient(os.getenv("MONGO_URI"))
    col = client.fastapi_session["sessions"]
    col.update_one({"session_id": session_id},
                   {"$set": {"session_id": session_id, "cdd": cdd}},
                   upsert=True)
    client.close()


def mongo_load(session_id: str):
    client = pymongo.MongoClient(os.getenv("MONGO_URI"))
    col = client.fastapi_session["sessions"]
    cursor = col.find({"session_id": session_id},
                      {"_id": 0, "cdd": 1})
    res = list(cursor)[0]
    return res["cdd"]


@app.get("/cdd_with_llm/web_search")
def web_search(request: Request,
               company_name: str,
               lang: str = "en-US",
               search_engine: str = "Bing",
               num_results: int = 5
               ):
    cdd = CDDwithLLM(company_name, lang)
    cdd.web_search(search_engine=search_engine, num_results=num_results)
    session_id = uuid4().hex
    request.session["id"] = session_id
    mongo_save(session_id, jsonpickle.encode(cdd))

    df_search = pd.DataFrame(cdd.search_results).sort_values(by="url")
    return df_search.to_html(table_id="tbl_search_res",
                             render_links=True,
                             index=False,
                             justify="left",
                             na_rep="NA")


@app.get("/cdd_with_llm/contents_from_crawler")
def contents_from_crawler(request: Request,
                          min_content_length: int = 100,
                          contents_load: str = "true",
                          contents_save: str = "true"
                          ):
    contents_load = contents_load == "true"
    contents_save = contents_save == "true"

    session_id = request.session["id"]
    cdd = jsonpickle.decode(mongo_load(session_id))
    cdd.contents_from_crawler(min_content_length, contents_load, contents_save)
    mongo_save(session_id, jsonpickle.encode(cdd))

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
               tagging_strategy: str = "all",
               tagging_chunk_size: int = 2000,
               tagging_llm_model: str = "GPT4",
               tags_load: str = "true",
               tags_save: str = "true",
               ):
    session_id = request.session["id"]
    cdd = jsonpickle.decode(mongo_load(session_id))

    tags_load = tags_load == "true"
    tags_save = tags_save == "true"
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
            summary_max_words: int = 300,
            summary_clus_docs: str = "true",
            summary_chunk_size: int = 2000,
            summary_llm_model: str = "GPT4",
            summary_num_clus: Optional[int] = 2):
    session_id = request.session["id"]
    cdd = jsonpickle.decode(mongo_load(session_id))

    clus_docs = summary_clus_docs == "true"

    summary = cdd.summary(max_words=summary_max_words,
                          clus_docs=clus_docs,
                          num_clus=summary_num_clus,
                          chunk_size=summary_chunk_size,
                          llm_model=summary_llm_model)

    return summary


@app.get("/cdd_with_llm/qa")
def qa(request: Request,
       ta_qa_query: str,
       with_his_data: str = "false",
       data_within_days: Optional[int] = 90,
       qa_chunk_size: int = 2000,
       qa_llm_model: str = "GPT4"
       ):
    session_id = request.session["id"]
    cdd = jsonpickle.decode(mongo_load(session_id))

    with_his_data = with_his_data == "true"

    answer = cdd.qa(query=ta_qa_query,
                    with_his_data=with_his_data,
                    data_within_days=data_within_days,
                    chunk_size=qa_chunk_size,
                    llm_model=qa_llm_model)

    return answer


if __name__ == '__main__':
    VI_DEPLOY = False

    if VI_DEPLOY:
        uvicorn.run('serv_fastapi:app',
                    host="0.0.0.0", port=8000,
                    ssl_keyfile="tf02+1-key.pem", ssl_certfile="tf02+1.pem")
    else:
        uvicorn.run('serv_fastapi:app', port=8000, reload=True)
