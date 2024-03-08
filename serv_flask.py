import os

import jsonpickle
import pandas as pd
import pymongo
from flask import Flask, request, session
from flask_cors import CORS
from flask_session import Session

from CDDwithLLM import CDDwithLLM

VI_DEPLOY = False


app = Flask(__name__)
CORS(app, supports_credentials=True, origins="http://127.0.0.1:5500")
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.secret_key = "192b9bdd22ab9ed4dc15f71bbf5dc987d54727823bcbf"
app.config["SESSION_TYPE"] = "mongodb"
app.config["SESSION_MONGODB"] = pymongo.MongoClient(os.getenv("MONGODB_URI"))

server_session = Session(app)


@app.get("/cdd_with_llm/web_search")
def web_search():
    company_name = request.args.get("company_name")
    lang = request.args.get("lang")

    search_engine = request.args.get("search_engine")
    num_results = request.args.get("num_results")

    cdd = CDDwithLLM(company_name, lang)

    cdd.web_search(search_engine=search_engine, num_results=num_results)
    session["cdd"] = jsonpickle.encode(cdd)

    df_search = pd.DataFrame(cdd.search_results).sort_values(by="url")
    return df_search.to_html(table_id="tbl_search_results", render_links=True, index=False)


@app.get("/cdd_with_llm/contents_from_crawler")
def contents_from_crawler():
    min_content_length = int(request.args.get("min_content_length"))
    contents_load = request.args.get("contents_load") == "true"
    contents_save = request.args.get("contents_save") == "true"

    cdd = jsonpickle.decode(session["cdd"])
    cdd.contents_from_crawler(min_content_length, contents_load, contents_save)
    session["cdd"] = jsonpickle.encode(cdd)

    df_search_results = pd.DataFrame(cdd.search_results)
    df_contents = pd.DataFrame(cdd.web_contents)
    df_merged = pd.merge(df_search_results, df_contents, how="left", on="url")
    df_merged["Content"] = df_merged["text"].notnull()
    df_merged.drop("text", axis=1, inplace=True)
    df_merged.sort_values(by="url", inplace=True)

    return df_merged.to_html(table_id="tbl_search_results", render_links=True, index=False)


@app.get("/cdd_with_llm/fc_tagging")
def fc_tagging():
    cdd = jsonpickle.decode(session["cdd"])

    strategy = request.args.get("tagging_strategy")
    chunk_size = int(request.args.get("tagging_chunk_size"))
    llm_model = request.args.get("tagging_llm_model")
    tags_load = request.args.get("tags_load") == "true"
    tags_save = request.args.get("tags_save") == "true"

    tags = cdd.fc_tagging(
        strategy=strategy, chunk_size=chunk_size, llm_model=llm_model, tags_load=tags_load, tags_save=tags_save)

    df_search_results = pd.DataFrame(cdd.search_results)
    df_contents = pd.DataFrame(cdd.web_contents)
    df_merged = pd.merge(df_search_results, df_contents, how="left", on="url")
    df_merged["Content"] = df_merged["text"].notnull()
    df_merged.drop("text", axis=1, inplace=True)
    df_tags = pd.DataFrame(tags)
    df_merged2 = pd.merge(df_merged, df_tags, how="left", on="url")
    df_merged2.sort_values(by="url", inplace=True)

    return df_merged2.to_html(table_id="tbl_search_results", render_links=True, index=False)


@app.get("/cdd_with_llm/summary")
def summary():
    cdd = jsonpickle.decode(session["cdd"])

    max_words = int(request.args.get("summary_max_words"))
    clus_docs = request.args.get("summary_clus_docs") == "true"
    num_clus = int(request.args.get("summary_num_clus", "5"))
    chunk_size = int(request.args.get("summary_chunk_size"))
    llm_model = request.args.get("summary_llm_model")

    summ = cdd.summary(max_words=max_words, clus_docs=clus_docs, num_clus=num_clus,
                       chunk_size=chunk_size, llm_model=llm_model)

    return summ


@app.get("/cdd_with_llm/qa")
def qa():
    cdd = jsonpickle.decode(session["cdd"])

    query = request.args.get("qa_query")
    with_his_data = request.args.get("with_his_data") == "true"
    data_within_days = int(request.args.get("data_within_days", "90"))
    chunk_size = int(request.args.get("qa_chunk_size"))
    llm_model = request.args.get("qa_llm_model")

    answer = cdd.qa(query=query, with_his_data=with_his_data,
                    data_within_days=data_within_days, chunk_size=chunk_size, llm_model=llm_model)

    return answer


if __name__ == "__main__":
    if VI_DEPLOY:
        app.run("0.0.0.0", 8000,
                ssl_context=("tf02+1.pem", "tf02+1-key.pem"))
    else:
        app.run("localhost", 8000, debug=True)
