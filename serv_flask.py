from flask import Flask, request
from flask_cors import CORS
import json
import pandas as pd

from CDDwithLLM import CDDwithLLM

VI_DEPLOY = False


app = Flask(__name__)
CORS(app, supports_credentials=True, origins="*")
# app.config["SESSION_COOKIE_SECURE"] = True
# app.config["SESSION_COOKIE_SAMESITE"] = "None"
# app.secret_key = "192b9bdd22ab9ed4dc15f71bbf5dc987d54727823bcbf"


@app.get("/cdd_with_llm/web_search")
def web_search():
    userid = request.args.get("userid")
    company_name = request.args.get("company_name")
    lang = request.args.get("lang")

    search_engine = request.args.get("search_engine")
    num_results = request.args.get("num_results")

    cdd = CDDwithLLM(company_name, lang)

    # cdd.web_search(search_engine=search_engine, num_results=num_results)
    # cdd.search_to_mongo(userid)

    cdd.search_from_mongo(userid)

    df_search = pd.DataFrame(cdd.search_results).sort_values(by="url")
    return df_search.to_html(table_id="tbl_search_results", render_links=True, index=False)


@app.get("/cdd_with_llm/contents_from_crawler")
def contents_from_crawler():
    userid = request.args.get("userid")
    company_name = request.args.get("company_name")
    lang = request.args.get("lang")

    cdd = CDDwithLLM(company_name, lang)
    cdd.search_from_mongo(userid)
    # cdd.contents_from_crawler()
    # cdd.contents_to_mongo(userid)
    cdd.contents_from_mongo(userid)

    df_search_results = pd.DataFrame(cdd.search_results)
    df_contents = pd.DataFrame(cdd.web_contents)
    df_merged = pd.merge(df_search_results, df_contents, how="left", on="url")
    df_merged["Content"] = df_merged["text"].notnull()
    df_merged.drop("text", axis=1, inplace=True)
    df_merged.sort_values(by="url", inplace=True)

    return df_merged.to_html(table_id="tbl_search_results", render_links=True, index=False)


@app.get("/cdd_with_llm/fca_tagging")
def fca_tagging():
    userid = request.args.get("userid")
    company_name = request.args.get("company_name")
    lang = request.args.get("lang")

    cdd = CDDwithLLM(company_name, lang)
    cdd.search_from_mongo(userid)
    cdd.contents_from_mongo(userid)

    strategy = request.args.get("strategy")
    chunk_size = int(request.args.get("chunk_size"))
    llm_model = request.args.get("llm_model")

    tags = cdd.fca_tagging(
        strategy=strategy, chunk_size=chunk_size, llm_model=llm_model)

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
async def summary():
    userid = request.args.get("userid")
    company_name = request.args.get("company_name")
    lang = request.args.get("lang")

    cdd = CDDwithLLM(company_name, lang)
    cdd.search_from_mongo(userid)
    cdd.contents_from_mongo(userid)

    max_words = int(request.args.get("max_words"))
    clus_docs = request.args.get("clus_docs") == "true"
    num_clus = int(request.args.get("num_clus"))
    chunk_size = int(request.args.get("chunk_size"))
    llm_model = request.args.get("llm_model")

    summ = cdd.summary(max_words=max_words, clus_docs=clus_docs, num_clus=num_clus,
                       chunk_size=chunk_size, llm_model=llm_model)
    
    return summ


@app.get("/cdd_with_llm/qa")
async def qa():
    userid = request.args.get("userid")
    company_name = request.args.get("company_name")
    lang = request.args.get("lang")

    cdd = CDDwithLLM(company_name, lang)
    cdd.search_from_mongo(userid)
    cdd.contents_from_mongo(userid)

    query = request.args.get("query")
    chunk_size = int(request.args.get("chunk_size"))
    llm_model = request.args.get("llm_model")

    answer = cdd.qa(query=query, chunk_size=chunk_size,
                    llm_model=llm_model)

    return answer


if __name__ == "__main__":
    if VI_DEPLOY:
        app.run("0.0.0.0", 8000,
                ssl_context=("tf02+1.pem", "tf02+1-key.pem"))
    else:
        app.run("localhost", 8000, debug=True)
