from flask import Flask, session, request
from flask_cors import CORS
import json
import pandas as pd
from typing import Optional, List

from CDDwithLLM import CDDwithLLM

app = Flask(__name__)
CORS(app)
app.secret_key = "192b9bdd22ab9ed4dc15f71bbf5dc987d54727823bcbf"

@app.get("/cdd_with_llm/web_search")
def web_search():
    company_name = request.args.get("company_name")
    lang = request.args.get("lang")
    search_engine = request.args.get("search_engine")
    num_results = request.args.get("num_results")
    cdd = CDDwithLLM(company_name, lang)
    cdd.web_search(search_engine=search_engine, num_results=num_results)
    # cdd.search_to_mongo()
    # cdd.search_from_mongo()
    session["company_name"] = cdd.company_name
    session["lang"] = cdd.lang
    session["search_results"] = cdd.search_results
    print(session)
    return pd.DataFrame(cdd.search_results).sort_values(by="url").to_html(table_id="tbl_search_results", render_links=True, index=False)

@app.get("/cdd_with_llm/contents_from_crawler")
def contents_from_crawler():
    print(session)
    cdd = CDDwithLLM(session["company_name"], session["lang"])
    cdd.search_results = session["search_results"]
    # cdd.search_from_mongo()
    cdd.contents_from_crawler()
    # cdd.contents_to_mongo()
    # cdd.search_from_mongo()
    # cdd.contents_from_mongo()
    session["web_contents"] = cdd.web_contents
    df_search_results = pd.DataFrame(cdd.search_results)
    df_contents = pd.DataFrame(cdd.web_contents)
    df_merged = pd.merge(df_search_results, df_contents, how="left", on="url")
    df_merged["Content"] = df_merged["text"].notnull()
    df_merged.drop("text", axis=1, inplace=True)
    return df_merged.sort_values(by="url").to_html(table_id="tbl_search_results", render_links=True, index=False)

if __name__=="__main__":
    app.run("localhost", 8000, debug=True)