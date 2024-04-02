import pandas as pd
import jsonpickle
import uvicorn
from fastapi import FastAPI, Request
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.cors import CORSMiddleware
from CDDwithLLM import CDDwithLLM

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


@app.get("/cdd_with_llm/web_search")
def web_search(company_name: str, 
               lang: str, 
               search_engine: str, 
               num_results: int, 
               request: Request):
    cdd = CDDwithLLM(company_name, lang)
    cdd.web_search(search_engine=search_engine, num_results=num_results)
    request.session["cdd"] = jsonpickle.encode(cdd)

    df_search = pd.DataFrame(cdd.search_results).sort_values(by="url")
    return df_search.to_html(table_id="tbl_search_res", 
                             render_links=True, 
                             index=False, 
                             justify="left", 
                             na_rep="NA")


@app.get("/cdd_with_llm/contents_from_crawler")
def contents_from_crawler(min_content_length: int, 
                          contents_load: str, 
                          contents_save: str, 
                          request: Request):
    contents_load = contents_load == "true"
    contents_save = contents_save == "true"

    cdd = jsonpickle.decode(request.session["cdd"])
    cdd.contents_from_crawler(min_content_length, contents_load, contents_save)
    request.session["cdd"] = jsonpickle.encode(cdd)

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
def fc_tagging(tagging_strategy: str, 
               tagging_chunk_size: int, 
               tagging_llm_model: str, 
               tags_load: str, 
               tags_save: str, 
               request: Request):
    cdd = jsonpickle.decode(request.session["cdd"])

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
def summary(summary_max_words: int, 
            summary_clus_docs: str, 
            summary_num_clus: int, 
            summary_chunk_size: int, 
            summary_llm_model: str, 
            request: Request):
    cdd = jsonpickle.decode(request.session["cdd"])

    clus_docs = summary_clus_docs == "true"
    num_clus = summary_num_clus or 2

    summary = cdd.summary(max_words=summary_max_words, 
                          clus_docs=clus_docs, 
                          num_clus=num_clus,
                          chunk_size=summary_chunk_size, 
                          llm_model=summary_llm_model)

    return summary


@app.get("/cdd_with_llm/qa")
def qa(ta_qa_query: str, 
       with_his_data: str, 
       data_within_days: int, 
       qa_chunk_size: int, 
       qa_llm_model: str, 
       request: Request):
    cdd = jsonpickle.decode(request.session["cdd"])

    with_his_data = with_his_data == "true"
    data_within_days = data_within_days or 90

    answer = cdd.qa(query=ta_qa_query, 
                    with_his_data=with_his_data,
                    data_within_days=data_within_days, 
                    chunk_size=qa_chunk_size, 
                    llm_model=qa_llm_model)

    return answer


if __name__ == '__main__':
    uvicorn.run('serv_fastapi:app', port=8000, reload=True)
