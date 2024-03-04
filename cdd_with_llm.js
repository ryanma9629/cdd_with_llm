const vi_deploy = false;
const html_tags = /(<([^>]+)>)/ig;

function txt2html(txt) {
    var html = txt.replace(/(?:\r\n|\r|\n)/g, " <br><br>");
    var urlReg = /(https?:\/\/[^\s]+)/g;
    return html.replace(urlReg, function (url) {
        return "<a href=\"" + url + "\" target=\"_blank\">" + url + "</a>";
    })
}

function uuidv4() {
    return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
        (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
    );
}


$(document).ready(function () {
    var div_ajax = $("#div_ajax");
    var p_ajax = $("#p_ajax");

    var frm_web_search = $("#frm_web_search");
    var div_search_results = $("#div_search_results");

    var div_operation = $("#div_operation");
    var btn_crawler = $("#btn_crawler");
    var btn_tagging = $("#btn_tagging");
    var btn_summary = $("#btn_summary");
    var btn_qa = $("#btn_qa");

    var div_tagging_frm = $("#div_tagging_frm");
    var frm_tagging = $("#frm_tagging");

    var div_summary_frm = $("#div_summary_frm");
    var frm_summary = $("#frm_summary");

    var div_qa_frm = $("#div_qa_frm");
    var frm_qa = $("#frm_qa");
    var qa_query = $("#qa_query");

    var div_summary = $("#div_summary");
    var p_summary = $("#p_summary");

    var div_answer = $("#div_answer");
    var p_question = $("#p_question");
    var p_answer = $("#p_answer");

    var company_name;
    if (vi_deploy) {
        company_name = new URLSearchParams(window.location.search).get("company_name");
    }

    var lang;

    if (vi_deploy) {
        $(".vi_deploy").remove();
        api_host = "https://tf02:8000/";
    } else {
        api_host = "http://localhost:8000/";
    }
    // var hostReg = new RegExp(/https?:\/\/[^/]+/);

    var userid = localStorage.getItem("userid");
    if (userid == null) {
        userid = uuidv4();
        localStorage.setItem("userid", userid);
    }

    frm_web_search.on("submit", function (e) {
        e.preventDefault();

        company_name = $("#company_name").val();
        // company_name = new URLSearchParams(window.location.search).get("company_name");
        lang = $("#lang").val();

        div_search_results.hide();
        if (div_search_results.length) {
            div_search_results.empty();
        }

        div_operation.hide();
        btn_tagging.prop("disabled", true)
        btn_summary.prop("disabled", true)
        btn_qa.prop("disabled", true)

        div_summary.hide();
        p_summary.empty();

        if (lang == "zh-CN") {
            qa_query.val(company_name + "有哪些负面新闻？总结不超过3条主要的，每条独立一行列出，并给出信息出处的URL");
        } else if (lang == "zh-HK" || lang == "zh-TW") {
            qa_query.val(company_name + "有哪些負面新聞？總結不超過3條主要的，每條獨立一行列出，並給出資訊出處的URL");
        } else if (lang == "ja-JP") {
            qa_query.val(company_name + "に関するネガティブなニュースをサーチしなさい。一番大事なものを三つ以内にまとめ、それぞれを箇条書きし、出典元URLを付けなさい");
        } else {
            qa_query.val("What is the negative news about " + company_name + "? Summarize no more than 3 major ones, list each on a separate line, and give the URL where the information came from.");
        }

        div_answer.hide();
        p_answer.empty();

        form_data = $(this).serializeArray();
        form_data.push({"name": "userid", "value": userid});
        if (vi_deploy) {
            form_data.push({ "name": "company_name", "value": company_name });
        }

        // form_data.push({ "name": "company_name", "value": company_name });

        $.ajax({
            // url: "http://localhost:8000/cdd_with_llm/web_search",
            // url: "https://tf02:8000/cdd_with_llm/web_search",
            url: api_host + "cdd_with_llm/web_search",
            data: form_data,
            type: "GET",
            xhrFields: {
                withCredentials: true
            },
            beforeSend: function () {
                div_ajax.show();
                p_ajax.html("Making web search... may take a few seconds")
            },
            complete: function () {
                p_ajax.empty();
                div_ajax.hide();
            },
        }).done(function (html) {
            // console.log(html);
            div_search_results.append(html);

            // remove html tabs in titles
            div_search_results.find("table tbody tr td:nth-child(2)").each(function () {
                text_with_tags = $(this).text();
                text_wo_tags = text_with_tags.replace(html_tags, "");
                $(this).text(text_wo_tags);
            });
            
            div_search_results.show();
            div_operation.show();
        });
    });

    btn_crawler.on("click", function (e) {
        e.preventDefault();
        $.ajax({
            // url: "http://localhost:8000/cdd_with_llm/contents_from_crawler",
            // url: "https://tf02:8000/cdd_with_llm/contents_from_crawler",
            url: api_host + "cdd_with_llm/contents_from_crawler",
            data: {
                "company_name": company_name,
                "lang": lang,
                "userid": userid,
            },
            type: "GET",
            xhrFields: {
                withCredentials: true
            },
            beforeSend: function () {
                div_ajax.show();
                p_ajax.text("Grabbing web conetents from each url... may take a few minutes")
            },
            complete: function () {
                p_ajax.empty();
                div_ajax.hide();
            },
        }).done(function (html) {
            // console.log(html);
            if (div_search_results.length) {
                div_search_results.empty();
            }
            div_search_results.append(html);
            div_search_results.find("table tbody tr td:nth-child(2)").each(function () {
                text_with_tags = $(this).text();
                text_wo_tags = text_with_tags.replace(html_tags, "");
                $(this).text(text_wo_tags);
            });

            div_operation.show();
            btn_tagging.prop("disabled", false)
            btn_summary.prop("disabled", false)
            btn_qa.prop("disabled", false)
        });
    });

    frm_tagging.on("submit", function (e) {
        e.preventDefault();
        div_tagging_frm.hide();

        $.ajax({
            // url: "http://localhost:8000/cdd_with_llm/fca_tagging",
            // url: "https://tf02:8000/cdd_with_llm/fca_tagging",
            url: api_host + "cdd_with_llm/fca_tagging",
            data: {
                "company_name": company_name,
                "lang": lang,
                "userid": userid,
                "strategy": $("#tagging_strategy").val(),
                "chunk_size": $("#tagging_chunk_size").val(),
                "llm_provider": $("#tagging_llm_provider").val(),
            },
            type: "GET",
            xhrFields: {
                withCredentials: true
            },
            beforeSend: function () {
                div_ajax.show();
                p_ajax.text("Tagging for each news... may take a few minutes")
            },
            complete: function () {
                p_ajax.empty();
                div_ajax.hide();
            },
        }).done(function (html) {
            // console.log(html);
            if (div_search_results.length) {
                div_search_results.empty();
            }
            div_search_results.append(html);
            div_search_results.find("table tbody tr td:nth-child(2)").each(function () {
                text_with_tags = $(this).text();
                text_wo_tags = text_with_tags.replace(html_tags, "");
                $(this).text(text_wo_tags);
            });

        });
    });

    frm_summary.on("submit", function (e) {
        e.preventDefault();
        div_summary_frm.hide();
        p_summary.empty();
        div_summary.hide();

        $.ajax({
            // url: "http://localhost:8000/cdd_with_llm/summary",
            // url: "https://tf02:8000/cdd_with_llm/summary",
            url: api_host + "cdd_with_llm/summary",
            data: {
                "company_name": company_name,
                "lang": lang,
                "userid": userid,
                "max_words": $("#summary_max_words").val(),
                "chunk_size": $("#summary_chunk_size").val(),
                "llm_provider": $("#summary_llm_provider").val(),
            },
            type: "GET",
            xhrFields: {
                withCredentials: true
            },
            beforeSend: function () {
                div_ajax.show();
                p_ajax.html("Making summary for there news... may take a few minutes")
            },
            complete: function () {
                p_ajax.empty();
                div_ajax.hide();
            },
        }).done(function (txt) {
            // console.log(txt);
            div_summary.show();
            p_summary.html(txt2html(txt));
        });
    });

    frm_qa.on("submit", function (e) {
        e.preventDefault();
        div_qa_frm.hide();
        p_answer.empty();
        div_answer.hide();

        $.ajax({
            // url: "http://localhost:8000/cdd_with_llm/qa",
            // url: "https://tf02:8000/cdd_with_llm/qa",
            url: api_host + "cdd_with_llm/qa",
            data: {
                "company_name": company_name,
                "lang": lang,
                "userid": userid,
                "query": $("#qa_query").val(),
                "chunk_size": $("#qa_chunk_size").val(),
                "llm_provider": $("#qa_llm_provider").val(),
            },
            type: "GET",
            xhrFields: {
                withCredentials: true
            },
            beforeSend: function () {
                div_ajax.show();
                p_ajax.html("Making question-answering on these news... may take a few minutes")
            },
            complete: function () {
                p_ajax.empty();
                div_ajax.hide();
            },
        }).done(function (txt) {
            // console.log(txt);
            div_answer.show();
            p_question.text(qa_query.val());
            p_answer.html(txt2html(txt));
        });
    });
});