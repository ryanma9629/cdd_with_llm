function txt2html(txt) {
    var html = txt.replace(/(?:\r\n|\r|\n)/g, " <br><br>");
    var urlReg = /(https?:\/\/[^\s]+)/g;
    return html.replace(urlReg, function (url) {
        return "<a href=\"" + url + "\" target=\"_blank\">" + url + "</a>";
    })
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

    var div_summary = $("#div_summary");
    var p_summary = $("#p_summary");

    var div_qa = $("#div_qa");
    var frm_qa = $("#frm_qa");
    var ta_query = $("#ta_query");

    var div_answer = $("#div_answer");
    var p_answer = $("#p_answer");


    var company_name = new URLSearchParams(window.location.search).get("company_name");
    console.log(company_name);

    var lang;

    // var hostReg = new RegExp(/https?:\/\/[^/]+/);

    frm_web_search.on("submit", function (e) {
        e.preventDefault();

        // company_name = $("#company_name").val();
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

        div_qa.hide();
        if (lang == "zh-CN") {
            ta_query.text(company_name + "有哪些负面新闻？总结不超过3条主要的，每条独立一行列出，并给出信息出处的URL。");
        } else if (lang == "zh-HK" || lang == "zh-TW") {
            ta_query.text(company_name + "有哪些負面新聞？總結不超過3條主要的，每條獨立一行列出，並給出資訊出處的URL。");
        } else {
            ta_query.text("What is the negative news about " + company_name + "? Summarize no more than 3 major ones, list each on a separate line, and give the URL where the information came from.");
        }

        div_answer.hide();
        p_answer.empty();

        form_data = $(this).serializeArray();
        form_data.push({"name": "company_name", "value": company_name});
        console.log(form_data);

        $.ajax({
            url: "http://localhost:8000/cdd_with_llm/web_search",
            // url: "http://tf02:8000/cdd_with_llm/web_search",
            data: form_data,
            type: "GET",
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
            div_search_results.show();
            div_operation.show();
        });
    });

    btn_crawler.on("click", function (e) {
        e.preventDefault();
        $.ajax({
            url: "http://localhost:8000/cdd_with_llm/contents_from_crawler",
            // url: "http://tf02:8000/cdd_with_llm/contents_from_crawler",
            data: {
                "company_name": company_name,
                "lang": lang,
            },
            type: "GET",
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
            div_operation.show();
            btn_tagging.prop("disabled", false)
            btn_summary.prop("disabled", false)
            btn_qa.prop("disabled", false)
        });
    });

    btn_tagging.on("click", function (e) {
        e.preventDefault();

        $.ajax({
            url: "http://localhost:8000/cdd_with_llm/fca_tagging",
            // url: "http://tf02:8000/cdd_with_llm/fca_tagging",
            data: {
                "company_name": company_name,
                "lang": lang,
            },
            type: "GET",
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
        });
    });

    btn_summary.on("click", function (e) {
        e.preventDefault();
        p_summary.empty();
        div_summary.hide();

        $.ajax({
            url: "http://localhost:8000/cdd_with_llm/summarization",
            // url: "http://tf02:8000/cdd_with_llm/summarization",
            data: {
                "company_name": company_name,
                "lang": lang,
            },
            type: "GET",
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
        div_qa.hide();
        p_answer.empty();
        div_answer.hide();

        $.ajax({
            url: "http://localhost:8000/cdd_with_llm/qa",
            // url: "http://tf02:8000/cdd_with_llm/qa",
            data: {
                "company_name": company_name,
                "lang": lang,
                "query": $("#ta_query").val(),
            },
            type: "GET",
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
            p_answer.html(txt2html(txt));
        });
    });
});