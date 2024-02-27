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
    var tbl_search_results = $("#tbl_search_results");
    var div_tagging_and_summary = $("#div_tagging_and_summary");
    var btn_tagging = $("#btn_tagging");
    var btn_summary = $("#btn_summary");
    var div_summary = $("#div_summary");
    var p_summary = $("#p_summary");
    var div_qa = $("#div_qa");
    var frm_qa = $("#frm_qa");
    var ta_query = $("#ta_query");
    var div_answer = $("#div_answer");
    var p_answer = $("#p_answer");

    var company_name;
    var lang;
    var embedding_provider;
    var llm_provider;

    // var search_results;
    var hostReg = new RegExp(/https?:\/\/[^/]+/);

    frm_web_search.on("submit", function (e) {
        e.preventDefault();

        company_name = $("#company_name").val();
        lang = $("#lang").val();
        embedding_provider = $("#embedding_provider").val();
        llm_provider = $("#llm_provider").val();

        div_search_results.hide();
        tbl_search_results.empty();
        div_tagging_and_summary.hide();
        div_summary.hide();
        p_summary.empty();
        div_qa.hide();
        div_answer.hide();
        p_answer.empty();

        head_html = "<thead><tr><th>URL</th><th>Title</th></tr></thead>";
        tbl_search_results.append($(head_html));
        var tbl_body = $("#tbl_search_results tbody");
        if (tbl_body.length) {
            tbl_body.remove();
        }

        $.ajax({
            url: "http://localhost:8000/cdd_with_llm/web_search",
            data: $(this).serialize(),
            type: "GET",
            dataType: "json",
            beforeSend: function () {
                div_ajax.show();
                p_ajax.html("Crawling web contents... may take a few minutes")
            },
            complete: function () {
                p_ajax.empty();
                div_ajax.hide();
            },
        }).done(function (json_obj) {
            console.log(json_obj);

            var tbl_body = document.createElement("tbody");
            $.each(json_obj, function () {
                var tbl_row = tbl_body.insertRow();
                $.each(this, function (k, v) {
                    var cell = tbl_row.insertCell();
                    if (k.toString() == "url") {
                        var a = document.createElement('a');
                        var linkText = document.createTextNode(hostReg.exec(v.toString()));
                        a.appendChild(linkText);
                        a.href = v.toString();
                        a.target = "_blank";
                        cell.appendChild(a);
                    } else {
                        cell.appendChild(document.createTextNode(v.toString().replace(/(<([^>]+)>)/ig, '')));
                    }
                });
            })
            tbl_search_results.append(tbl_body);
            div_search_results.show();
            div_tagging_and_summary.show();

            if (lang == "zh-CN") {
                ta_query.text(company_name + "有哪些负面新闻？总结不超过3条主要的，每条独立一行列出，并给出信息出处的URL。");
            } else if (lang == "zh-HK" || lang == "zh-TW") {
                ta_query.text(company_name + "有哪些負面新聞？總結不超過3條主要的，每條獨立一行列出，並給出資訊出處的URL。");
            } else {
                ta_query.text("What is the negative news about " + company_name + "? Summarize no more than 3 major ones, list each on a separate line, and give the URL where the information came from.");
            }
            div_qa.show();
        });
    });

    btn_tagging.on("click", function (e) {
        e.preventDefault();

        $.ajax({
            url: "http://localhost:8000/cdd_with_llm/fca_tagging",
            data: {
                "company_name": company_name,
                "lang": lang,
                "llm_provider": llm_provider,
            },
            type: "GET",
            dataType: "json",
            beforeSend: function () {
                div_ajax.show();
                p_ajax.text("Tagging for each news... may take a few minutes")
            },
            complete: function () {
                p_ajax.empty();
                div_ajax.hide();
            },
        }).done(function (json_obj) {
            console.log(json_obj);

            if ($("#tbl_search_results thead tr th").length == 5) {
                for (var i = 3; i--;) {
                    $("#tbl_search_results th:last-child, #tbl_search_results td:last-child").remove();
                }
            };

            new_head_html = "<th style=\"width: 10%\">Suspect?</th><th style=\"width: 10%\">Type</th style=\"width: 10%\"><th>Probability</th>"
            $("#tbl_search_results thead tr").append($(new_head_html));
            var tr = $("#tbl_search_results tbody tr");
            var na = $("<td>NA</td>");
            for (var i = 0; i < tr.length; i++) {
                console.log(json_obj[i].length);
                if (Object.keys(json_obj[i]).length) {
                    tr.eq(i).append($("<td>" + json_obj[i]["suspected of financial crimes"].toString() + "</td>"));
                    tr.eq(i).append($("<td>" + json_obj[i]["types of suspected financial crimes"].toString() + "</td>"));
                    tr.eq(i).append($("<td>" + json_obj[i]["probability"].toString() + "</td>"));
                } else {
                    tr.eq(i).append(na);
                    tr.eq(i).append(na);
                    tr.eq(i).append(na);
                }
            };
        });
    });

    btn_summary.on("click", function (e) {
        e.preventDefault();
        $.ajax({
            url: "http://localhost:8000/cdd_with_llm/summarization",
            data: {
                "company_name": company_name,
                "lang": lang,
                "llm_provider": llm_provider,
            },
            type: "GET",
            dataType: "json",
            beforeSend: function () {
                div_ajax.show();
                p_ajax.html("Making summary for there news... may take a few minutes")
            },
            complete: function () {
                p_ajax.empty();
                div_ajax.hide();
            },
        }).done(function (json_obj) {
            console.log(json_obj);
            div_summary.show();
            p_summary.html(txt2html(json_obj));
        });
    });

    frm_qa.on("submit", function (e) {
        e.preventDefault();

        $.ajax({
            url: "http://localhost:8000/cdd_with_llm/qa",
            data: {
                "company_name": company_name,
                "lang": lang,
                "query": $("#ta_query").val(),
                "embedding_provider": embedding_provider,
                "llm_provider": llm_provider,
            },
            type: "GET",
            dataType: "json",
            beforeSend: function () {
                div_ajax.show();
                p_ajax.html("Making question-answering on these news... may take a few minutes")
            },
            complete: function () {
                p_ajax.empty();
                div_ajax.hide();
            },
        }).done(function (json_obj) {
            console.log(json_obj);
            div_answer.show();
            p_answer.html(txt2html(json_obj));
        });
    });
});