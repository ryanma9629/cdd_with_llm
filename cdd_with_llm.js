function txt2html(txt) {
    var html = txt.replace(/(?:\r\n|\r|\n)/g, " <br><br>");
    // var urlRegex = /(https?:\/\/[^\s]+)/g;
    return html.replace(/(https?:\/\/[^\s]+)/g, function (url) {
        return '<a href="' + url + '">' + url + '</a>';
    })

}

$(document).ready(function () {
    var p_ajax = $("#p_ajax");
    var frm_web_search = $("#frm_web_search");
    var tbl_search_results = $("#tbl_search_results");
    var btn_fca_tagging = $("#btn_fca_tagging");
    var btn_summary = $("#btn_summary");
    var p_summary = $("#p_summary");
    var frm_qa = $("#frm_qa");
    var ta_query = $("#ta_query");
    var p_qa = $("#p_qa");

    var company_name;
    var lang;
    var embedding_provider;
    var llm_provider;

    // var search_results;

    frm_web_search.on("submit", function (e) {
        e.preventDefault();

        company_name = $("#company_name").val();
        lang = $("#lang").val();
        embedding_provider = $("#embedding_provider").val();
        llm_provider = $("#llm_provider").val();

        tbl_search_results.hide();
        tbl_search_results.empty();
        btn_fca_tagging.hide();
        btn_summary.hide();
        p_summary.hide();
        p_summary.empty();
        frm_qa.hide();
        p_qa.hide();
        p_qa.empty();

        head_html = "<thead><tr><th style=\"max-witdh: 20%\">URL</th><th>Title</th></tr></thead>";
        tbl_search_results.html(head_html);
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
                p_ajax.show();
                p_ajax.html("Crawling web contents... may take a few minutes")
            },
            complete: function () {
                p_ajax.empty();
                p_ajax.hide();
            },
        }).done(function (json_obj) {
            console.log(json_obj);
            // search_results = json_obj;

            // var odd_even = false;
            var tbl_body = document.createElement("tbody");
            $.each(json_obj, function () {
                var tbl_row = tbl_body.insertRow();
                // tbl_row.className = odd_even ? "odd" : "even";
                $.each(this, function (k, v) {
                    var cell = tbl_row.insertCell();
                    if (k.toString() == "url") {
                        var a = document.createElement('a');
                        var linkText = document.createTextNode(v.toString());
                        a.appendChild(linkText);
                        a.href = v.toString();
                        cell.appendChild(a);
                    } else {
                        cell.appendChild(document.createTextNode(v.toString()));
                    }
                });
                // odd_even = !odd_even;
            })
            tbl_search_results.append(tbl_body);
            tbl_search_results.show();
            btn_fca_tagging.show();
            btn_summary.show();
            if (lang == "zh-CN") {
                ta_query.text(company_name + "有哪些负面新闻？总结不超过3条主要的，每条独立一行列出，并给出信息出处的URL。");
            } else if (lang == "zh-HK" || lang == "zh-TW") {
                ta_query.text(company_name + "有哪些負面新聞？總結不超過3條主要的，每條獨立一行列出，並給出資訊出處的URL。");
            } else {
                ta_query.text("What is the negative news about " + company_name + "? Summarize no more than 3 major ones, list each on a separate line, and give the URL where the information came from.");
            }
            frm_qa.show();
        });
    });

    btn_fca_tagging.on("click", function (e) {
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
                p_ajax.show();
                p_ajax.html("Tagging for each article... may take a few minutes")
            },
            complete: function () {
                p_ajax.empty();
                p_ajax.hide();
            },
        }).done(function (json_obj) {
            console.log(json_obj);

            if ($("#tbl_search_results thead tr th").length == 5) {
                for (var i = 3; i--;) {
                    $("#tbl_search_results th:last-child, #tbl_search_results td:last-child").remove();
                }
            };

            $("#tbl_search_results thead tr").append($("<th style=\"max-width: 10%\">Suspect?</th><th style=\"max-width: 10%\">Type of Crime</th style=\"max-width: 10%\"><th>Probability</th>"));
            var tr = $("#tbl_search_results tbody tr");
            for (var i = 0; i < tr.length; i++) {
                tr.eq(i).append($("<td>" + json_obj[i]["suspected of financial crimes"].toString() + "</td>"));
                tr.eq(i).append($("<td>" + json_obj[i]["types of suspected financial crimes"].toString() + "</td>"));
                tr.eq(i).append($("<td>" + json_obj[i]["probability"].toString() + "</td>"));
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
                p_ajax.show();
                p_ajax.html("Making summary for there articles... may take a few minutes")
            },
            complete: function () {
                p_ajax.empty();
                p_ajax.hide();
            },
        }).done(function (json_obj) {
            console.log(json_obj);
            p_summary.show();
            p_summary.html(txt2html(json_obj));
        });
    });

    frm_qa.on("submit", function (e) {
        e.preventDefault();
        query = $("#ta_query").val();
        $.ajax({
            url: "http://localhost:8000/cdd_with_llm/qa",
            data: {
                "company_name": company_name,
                "lang": lang,
                "query": query,
                "embedding_provider": embedding_provider,
                "llm_provider": llm_provider,
            },
            type: "GET",
            dataType: "json",
            beforeSend: function () {
                p_ajax.show();
                p_ajax.html("Making question-answering on these articles... may take a few minutes")
            },
            complete: function () {
                p_ajax.empty();
                p_ajax.hide();
            },
        }).done(function (json_obj) {
            console.log(json_obj);
            p_qa.show();
            p_qa.html(txt2html(json_obj));
        });
    });
});