$(document).ready(function () {

    var frm_web_search = $("#frm_web_search");
    var tbl_search_results = $("#tbl_search_results");
    var btn_fca_tagging = $("#btn_fca_tagging");
    var btn_summary = $("#btn_summary");
    var p_summary = $("#p_summary");
    var frm_qa = $("#frm_qa");
    var p_qa = $("#p_qa");

    var company_name;
    var lang;
    var embedding_provider;
    var llm_provider;

    var search_results;

    frm_web_search.submit(function (e) {
        e.preventDefault();

        company_name = $("#company_name").val();
        lang = $("#lang").val();
        embedding_provider = $("#embedding_provider").val();
        llm_provider = $("#llm_provider").val();

        tbl_search_results.hide();
        tbl_search_results.empty();
        head_html = "<thead><tr><th>URL</th><th>Title</th></tr></thead>";
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
        }).done(function (json_str) {
            search_results = json_str;

            var odd_even = false;
            var tbl_body = document.createElement("tbody");
            $.each(JSON.parse(json_str), function () {
                var tbl_row = tbl_body.insertRow();
                tbl_row.className = odd_even ? "odd" : "even";
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
                odd_even = !odd_even;
            })
            tbl_search_results.append(tbl_body);
            tbl_search_results.show();
            btn_fca_tagging.show();
            btn_summary.show();
            frm_qa.show();
        });
    });

    btn_fca_tagging.click(function (e) {
        e.preventDefault();

        tbl_search_results.hide();
        tbl_search_results.empty();
        head_html = "<thead><tr><th>URL</th><th>Title</th><th>Suspect?</th><th>Type of Crime</th><th>Certainty</th></tr></thead>";
        tbl_search_results.html(head_html);
        var tbl_body = $("#tbl_search_results tbody");
        if (tbl_body.length) {
            tbl_body.remove();
        }

        $.ajax({
            url: "http://localhost:8000/cdd_with_llm/fca_tagging",
            data: {
                "company_name": company_name,
                "lang": lang,
                "llm_provider": llm_provider,
            },
            type: "GET",
            dataType: "json",
        }).done(function (json_str) {
            // console.log(json_str);
            // console.log(search_results);
            // console.log(JSON.parse(search_results).concat(JSON.parse(json_str)));
            obj = {};
            for (var i in search_results) {
                // dat = {}
                // $.extend(dat, JSON.parse(search_results[i]), JSON.parse(json_str[i]))
                obj.append({...JSON.parse(search_results[i]), ...JSON.parse(json_str)})
            }
            console.log(obj)
            // var data = {}
            // $.extend(data, JSON.parse(search_results), JSON.parse(json_str))
            // console.log(data)

            // var odd_even = false;
            // var tbl_body = document.createElement("tbody");
            // $.each(data, function () {
            //     var tbl_row = tbl_body.insertRow();
            //     tbl_row.className = odd_even ? "odd" : "even";
            //     $.each(this, function (k, v) {
            //         var cell = tbl_row.insertCell();
            //         if (k.toString() == "url") {
            //             var a = document.createElement('a');
            //             var linkText = document.createTextNode(v.toString());
            //             a.appendChild(linkText);
            //             a.href = v.toString();
            //             cell.appendChild(a);
            //         } else {
            //             cell.appendChild(document.createTextNode(v.toString()));
            //         }
            //     });
            //     odd_even = !odd_even;
            // })
            // tbl_search_results.append(tbl_body);
            // tbl_search_results.show();
        });
    });

    btn_summary.click(function (e) {
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
        }).done(function (json_str) {
            p_summary.show();
            p_summary.text(JSON.parse(json_str));
        });
    });

    frm_qa.submit(function (e) {
        e.preventDefault();
        query = $("#query").val();
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
        }).done(function (json_str) {
            p_qa.show();
            p_qa.text(JSON.parse(json_str));
        });
    });
});