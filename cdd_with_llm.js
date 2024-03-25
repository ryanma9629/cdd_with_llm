"use strict";

(function ($) {
    $.fn.serialize = function (options) {
        return $.param(this.serializeArray(options));
    };

    $.fn.serializeArray = function (options) {
        const o = $.extend({
            checkboxesAsBools: false
        }, options || {});

        const rselectTextarea = /select|textarea/i;
        const rinput = /text|hidden|password|search|number/i;

        return this.map(function () {
            return this.elements ? $.makeArray(this.elements) : this;
        })
            .filter(function () {
                return this.name && !this.disabled &&
                    (this.checked
                        || (o.checkboxesAsBools && this.type === 'checkbox')
                        || rselectTextarea.test(this.nodeName)
                        || rinput.test(this.type));
            })
            .map(function (i, elem) {
                const val = $(this).val();
                return val == null ?
                    null :
                    $.isArray(val) ?
                        $.map(val, function (val, i) {
                            return { name: elem.name, value: val };
                        }) :
                        {
                            name: elem.name,
                            value: (o.checkboxesAsBools && this.type === 'checkbox') ?
                                (this.checked ? true : false) :
                                val
                        };
            }).get();
    };
})(jQuery);

function txt2html(txt) {
    const html = txt.replace(/(?:\r\n|\r|\n)/g, " <br>");
    const rURL = /(https?:\/\/[^\s]+)/g;
    return html.replace(rURL, function (url) {
        return "<a href=\"" + url + "\" target=\"_blank\">" + url + "</a>";
    })
}

const vi_deploy = false;
const html_tags = /(<([^>]+)>)/ig;

$(document).ready(function () {
    const div_ajax = $("#div_ajax");

    const frm_web_search = $("#frm_web_search");
    const div_search_results = $("#div_search_results");

    const div_operation = $("#div_operation");
    const btn_crawler = $("#btn_crawler");
    const btn_tagging = $("#btn_tagging");
    const btn_summary = $("#btn_summary");
    const btn_qa = $("#btn_qa");

    const div_crawler_frm = $("#div_crawler_frm");
    const frm_cralwer = $("#frm_crawler");
    const btn_crawler_submit = $("#btn_crawler_submit");

    const div_tagging_frm = $("#div_tagging_frm");
    const frm_tagging = $("#frm_tagging");
    const btn_tagging_submit = $("#btn_tagging_submit");

    const div_summary_frm = $("#div_summary_frm");
    const frm_summary = $("#frm_summary");
    const btn_summary_submit = $("#btn_summary_submit");
    const div_summary = $("#div_summary");

    const div_qa_frm = $("#div_qa_frm");
    const frm_qa = $("#frm_qa");
    const qa_query = $("#qa_query");
    const btn_qa_submit = $("#btn_qa_submit");
    const div_qa = $("#div_qa");

    let api_host;
    if (vi_deploy) {
        $(".vi_deploy").remove();
        api_host = "https://tf02:8000/";
    } else {
        api_host = "http://localhost:8000/";
    }

    function adj_tbl() {
        const tbl_search_results = $("#tbl_search_results");
        tbl_search_results.removeClass();
        tbl_search_results.removeAttr("border");
        tbl_search_results.addClass("table table-striped table-hover");

        tbl_search_results.find("tbody tr td:nth-child(1)").each(function () {
            const long_url = $(this).text();
            const short_url = (new URL(long_url)).hostname;
            $(this).html("<a href='" + long_url + "' target='_blank'>" + short_url + "</a>");

        });
        tbl_search_results.find("tbody tr td:nth-child(2)").each(function () {
            const text_with_tags = $(this).text();
            const text_wo_tags = text_with_tags.replace(html_tags, "");
            $(this).text(text_wo_tags);
        });
        tbl_search_results.find("tbody tr td:nth-child(3)").each(function () {
            const cell_text = $(this).text();
            if (cell_text == "True") {
                $(this).html("&#10004;");
            } else {
                $(this).html("&#10006;");
            }
        });
    }

    frm_web_search.on("submit", function (e) {
        e.preventDefault();

        let company_name;
        if (vi_deploy) {
            company_name = new URLSearchParams(window.location.search).get("company_name");
        } else {
            company_name = $("#company_name").val();
        }
        const lang = $("#lang").val();

        div_search_results.hide();
        div_search_results.empty();

        div_operation.hide();

        div_summary.hide();
        div_summary.empty();

        div_qa.hide();
        div_qa.empty();

        if (lang == "zh-CN") {
            qa_query.val(company_name +
                "有哪些负面新闻？总结不超过3条主要的，每条独立一行列出，并给出信息出处的URL");
        } else if (lang == "zh-HK" || lang == "zh-TW") {
            qa_query.val(company_name +
                "有哪些負面新聞？總結不超過3條主要的，每條獨立一行列出，並給出資訊出處的URL");
        } else if (lang == "ja-JP") {
            qa_query.val(company_name +
                "に関するネガティブなニュースをサーチしなさい。一番大事なものを三つ以内にまとめ、それぞれを箇条書きし、出典元URLを付けなさい");
        } else {
            qa_query.val("What is the negative news about " + company_name +
                "? Summarize no more than 3 major ones, list each on a separate line, and give the URL where the information came from.");
        }

        const form_data = $(this).serializeArray();
        if (vi_deploy) {
            form_data.push({ "name": "company_name", "value": company_name });
        }

        $.ajax({
            url: api_host + "cdd_with_llm/web_search",
            data: form_data,
            type: "GET",
            xhrFields: {
                withCredentials: true
            },
            beforeSend: function () {
                div_ajax.html("<span class='spinner-border spinner-border-sm me-2'></span> \
                Making web search... may take some time");
                div_ajax.show();
            },
            complete: function () {
                div_ajax.hide();
                div_ajax.empty();
            },
        }).done(function (html) {
            div_search_results.html(html);
            adj_tbl();
            div_search_results.show();

            div_operation.show();
            btn_crawler.removeClass("disabled");
        });
    });

    btn_crawler_submit.on("click", function (e) {
        e.preventDefault();
        div_crawler_frm.hide();

        $.ajax({
            url: api_host + "cdd_with_llm/contents_from_crawler",
            data: frm_cralwer.serializeArray({ checkboxesAsBools: true }),
            type: "GET",
            xhrFields: {
                withCredentials: true
            },
            beforeSend: function () {
                div_ajax.html("<span class='spinner-border spinner-border-sm me-2'></span> \
                Grabbing web conetents from each url... may take some time");
                div_ajax.show();
            },
            complete: function () {
                div_ajax.hide();
                div_ajax.empty();
            },
        }).done(function (html) {
            div_search_results.hide();
            div_search_results.empty();
            div_search_results.html(html);
            adj_tbl();
            div_search_results.show();

            btn_tagging.removeClass("disabled");
            btn_summary.removeClass("disabled");
            btn_qa.removeClass("disabled");
            btn_crawler.addClass("disabled");
        });
    });

    btn_tagging_submit.on("click", function (e) {
        e.preventDefault();
        div_tagging_frm.hide();

        if ($("#tbl_search_results tr th").length == 5) {
            $("#tbl_search_results td:nth-child(4),th:nth-child(4),td:nth-child(5),th:nth-child(5)").remove();
        }

        $.ajax({
            url: api_host + "cdd_with_llm/fc_tagging",
            data: frm_tagging.serializeArray({ checkboxesAsBools: true }),
            type: "GET",
            xhrFields: {
                withCredentials: true
            },
            beforeSend: function () {
                div_ajax.html("<span class='spinner-border spinner-border-sm me-2'></span> \
                Tagging for each news... may take some time");
                div_ajax.show();
            },
            complete: function () {
                div_ajax.hide();
                div_ajax.empty();
            },
        }).done(function (html) {
            div_search_results.hide();
            div_search_results.empty();
            div_search_results.html(html);
            adj_tbl();
            div_search_results.show();
        });
    });

    btn_summary_submit.on("click", function (e) {
        e.preventDefault();
        div_summary_frm.hide();
        div_summary.hide();
        div_summary.empty();

        $.ajax({
            url: api_host + "cdd_with_llm/summary",
            data: frm_summary.serializeArray({ checkboxesAsBools: true }),
            type: "GET",
            xhrFields: {
                withCredentials: true
            },
            beforeSend: function () {
                div_ajax.html("<span class='spinner-border spinner-border-sm me-2'></span> \
                Making summary for these news... may take some time");
                div_ajax.show();
            },
            complete: function () {
                div_ajax.hide();
                div_ajax.empty();
            },
        }).done(function (txt) {
            div_summary.html("<strong>Summary:</strong> <br>");
            div_summary.append(txt2html(txt));
            div_summary.show();
        });
    });

    btn_qa_submit.on("click", function (e) {
        e.preventDefault();
        div_qa_frm.hide();

        $.ajax({
            url: api_host + "cdd_with_llm/qa",
            data: frm_qa.serializeArray({ checkboxesAsBools: true }),
            type: "GET",
            xhrFields: {
                withCredentials: true
            },
            beforeSend: function () {
                div_ajax.html("<span class='spinner-border spinner-border-sm me-2'></span> \
                Making question-answering on these news... may take some time");
                div_ajax.show();
            },
            complete: function () {
                div_ajax.hide();
                div_ajax.empty();
            },
        }).done(function (txt) {
            div_qa.append("<strong>Question:</strong> <br>");
            div_qa.append(qa_query.val());
            div_qa.append("<br>");
            div_qa.append("<strong>Answer:<strong> <br>");
            div_qa.append(txt2html(txt));
            div_qa.append("<br><br>");
            div_qa.show();
        });
    });
});