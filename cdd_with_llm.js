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

function addLink(txt) {
    const html = txt.replace(/(?:\r\n|\r|\n)/g, " <br>");
    const rURL = /(https?:\/\/[^\s]+)/g;
    return html.replace(rURL, function (url) {
        return "<a href=\"" + url + "\" target=\"_blank\">" + url + "</a>";
    })
}

const viDeploy = false;


$(document).ready(function () {
    const divAjaxInfo = $("#div_ajax_info");

    const frmWebSearch = $("#frm_web_search");
    const divSearchRes = $("#div_search_res");

    const divOper = $("#div_operation");
    const btnCrawler = $("#btn_crawler");
    const btnTagging = $("#btn_tagging");
    const btnSummary = $("#btn_summary");
    const btnQA = $("#btn_qa");

    const frmCrawler = $("#frm_crawler");
    const btnCrawlerSubmit = $("#btn_crawler_submit");

    const frmTagging = $("#frm_tagging");
    const btnTaggingSubmit = $("#btn_tagging_submit");

    const frmSummary = $("#frm_summary");
    const btnSummarySubmit = $("#btn_summary_submit");
    const divSummaryRes = $("#div_summary_res");

    const frmQA = $("#frm_qa");
    const taQAQuery = $("#ta_qa_query");
    const btnQASubmit = $("#btn_qa_submit");
    const divQARes = $("#div_qa_res");

    let apiHost;
    if (viDeploy) {
        $(".vi_deploy").remove();
        apiHost = "https://tf02:8000/";
    } else {
        apiHost = "http://localhost:8000/";
    }

    function adjResTable() {
        const tblSearchRes = $("#tbl_search_res");

        tblSearchRes.removeClass();
        tblSearchRes.removeAttr("border");
        tblSearchRes.addClass("table table-striped table-hover");

        tblSearchRes.find("tbody tr td:nth-child(1)").each(function () {
            $(this).html("<a href='" + $(this).text() + "' target='_blank'>" + 
            (new URL($(this).text())).hostname + "</a>");

        });
        tblSearchRes.find("tbody tr td:nth-child(2)").each(function () {
            const htmlTags = /(<([^>]+)>)/ig;
            $(this).text($(this).text().replace(htmlTags, ""));
        });
        tblSearchRes.find("tbody tr td:nth-child(3)").each(function () {
            if ($(this).text() == "True") {
                $(this).html("&#10004;");
            } else {
                $(this).html("&#10006;");
            }
        });
    }

    frmWebSearch.on("submit", function (e) {
        e.preventDefault();

        let compnanyName;
        if (viDeploy) {
            compnanyName = new URLSearchParams(window.location.search).get("company_name");
        } else {
            compnanyName = $("#company_name").val();
        }
        const lang = $("#lang").val();

        divSearchRes.hide();
        divSearchRes.empty();

        divOper.hide();

        divSummaryRes.hide();
        divSummaryRes.empty();

        divQARes.hide();
        divQARes.empty();

        if (lang == "zh-CN") {
            taQAQuery.val(compnanyName +
                "有哪些负面新闻？总结不超过3条主要的，每条独立一行列出，并给出信息出处的URL");
        } else if (lang == "zh-HK" || lang == "zh-TW") {
            taQAQuery.val(compnanyName +
                "有哪些負面新聞？總結不超過3條主要的，每條獨立一行列出，並給出資訊出處的URL");
        } else if (lang == "ja-JP") {
            taQAQuery.val(compnanyName +
                "に関するネガティブなニュースをサーチしなさい。一番大事なものを三つ以内にまとめ、それぞれを箇条書きし、出典元URLを付けなさい");
        } else {
            taQAQuery.val("What is the negative news about " + compnanyName +
                "? Summarize no more than 3 major ones, list each on a separate line, and give the URL where the information came from.");
        }

        const frmData = $(this).serializeArray();
        if (viDeploy) {
            frmData.push({ "name": "company_name", "value": compnanyName });
        }

        $.ajax({
            url: apiHost + "cdd_with_llm/web_search",
            data: frmData,
            type: "GET",
            xhrFields: {
                withCredentials: true
            },
            beforeSend: function () {
                divAjaxInfo.html("<span class='spinner-border spinner-border-sm me-2'></span> \
                Making web search... may take some time");
                divAjaxInfo.show();
            },
            complete: function () {
                divAjaxInfo.hide();
                divAjaxInfo.empty();
            },
        }).done(function (html) {
            divSearchRes.html(html);
            adjResTable();
            divSearchRes.show();

            divOper.show();
            btnCrawler.removeClass("disabled");
        });
    });

    btnCrawlerSubmit.on("click", function (e) {
        e.preventDefault();
        // divSearchRes.hide();
        // divSearchRes.empty();

        $.ajax({
            url: apiHost + "cdd_with_llm/contents_crawler",
            data: frmCrawler.serializeArray({ checkboxesAsBools: true }),
            type: "GET",
            xhrFields: {
                withCredentials: true
            },
            beforeSend: function () {
                divAjaxInfo.html("<span class='spinner-border spinner-border-sm me-2'></span> \
                Grabbing web conetents from each url... may take some time");
                divAjaxInfo.show();
            },
            complete: function () {
                divAjaxInfo.hide();
                divAjaxInfo.empty();
            },
        }).done(function (html) {
            divSearchRes.html(html);
            adjResTable();
            divSearchRes.show();

            btnTagging.removeClass("disabled");
            btnSummary.removeClass("disabled");
            btnQA.removeClass("disabled");
            btnCrawler.addClass("disabled");
        });
    });

    btnTaggingSubmit.on("click", function (e) {
        e.preventDefault();

        if ($("#tbl_search_res tr th").length == 5) {
            $("#tbl_search_res td:nth-child(4),th:nth-child(4),td:nth-child(5),th:nth-child(5)").remove();
        }

        $.ajax({
            url: apiHost + "cdd_with_llm/fc_tagging",
            data: frmTagging.serializeArray({ checkboxesAsBools: true }),
            type: "GET",
            xhrFields: {
                withCredentials: true
            },
            beforeSend: function () {
                divAjaxInfo.html("<span class='spinner-border spinner-border-sm me-2'></span> \
                Tagging for each news... may take some time");
                divAjaxInfo.show();
            },
            complete: function () {
                divAjaxInfo.hide();
                divAjaxInfo.empty();
            },
        }).done(function (html) {
            divSearchRes.html(html);
            adjResTable();
        });
    });

    btnSummarySubmit.on("click", function (e) {
        e.preventDefault();
        divSummaryRes.hide();
        divSummaryRes.empty();

        $.ajax({
            url: apiHost + "cdd_with_llm/summary",
            data: frmSummary.serializeArray({ checkboxesAsBools: true }),
            type: "GET",
            xhrFields: {
                withCredentials: true
            },
            beforeSend: function () {
                divAjaxInfo.html("<span class='spinner-border spinner-border-sm me-2'></span> \
                Making summary for these news... may take some time");
                divAjaxInfo.show();
            },
            complete: function () {
                divAjaxInfo.hide();
                divAjaxInfo.empty();
            },
        }).done(function (txt) {
            divSummaryRes.html("<strong>Summary:</strong> <br>");
            divSummaryRes.append(addLink(txt));
            divSummaryRes.show();
        });
    });

    btnQASubmit.on("click", function (e) {
        e.preventDefault();

        $.ajax({
            url: apiHost + "cdd_with_llm/qa",
            data: frmQA.serializeArray({ checkboxesAsBools: true }),
            type: "GET",
            xhrFields: {
                withCredentials: true
            },
            beforeSend: function () {
                divAjaxInfo.html("<span class='spinner-border spinner-border-sm me-2'></span> \
                Making question-answering on these news... may take some time");
                divAjaxInfo.show();
            },
            complete: function () {
                divAjaxInfo.hide();
                divAjaxInfo.empty();
            },
        }).done(function (txt) {
            divQARes.append("<strong>Question:</strong> <br>");
            divQARes.append(document.createTextNode(taQAQuery.val()));
            divQARes.append("<br>");
            divQARes.append("<strong>Answer:<strong> <br>");
            divQARes.append(addLink(txt));
            divQARes.append("<br><br>");
            divQARes.show();
        });
    });
});