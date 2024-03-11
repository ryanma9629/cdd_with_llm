(function () {
    "use strict";

    const vi_deploy = false;
    const html_tags = /(<([^>]+)>)/ig;

    function txt2html(txt) {
        const html = txt.replace(/(?:\r\n|\r|\n)/g, " <br><br>");
        const urlReg = /(https?:\/\/[^\s]+)/g;
        return html.replace(urlReg, function (url) {
            return "<a href=\"" + url + "\" target=\"_blank\">" + url + "</a>";
        })
    }

    function uuidv4() {
        return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
            (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
        );
    }

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

    $(document).ready(function () {
        const div_ajax = $("#div_ajax");
        const p_ajax = $("#p_ajax");

        const frm_web_search = $("#frm_web_search");
        const div_search_results = $("#div_search_results");

        const div_operation = $("#div_operation");
        const btn_crawler = $("#btn_crawler");
        const btn_tagging = $("#btn_tagging");
        const btn_summary = $("#btn_summary");
        const btn_qa = $("#btn_qa");

        const div_crawler_frm = $("#div_crawler_frm");
        const frm_cralwer = $("#frm_crawler");

        const div_tagging_frm = $("#div_tagging_frm");
        const frm_tagging = $("#frm_tagging");

        const div_summary_frm = $("#div_summary_frm");
        const frm_summary = $("#frm_summary");

        const div_qa_frm = $("#div_qa_frm");
        const frm_qa = $("#frm_qa");
        const qa_query = $("#qa_query");

        const div_summary = $("#div_summary");
        const p_summary = $("#p_summary");

        const div_answer = $("#div_answer");
        const p_question = $("#p_question");
        const p_answer = $("#p_answer");

        let api_host;
        if (vi_deploy) {
            $(".vi_deploy").remove();
            api_host = "https://tf02:8000/";
        } else {
            api_host = "http://localhost:8000/";
        }
        // const hostReg = new RegExp(/https?:\/\/[^/]+/);

        // const userid = localStorage.getItem("userid");
        // if (userid == null) {
        //     userid = uuidv4();
        //     localStorage.setItem("userid", userid);
        // }

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
                    const text_with_tags = $(this).text();
                    const text_wo_tags = text_with_tags.replace(html_tags, "");
                    $(this).text(text_wo_tags);
                });

                div_search_results.show();
                div_operation.show();
                btn_crawler.prop("disabled", false);
            });
        });

        frm_cralwer.on("submit", function (e) {
            e.preventDefault();
            div_crawler_frm.hide();

            $.ajax({
                url: api_host + "cdd_with_llm/contents_from_crawler",
                data: $(this).serializeArray({ checkboxesAsBools: true }),
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
                    const text_with_tags = $(this).text();
                    const text_wo_tags = text_with_tags.replace(html_tags, "");
                    $(this).text(text_wo_tags);
                });

                div_operation.show();
                btn_tagging.prop("disabled", false);
                btn_summary.prop("disabled", false);
                btn_qa.prop("disabled", false);
                btn_crawler.prop("disabled", true);
            });
        });

        frm_tagging.on("submit", function (e) {
            e.preventDefault();
            div_tagging_frm.hide();

            if ($("#tbl_search_results tr th").length == 5) {
                $("#tbl_search_results td:nth-child(4),th:nth-child(4),td:nth-child(5),th:nth-child(5)").remove();
            }

            $.ajax({
                url: api_host + "cdd_with_llm/fc_tagging",
                data: $(this).serializeArray({ checkboxesAsBools: true }),
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
                    const text_with_tags = $(this).text();
                    const text_wo_tags = text_with_tags.replace(html_tags, "");
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
                url: api_host + "cdd_with_llm/summary",
                data: $(this).serializeArray({ checkboxesAsBools: true }),
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
                url: api_host + "cdd_with_llm/qa",
                data: $(this).serializeArray({ checkboxesAsBools: true }),
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
})();