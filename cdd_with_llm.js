$(document).ready(function () {
    $("#frm_web_search").submit(function (e) {
        e.preventDefault();
        tbl = $("#tbl_search_results")
        tbl.empty()
        $.ajax({
            url: "http://localhost:7980/cdd_with_llm/web_search",
            data: $(this).serialize(),
            type: "GET",
            dataType: "json",
        })
            .done(function (json_obj) {
                // $("#search_results").text(JSON.stringify(json))
                var tbl_body = document.createElement("tbody");
                var odd_even = false;
                // var data = JSON.stringify(json)
                $.each(JSON.parse(json_obj), function () {
                    var tbl_row = tbl_body.insertRow();
                    tbl_row.className = odd_even ? "odd" : "even";
                    $.each(this, function (k, v) {
                        var cell = tbl_row.insertCell();
                        cell.appendChild(document.createTextNode(v.toString()));
                    });
                    odd_even = !odd_even;
                })
                tbl.append(tbl_body);
            });
        $("#p_ajax_msg")
            .ajaxStart(function () {
                $(this).show();
            })
            .ajaxStop(function () {
                $(this).hide();
            });
    });
})