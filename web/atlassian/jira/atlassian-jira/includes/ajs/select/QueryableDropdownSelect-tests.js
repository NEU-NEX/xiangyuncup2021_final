AJS.test.require(["jira.webresources:select-pickers"], function () {
    "use strict";

    var jQuery = require("jquery");
    var Deferred = require("jira/jquery/deferred");
    var QueryableDropdownSelect = require("jira/ajs/select/queryable-dropdown-select");

    module("QueryableDropdownSelect");

    test("Should get results before rendering dropdown (such that positioning won't be borked)", function () {
        var sandbox = this;
        var qdds = new QueryableDropdownSelect({
            element: jQuery("<ul></ul>")
        });

        sandbox.stub(qdds, "getQueryVal", function () {
            return "one";
        });
        sandbox.stub(qdds, 'requestSuggestions', function () {
            return new Deferred().resolve(["one", "two"]).promise();
        });

        var genList = sandbox.spy(qdds.listController, "generateListFromJSON");
        var showDropdown = sandbox.spy(qdds.dropdownController, "show");
        var positionDropdown = sandbox.spy(qdds.dropdownController, "setPosition");

        qdds.onEdit();

        ok(genList.calledBefore(showDropdown), "should have results before we render a dropdown with them in it (to prevent things like TF-39)");
        ok(genList.calledBefore(positionDropdown), "should have results before we calc and position a dropdown with them in it (to prevent things like TF-39)");
    });

    test("ariaLabel", function () {
        var qdds = new QueryableDropdownSelect({
            element: jQuery("<ul></ul>"),
            ariaLabel: "Test Label"
        });

        equal(qdds.$field.attr("aria-label"), "Test Label", "aria-label attribute should be set from options");
    });
});