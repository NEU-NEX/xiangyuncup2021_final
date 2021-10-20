AJS.test.require(["jira.webresources:viewissue"],function(){"use strict";var t=require("jquery"),a=require("underscore"),e=require("jira/util/strings");module("ViewIssueAnalytics",{setup:function(){var a=this;this.sandbox=sinon.sandbox.create(),this.context=AJS.test.mockableModuleContext(),this.issueContext="unknown",this.fakeSender=this.sandbox.spy(),this.context.mock("jira/analytics",{send:a.fakeSender}),this.context.mock("jira/viewissue/analytics-utils",{context:function(){return a.issueContext}}),this.analytics=this.context.require("jira/viewissue/tabs/analytics"),this.container=t("<div class='tabwrap'>")},teardown:function(){this.sandbox.restore()},getTabLink:function(t){return this.container.find("a.tab-link:eq("+t+")")},addAdditionalButton:function(e){e=a.extend({sort:!0,order:"asc"},e);var i=t("<a>");return e.sort&&i.attr({"data-tab-sort":"","data-order":e.order}),i.appendTo(this.container),i},addTab:function(e){e=a.extend({id:"com.atlassian.jira.plugin.system.issuetabpanels:all-tabpanel",active:!1,href:"href"},e);var i=t("<li>").attr({"data-key":e.id,"data-href":e.href,class:e.active?"active":""});i.appendTo(this.container),t("<a class='tab-link'>").appendTo(i)},assertSortButtonEvent:function(t,a){this.assertEventCall("jira.viewissue.tabsort.clicked",t,a)},assertTabEvent:function(t,a){this.assertEventCall("jira.viewissue.tab.clicked",t,a)},assertEventCall:function(t,a,e){var i=this.fakeSender.getCall(a),s=i&&i.args[0];deepEqual(s,{name:t,properties:e})},defaultEventData:function(t){return a.extend({inNewWindow:!1,keyboard:!1,context:this.issueContext,tabPosition:0,tab:"com.atlassian.jira.plugin.system.issuetabpanels:all-tabpanel"},t)}}),test("Should send data after click on tab",function(){this.addTab(),this.analytics.tabClicked(this.getTabLink(0),!1,!1),this.assertTabEvent(0,this.defaultEventData())}),test("Should pass additional parameters after click on tab",function(){this.addTab(),this.analytics.tabClicked(this.getTabLink(0),!0,!1),this.assertTabEvent(0,this.defaultEventData({inNewWindow:!0})),this.analytics.tabClicked(this.getTabLink(0),!1,!0),this.assertTabEvent(1,this.defaultEventData({keyboard:!0}))}),test("Should send data about active tab after click on sort button",function(){this.addTab({active:!1}),this.addTab({id:"com.atlassian.jira.plugin.system.issuetabpanels:worklog-tabpanel",active:!0});var t=this.addAdditionalButton();this.analytics.buttonClicked(t,!1,!1),this.assertSortButtonEvent(0,this.defaultEventData({tab:"com.atlassian.jira.plugin.system.issuetabpanels:worklog-tabpanel",tabPosition:1,order:"asc"}))}),test("Should send data about sort order after click on sort button",function(){this.addTab({active:!0});var t=this.addAdditionalButton({order:"desc"});this.analytics.buttonClicked(t,!1,!1),this.assertSortButtonEvent(0,this.defaultEventData({order:"desc"}))}),test("Should whitelist tab name after click on tab",function(){this.addTab({id:"non-whitelisted-tab"}),this.analytics.tabClicked(this.getTabLink(0),!1,!1),this.assertTabEvent(0,this.defaultEventData({tab:e.hashCode("non-whitelisted-tab")}))}),test("Should whitelist tab name after click on sort button",function(){this.addTab({id:"non-whitelisted-tab",active:!0});var t=this.addAdditionalButton();this.analytics.buttonClicked(t,!1,!1),this.assertSortButtonEvent(0,this.defaultEventData({tab:e.hashCode("non-whitelisted-tab"),order:"asc"}))}),test("Should do not send analytics after click on something which is not tab or sort",function(){this.addTab({active:!0});var t=this.addAdditionalButton({sort:!1});this.analytics.buttonClicked(t,!1,!1),ok(!this.fakeSender.called)}),test("Should send position of tab",function(){this.addTab({id:"tab-one"}),this.addTab({id:"tab-two"}),this.addTab({id:"tab-three"}),this.analytics.tabClicked(this.getTabLink(2),!1,!1),this.analytics.tabClicked(this.getTabLink(0),!1,!1),this.analytics.tabClicked(this.getTabLink(1),!1,!1),ok(this.fakeSender.calledThrice),this.assertTabEvent(0,this.defaultEventData({tabPosition:2,tab:e.hashCode("tab-three")})),this.assertTabEvent(1,this.defaultEventData({tabPosition:0,tab:e.hashCode("tab-one")})),this.assertTabEvent(2,this.defaultEventData({tabPosition:1,tab:e.hashCode("tab-two")}))})});