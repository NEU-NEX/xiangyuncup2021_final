AJS.test.require(["jira.webresources:key-commands"],function(){function e(e,t){r(t||document).trigger({type:"keydown",keyCode:e,which:e})}function t(e,t){r(t||document).trigger({type:"keypress",which:e.charCodeAt(0)})}function n(e){var t=0;this.valueOf=function(){return t},r(document).bind("shortcut",e,function(){t++})}var r=require("jquery");module("Sequenced Shortcut Events",{teardown:function(){r(document).unbind("shortcut")}}),test('"shortcut" event fires for single character shortcut',function(){var e=new n("x");t("x"),equal(e.valueOf(),1,'Shortcut "x" event fires for input stream [x]')}),test('"shortcut" event fires for multi-character shortcut sequence',function(){var e=new n("xyz");t("x"),t("y"),t("z"),equal(e.valueOf(),1,'Shortcut "xyz" event fires for input stream [x,y,z]')}),test('"shortcut" event does not fire for non-matching character sequences',function(){var e=new n("xyz");t("z"),t("x"),t("z"),t("x"),t("y"),t("y"),t("z"),equal(e.valueOf(),0,'Shortcut "xyz" event does not fire for input stream [z,x,z,x,y,y,z]')}),test("Conflicting shortcut sequences do not throw errors",function(){r(document).bind("shortcut","xyz",r.noop),r(document).bind("shortcut","xy",r.noop),r(document).bind("shortcut","y",r.noop),r(document).bind("shortcut","xyzy",r.noop),expect(0)}),test('"shortcut" event only fires for longest matching sequence',function(){var e=new n("yz"),r=new n("xyz");t("x"),t("y"),t("z"),equal(e.valueOf(),0,'Shortcut "yz" event does not fire for input stream [x,y,z]'),equal(r.valueOf(),1,'Shortcut "xyz" event fires for input stream [x,y,z]')}),test("Input stream resets after a completed shortcut",function(){var e=new n("zy"),r=new n("xyz");t("z"),t("x"),t("y"),t("z"),t("y"),equal(e.valueOf(),0,'Shortcut "zy" does not fire for input stream [z,x,y,z,y]'),equal(r.valueOf(),1,'Shortcut "xyz" fires for input stream [z,x,y,z,y]')}),test("Input stream resets after a non-character key is pressed",function(){var r=new n("xy");t("x"),e(17),e(89),t("y"),equal(r.valueOf(),0,"CTRL key resets input stream");var o=new n("xz");t("x"),e(16),e(90),t("z"),equal(o.valueOf(),1,"SHIFT key does not reset input stream")}),test('Key events targeted at form element do not fire "shortcut" events',function(){var e=new n("xy"),r=document.getElementById("qunit-fixture").appendChild(document.createElement("textarea"));t("x",document),t("y",r),t("x",r),t("y",r),equal(e.valueOf(),0,'Key events targeted at <textarea> do not fire "shortcut" event'),t("x",r),t("y",document),equal(e.valueOf(),0,"Key events targeted at <textarea> reset the character input stream")})});