define("jira/dropdown",["require"],function(i){function o(){try{return Boolean(n.require("jira/dialog/dialog").current)}catch(i){}return!1}var n=i("jira/util/top-same-origin-window")(window),t=i("jira/ajs/layer/layer-interactions"),e=i("jquery"),d=i("jira/util/key-code"),r=[],s={current:null,addInstance:function(){r.push(this)},hideInstances:function(){var i=this;e(r).each(function(){i!==this&&this.hideDropdown()})},getHash:function(){return this.hash||(this.hash={container:this.dropdown,hide:this.hideDropdown,show:this.displayDropdown}),this.hash},displayDropdown:function(){if(this.current!==this){this.hideInstances(),this.current=this,this.dropdown.css({display:"block"}),this.displayed=!0;var i=this.dropdown;o()||setTimeout(function(){var o=e(window),n=i.offset().top+i.prop("offsetHeight")-o.height()+10;o.scrollTop()<n&&e("html,body").animate({scrollTop:n},300,"linear")},100)}},hideDropdown:function(){!1!==this.displayed&&(this.current=null,this.dropdown.css({display:"none"}),this.displayed=!1)},init:function(i,o){var n=this;this.addInstance(this),this.dropdown=e(o),this.dropdown.css({display:"none"}),e(document).keydown(function(i){i.keyCode===d.TAB&&n.hideDropdown()}),i.target?e.aop.before(i,function(){n.displayed||n.displayDropdown()}):(n.dropdown.css("top",e(i).outerHeight()+"px"),i.click(function(i){n.displayed?n.hideDropdown():(n.displayDropdown(),i.stopPropagation()),i.preventDefault()})),e(document.body).click(function(){n.displayed&&n.hideDropdown()})}};return t.preventDialogHide(s),t.hideBeforeDialogShown(s),s}),AJS.namespace("JIRA.Dropdown",null,require("jira/dropdown")),AJS.namespace("jira.widget.dropdown",null,require("jira/dropdown"));