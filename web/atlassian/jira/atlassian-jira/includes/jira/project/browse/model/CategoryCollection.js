define('jira/project/browse/categorycollection', ['jira/backbone-1.3.3'], function (Backbone) {
    'use strict';

    return Backbone.Collection.extend({
        unselect: function unselect() {
            this.filter(function (category) {
                return category.get('selected');
            }).forEach(function (category) {
                category.set('selected', false, { silent: true });
            });
        },
        getSelected: function getSelected() {
            return this.find(function (category) {
                return category.get('selected');
            });
        },
        selectCategory: function selectCategory(id) {
            var category = this.get(id);

            if (!category) {
                return false;
            }

            this.unselect();

            category.set('selected', true);
            return category;
        }
    });
});