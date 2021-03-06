define('jira/bigpipe/element', ['jquery', 'wrm/data', 'jira/skate', 'jira/util/logger'], function ($, wrmData, skate, logger) {

    /**
     * An element that represents a parallel rendering of HTML and its eventual
     * delivery to the front-end.
     * @fires success - when the HTML is successfully rendered on server, passed to the browser,
     * parsed by the browser, and added to the page.
     * @fires error - if anything at all goes wrong.
     */
    return skate('big-pipe', {

        attached: function attached(element) {

            function successCallback() {
                var event = new CustomEvent('success');
                element.dispatchEvent(event);
            }

            function errorCallback(e, errorSignature) {
                var event = new CustomEvent('error');
                event.data = {
                    event: e,
                    signature: errorSignature
                };
                element.dispatchEvent(event);
            }

            function dataArrived(data) {
                try {
                    var parsedHtml = $(data);
                    var $newDom = $(element).replaceWith(parsedHtml);
                    // APDEX-1370 - temporarily force synchronous initialisation instead of async :(
                    $newDom.each(function () {
                        skate.init(this);
                    });
                    mark("end");
                    successCallback();
                } catch (e) {
                    logger.error('Error while parsing html: ' + e);
                    dataError(e, "parsing");
                }
            }

            function dataError(e, errorSignature) {
                mark("error");
                errorCallback(e, errorSignature);
            }

            function mark(name) {
                'performance' in window && performance.mark && performance.mark(markPrefix + name);
            }

            var pipeId = element.getAttribute('data-id');
            if (pipeId === null) {
                logger.error('No data-id attribute provided for tag <big-pipe/> for element:', element);
                dataError({
                    name: "NoPipeIdError",
                    message: "Unable to render element. Element does not contain a pipe id.",
                    element: element
                }, "no.pipe.id");
                return;
            }

            var markPrefix = "bigPipe." + pipeId + ".";
            mark("start");

            // APDEX-1370 - temporarily force synchronous initialisation instead of async :(
            var data = wrmData.claim(pipeId);
            if (data) {
                dataArrived(data);
            } else {
                dataError({ name: "NoDataError", message: "BigPipe response is empty." }, "no.data");
            }
        },

        detached: function detached() {},

        type: skate.type.ELEMENT,

        resolvedAttribute: 'resolved',
        unresolvedAttribute: 'unresolved'
    });
});