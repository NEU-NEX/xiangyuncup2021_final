define('jira/dialog/init-non-dashboard-dialogs', ['jira/dialog/dialog-register', 'jira/dialog/dialog-util', 'jira/dialog/form-dialog', 'jira/dialog/labels-dialog', 'jira/dialog/link-issue-dialog-factory', 'jira/dialog/screenshot-window', 'jira/issuenavigator/issue-navigator', 'jquery', 'exports'], function (DialogRegister, DialogUtil, FormDialog, LabelsDialog, LinkIssueDialogFactory, ScreenshotWindow, IssueNavigator, jQuery, exports) {
    'use strict';

    exports.init = function () {
        // Issue-related dialogs should not be active on the dashboard.
        if (document.getElementById('dashboard')) {
            return false;
        }

        DialogRegister.linkIssue = LinkIssueDialogFactory.createLinkIssueDialog(".issueaction-link-issue");

        DialogRegister.deleteIssue = new FormDialog({
            id: "delete-issue-dialog",
            trigger: ".issueaction-delete-issue",
            targetUrl: "#delete-issue-return-url",
            ajaxOptions: DialogUtil.getDefaultAjaxOptions,
            onSuccessfulSubmit: DialogUtil.storeCurrentIssueIdOnSucessfulSubmit,
            issueMsg: 'thanks_issue_deleted',
            delayShowUntil: DialogUtil.BeforeShowIssueDialogHandler.execute,
            isIssueDialog: true
        });

        DialogRegister.archiveIssue = new FormDialog({
            id: "archive-issue-dialog",
            trigger: ".issueaction-archive-issue",
            targetUrl: "#archive-issue-return-url",
            ajaxOptions: DialogUtil.getDefaultAjaxOptions,
            onSuccessfulSubmit: DialogUtil.storeCurrentIssueIdOnSucessfulSubmit,
            issueMsg: 'thanks_issue_archived',
            delayShowUntil: DialogUtil.BeforeShowIssueDialogHandler.execute,
            isIssueDialog: true
        });

        DialogRegister.restoreIssue = new FormDialog({
            id: "restore-issue-dialog",
            trigger: ".issueaction-restore-issue",
            targetUrl: "#restore-issue-return-url",
            ajaxOptions: DialogUtil.getDefaultAjaxOptions,
            onSuccessfulSubmit: DialogUtil.storeCurrentIssueIdOnSucessfulSubmit,
            issueMsg: 'thanks_issue_restored',
            delayShowUntil: DialogUtil.BeforeShowIssueDialogHandler.execute,
            isIssueDialog: true
        });

        DialogRegister.cloneIssue = new FormDialog({
            id: "clone-issue-dialog",
            trigger: ".issueaction-clone-issue",
            ajaxOptions: DialogUtil.getDefaultAjaxOptions,
            onSuccessfulSubmit: DialogUtil.storeCurrentIssueIdOnSucessfulSubmit,
            issueMsg: 'thanks_issue_cloned',
            delayShowUntil: DialogUtil.BeforeShowIssueDialogHandler.execute,
            isIssueDialog: true
        });

        DialogRegister.assignIssue = new FormDialog({
            id: "assign-dialog",
            trigger: ".issueaction-assign-issue",
            ajaxOptions: DialogUtil.getDefaultAjaxOptions,
            onSuccessfulSubmit: DialogUtil.storeCurrentIssueIdOnSucessfulSubmit,
            issueMsg: 'thanks_issue_assigned',
            delayShowUntil: DialogUtil.BeforeShowIssueDialogHandler.execute,
            isIssueDialog: true,
            widthClass: "large"
        });

        DialogRegister.assignIssueToMe = new FormDialog({
            id: "assign-dialog",
            trigger: ".issueaction-assign-to-me",
            ajaxOptions: DialogUtil.getDefaultAjaxOptions,
            onSuccessfulSubmit: DialogUtil.storeCurrentIssueIdOnSucessfulSubmit,
            issueMsg: 'thanks_issue_assigned',
            delayShowUntil: DialogUtil.BeforeShowIssueDialogHandler.execute,
            isIssueDialog: true
        });

        DialogRegister.logWork = new FormDialog({
            id: "log-work-dialog",
            trigger: ".issueaction-log-work",
            ajaxOptions: DialogUtil.getDefaultAjaxOptions,
            onSuccessfulSubmit: DialogUtil.storeCurrentIssueIdOnSucessfulSubmit,
            issueMsg: 'thanks_issue_worklogged',
            delayShowUntil: DialogUtil.BeforeShowIssueDialogHandler.execute,
            isIssueDialog: true,
            widthClass: "large"
        });

        DialogRegister.attachFile = new FormDialog({
            id: "attach-file-dialog",
            trigger: ".issueaction-attach-file",
            ajaxOptions: DialogUtil.getDefaultAjaxOptions,
            onSuccessfulSubmit: DialogUtil.storeCurrentIssueIdOnSucessfulSubmit,
            issueMsg: 'thanks_issue_attached',
            isIssueDialog: true,
            widthClass: "large"
        });

        DialogRegister.attachScreenshot = new ScreenshotWindow({
            id: "attach-screenshot-window",
            trigger: ".issueaction-attach-screenshot"
        });

        DialogRegister.manageAttachment = new FormDialog({
            id: 'manage-attachment-dialog',
            trigger: '#manage-attachment-link',
            stacked: true,
            reloadOnPop: true,
            isIssueDialog: true,
            widthClass: "large"
        });

        DialogRegister.comment = new FormDialog({
            id: "comment-add-dialog",
            trigger: ".issueaction-comment-issue:not(.inline-comment)",
            ajaxOptions: DialogUtil.getDefaultAjaxOptions,
            onSuccessfulSubmit: DialogUtil.storeCurrentIssueIdOnSucessfulSubmit,
            issueMsg: 'thanks_issue_commented',
            isIssueDialog: true,
            widthClass: "large"
        }).dirtyFormWarning();

        DialogRegister.editLabels = new LabelsDialog({
            id: "edit-labels-dialog",
            trigger: ".issueaction-edit-labels,aui-item-link.edit-labels",
            autoClose: true,
            ajaxOptions: DialogUtil.getDefaultAjaxOptions,
            onSuccessfulSubmit: DialogUtil.storeCurrentIssueIdOnSucessfulSubmit,
            issueMsg: 'thanks_issue_labelled',
            delayShowUntil: DialogUtil.BeforeShowIssueDialogHandler.execute,
            labelsProvider: labelsProvider,
            isIssueDialog: true
        });

        DialogRegister.editComment = new FormDialog({
            type: "ajax",
            id: "edit-comment",
            trigger: ".action-links .edit-comment",
            delayShowUntil: DialogUtil.BeforeShowIssueDialogHandler.execute,
            isIssueDialog: true,
            widthClass: "large"
        }).dirtyFormWarning();

        DialogRegister.editLogWork = new FormDialog({
            id: "edit-log-work-dialog",
            trigger: ".action-links .edit-worklog-trigger",
            isIssueDialog: true,
            onSuccessfulSubmit: DialogUtil.storeCurrentIssueIdOnSucessfulSubmit,
            delayShowUntil: DialogUtil.BeforeShowIssueDialogHandler.execute,
            widthClass: "large"
        });

        DialogRegister.deleteLogWork = new FormDialog({
            id: "delete-log-work-dialog",
            trigger: ".action-links .delete-worklog-trigger",
            isIssueDialog: true,
            onSuccessfulSubmit: DialogUtil.storeCurrentIssueIdOnSucessfulSubmit,
            delayShowUntil: DialogUtil.BeforeShowIssueDialogHandler.execute,
            widthClass: "large"
        });

        DialogRegister.deleteComment = new FormDialog({
            type: "ajax",
            id: "delete-comment-dialog",
            trigger: ".action-links [id^='delete_comment']",
            delayShowUntil: DialogUtil.BeforeShowIssueDialogHandler.execute,
            isIssueDialog: true,
            widthClass: "large"
        }).dirtyFormWarning();

        DialogRegister.watchIssue = new FormDialog({
            ajaxOptions: DialogUtil.getDefaultAjaxOptions,
            onSuccessfulSubmit: DialogUtil.storeCurrentIssueIdOnSucessfulSubmit,
            delayShowUntil: DialogUtil.BeforeShowIssueDialogHandler.execute,
            trigger: ".issueaction-watch-issue:not([id='toggle-watch-issue'])",
            isIssueDialog: true
        });

        DialogRegister.stopWatchingIssue = new FormDialog({
            ajaxOptions: DialogUtil.getDefaultAjaxOptions,
            onSuccessfulSubmit: DialogUtil.storeCurrentIssueIdOnSucessfulSubmit,
            delayShowUntil: DialogUtil.BeforeShowIssueDialogHandler.execute,
            trigger: ".issueaction-unwatch-issue:not([id='toggle-unwatch-issue'])",
            isIssueDialog: true
        });

        DialogRegister.addVoteForIssue = new FormDialog({
            ajaxOptions: DialogUtil.getDefaultAjaxOptions,
            onSuccessfulSubmit: DialogUtil.storeCurrentIssueIdOnSucessfulSubmit,
            delayShowUntil: DialogUtil.BeforeShowIssueDialogHandler.execute,
            trigger: ".issueaction-vote-issue:not([id='toggle-vote-issue'])",
            isIssueDialog: true
        });

        DialogRegister.removeVoteForIssue = new FormDialog({
            ajaxOptions: DialogUtil.getDefaultAjaxOptions,
            onSuccessfulSubmit: DialogUtil.storeCurrentIssueIdOnSucessfulSubmit,
            delayShowUntil: DialogUtil.BeforeShowIssueDialogHandler.execute,
            trigger: ".issueaction-unvote-issue:not([id='toggle-unvote-issue'])",
            isIssueDialog: true
        });

        new FormDialog({
            type: "ajax",
            id: "delete-attachment-dialog",
            trigger: "#attachmentmodule .attachment-delete a, #issue-attachments-table a.delete",
            isIssueDialog: true
        });

        function labelsProvider(labelsPopup) {
            var $trigger = labelsPopup.$activeTrigger;
            var $labelsContainer = $trigger.closest(".labels-wrap");
            var isSubtaskForm = $trigger.parents("#view-subtasks").length !== 0;

            if (isSubtaskForm) {
                //if we clicked the subtask form, only look within the current row for a labels wrap!
                $labelsContainer = $trigger.parents("tr").find(".labels-wrap");
            } else if ($trigger.hasClass("issueaction-edit-labels")) {
                // we clicked the issueaction which should only update the system field labels!
                if (IssueNavigator.isNavigator()) {
                    $labelsContainer = jQuery("#issuetable tr.issuerow.focused td.labels .labels-wrap");
                } else {
                    $labelsContainer = jQuery("#wrap-labels .labels-wrap");
                }
            }

            if ($labelsContainer.length > 0) {
                return $labelsContainer;
            }
            return false;
        }
    };
});