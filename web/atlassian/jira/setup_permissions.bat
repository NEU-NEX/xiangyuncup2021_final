@echo off
if "%OS%" == "Windows_NT" setlocal
rem ---------------------------------------------------------------------------
rem Directory permissions script for Jira, reducing permissions for regular users on the machine
rem while granting required permissions to the "Network Service" user, which Jira Service is run as.

rem This variable is modified by the installer
set AS_SERVICE=NO

icacls <JIRA_INST> /inheritance:d
icacls <JIRA_INST> /remove "Users"
icacls <JIRA_INST> /remove "Authenticated Users"
icacls <JIRA_HOME> /inheritance:d
icacls <JIRA_HOME> /remove "Users"
icacls <JIRA_HOME> /remove "Authenticated Users"
if "%AS_SERVICE%" == "YES" icacls <JIRA_HOME> /grant "NT AUTHORITY\NetworkService":(OI)(CI)(F)
