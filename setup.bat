@echo off
setlocal EnableDelayedExpansion

@REM set the current directory
cd %~dp0

@REM install pyenv through pip
python -m pip install virtualenv

@REM create a virtual environment
python -m virtualenv env

@REM activate the virtual environment
call env\Scripts\activate.bat

@REM install the required packages
python -m pip install -r requirements.txt