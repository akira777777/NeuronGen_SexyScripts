@echo off
cd /d "%~dp0"
chcp 65001 > nul
title NeuronGen Studio v2.2 - CLI
cls
python generate.py
pause
