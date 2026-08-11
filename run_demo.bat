@echo off
rem Visual RAG demo launcher — always uses the visualrag conda env, so it works
rem no matter what is on PATH. Double-click, or run from any terminal.
cd /d "%~dp0"
D:\Anaconda\envs\visualrag\python.exe -m streamlit run ui/app.py %*
