#!/bin/bash
# 環境変数を追加
#export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=true
export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1

# Streamlit 起動
sudo -E -u appuser /home/adminuser/venv/bin/streamlit run app.py
