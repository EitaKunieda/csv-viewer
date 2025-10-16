#sudo apt-get update
#sudo apt-get install -y libicu-dev

#!/usr/bin/env bash
sudo apt-get update
sudo apt-get install -y icu-devtools libicu-dev libicu70
export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1

#!/bin/bash
# 環境変数を追加
#export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=true
#export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1

# Streamlit 起動
#sudo -E -u appuser /home/adminuser/venv/bin/streamlit run app.py
