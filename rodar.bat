@echo off
cd /d "%~dp0"
echo Verificando e instalando dependencias...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo Iniciando o dashboard Streamlit...
python -m streamlit run main.py
pause