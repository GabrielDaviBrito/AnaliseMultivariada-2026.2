#!/bin/zsh
cd "$(dirname "$0")"
echo "Verificando e instalando dependencias..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
echo "Iniciando o dashboard Streamlit..."
python3 -m streamlit run main.py
echo "Pressione qualquer tecla para fechar..."
read -k 1