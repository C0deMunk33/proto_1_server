# proto_1_server

curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/ubuntu/.local/bin:$PATH"

poetry install

poetry shell 

# install llamacpp

mkdir -p '../models/text'

download model from HF

