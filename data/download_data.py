#importar la funcion de descarga de datos desde Hugging Face Hub
from huggingface_hub import hf_hub_download

#Constantes para la descarga de datos
repo = "AgentRisk/agentrisk-data-2027"
tipo = "dataset"
destino = "data/raw"
archivos = ["data/train.jsonl", "data/dev.jsonl", "metadata/split_manifest.jsonl"]

def download_data():
    for archivo in archivos:
        hf_hub_download(repo_id=repo, repo_type=tipo, filename=archivo, local_dir=destino)
        
if __name__ == "__main__":
    download_data()