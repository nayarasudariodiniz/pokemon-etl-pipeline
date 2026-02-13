import os
import sys
import ipdb
from dotenv import load_dotenv
# Adiciona a pasta raiz ao caminho de busca do Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import requests
from constantes import URL_BASE
import time
import json
from datetime import datetime

load_dotenv()

def realizar_requisicao(url, max_tentativas=3):
    """Realiza a requisição com sistema de tentativas (retries)."""
    tentativa = 0
    espera = 2  # Segundos iniciais de espera

    while tentativa < max_tentativas:
        try:
            user_agent = os.getenv('POKEAPI_USER_AGENT')
            headers = {"User-Agent": user_agent}
            
            response = requests.get(url, headers=headers, timeout=10) # Adicionado timeout
            response.raise_for_status() 
            return response.json()

        except requests.exceptions.RequestException as err:
            tentativa += 1
            print(f"⚠️ Tentativa {tentativa}/{max_tentativas} falhou para {url}. Erro: {err}")
            
            if tentativa < max_tentativas:
                print(f"🔄 Aguardando {espera}s para tentar novamente...")
                time.sleep(espera)
                espera *= 2  # Aumenta o tempo de espera (exponencial)
            else:
                print(f"❌ Limite de tentativas atingido para {url}.")
                return None

# --- ACRESCENTADO: Função para percorrer a paginação da API ---
def extrair_todas_as_paginas(endpoint):
    """Percorre as páginas da API para obter a lista básica de recursos."""
    url_atual = f"{URL_BASE.rstrip('/')}/{endpoint}?limit=100"
    todos_os_resultados = []

    while url_atual:
        dados = realizar_requisicao(url_atual)
        if not dados:
            break
        
        todos_os_resultados.extend(dados['results'])
        url_atual = dados.get('next') # Pega a próxima página, se houver
        time.sleep(0.5) # Delay pequeno entre páginas

    return todos_os_resultados

def extrair_detalhes_pokemon(url_pokemon):
    """Nova função: Entra na URL do Pokémon e extrai o que importa para ML."""
    dados = realizar_requisicao(url_pokemon)
    if not dados:
        return None
    
    # Extraindo apenas o necessário para o modelo de ML
    detalhes = {
        "id": dados.get("id"),
        "name": dados.get("name"),
        "height": dados.get("height"),
        "weight": dados.get("weight"),
        "types": [t["type"]["name"] for t in dados.get("types", [])],
        "stats": {s["stat"]["name"]: s["base_stat"] for s in dados.get("stats", [])}
    }
    return detalhes


def salvar_dados(dados, nome_arquivo):
    """Verifica se há dados e salva no computador."""
    if dados:
        os.makedirs('data/raw', exist_ok=True)
        caminho = f'data/raw/{nome_arquivo}.json'
        
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(dados, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Sucesso! {len(dados)} registros salvos em: {caminho}")
    else:
        print(f"❌ Erro: Não foi possível capturar dados para {nome_arquivo}.")


if __name__ == "__main__":
    # 1. Extraindo a lista de todos os Pokémons
    print("Iniciando extração de Pokémons...")
    lista_pokemons = extrair_todas_as_paginas("pokemon")
    salvar_dados(lista_pokemons, "pokemons_list")

    # --- ALTERADO: Adição da lógica de detalhes para o modelo de ML ---
    # Razão: Para prever batalhas, precisamos dos atributos numéricos de cada pokemon.
    print("\nIniciando extração de DETALHES dos Pokémons...")
    detalhes_totais = []
    for p in lista_pokemons:
        print(f"📦 Coletando detalhes: {p['name']}", end='\r')
        info = extrair_detalhes_pokemon(p['url'])
        if info:
            detalhes_totais.append(info)
        time.sleep(0.1) # Respeito à API para não ser bloqueada
    
    salvar_dados(detalhes_totais, "pokemons_detalhes")

    # 2. Extraindo a lista de todos os Tipos (Fire, Water, etc)
    print("\nIniciando extração de Tipos...")
    lista_tipos = extrair_todas_as_paginas("type")
    salvar_dados(lista_tipos, "tipos_list")

    # 3. Extraindo a lista de todas as Habilidades
    print("\nIniciando extração de Habilidades...")
    lista_habilidades = extrair_todas_as_paginas("ability")
    salvar_dados(lista_habilidades, "habilidades_list")

    print(f"\nSucesso! Capturamos {len(lista_pokemons)} Pokémons (com detalhes), "
          f"{len(lista_tipos)} Tipos e {len(lista_habilidades)} Habilidades.")