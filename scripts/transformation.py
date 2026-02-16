import pandas as pd
import json
import os
import sqlite3

# Caminhos de arquivos
RAW_DIR = 'data/raw/'
DB_PATH = 'data/database/pokemon_data.db'

def carregar_json(nome_arquivo):
    caminho = os.path.join(RAW_DIR, nome_arquivo)
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)

def transformar_pokemons():
    """Achata dados de stats e tipos, tratando valores nulos para o modelo de ML."""
    dados = carregar_json('pokemons_detalhes.json')
    lista_plana = []
    
    for p in dados:
        # Extração básica e estatísticas
        registro = {
            "id": p.get("id"),
            "name": p.get("name"),
            "height": p.get("height"),
            "weight": p.get("weight"),
            "type_1": p.get("types")[0] if p.get("types") else None,
            "type_2": p.get("types")[1] if len(p.get("types", [])) > 1 else None
        }
        # Adiciona os stats (HP, Attack, etc) como colunas diretas
        registro.update(p.get("stats", {}))
        lista_plana.append(registro)
    
    df = pd.DataFrame(lista_plana)

    # --- TRATAMENTO DE DADOS FALTANTES (ESSENCIAL PARA ML) ---
    
    # 1. Tratando o type_2: Se for nulo, significa que o Pokémon tem tipo único.
    # Preenchemos com 'none' para que o SQLite não exiba NULL e o modelo entenda a ausência.
    df['type_2'] = df['type_2'].fillna('none')

    # 2. Garantia de Stats: Caso algum stat venha vazio por erro da API.
    # Definimos a lista de colunas numéricas de combate.
    cols_stats = ['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']
    
    # Preenchemos qualquer eventual nulo com 0 ou com a média (usaremos 0 para não inventar poder).
    for col in cols_stats:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df

def transformar_habilidades():
    """Achata os detalhes das habilidades."""
    dados = carregar_json('habilidades_detalhes.json')
    # Como já estruturamos o JSON na extração, o Pandas lê quase direto
    return pd.DataFrame(dados)

def transformar_tipos():
    """Achata as relações de dano e trata valores ausentes."""
    dados = carregar_json('tipos_detalhes.json')
    lista_plana = []
    
    for t in dados:
        # Criamos o registro tratando cada lista de dano
        # Se a lista estiver vazia, o join retornará uma string vazia '', 
        # mas podemos garantir um padrão 'none' se preferir.
        registro = {
            "id": t.get("id"),
            "name": t.get("name"),
            "double_damage_to": ",".join(t.get("double_damage_to", [])) or "none",
            "double_damage_from": ",".join(t.get("double_damage_from", [])) or "none",
            "half_damage_to": ",".join(t.get("half_damage_to", [])) or "none",
            "half_damage_from": ",".join(t.get("half_damage_from", [])) or "none",
            "no_damage_to": ",".join(t.get("no_damage_to", [])) or "none",
            "no_damage_from": ",".join(t.get("no_damage_from", [])) or "none"
        }
        lista_plana.append(registro)
        
    return pd.DataFrame(lista_plana)

def salvar_no_sqlite(dfs_dict):
    """Recebe um dicionário de {nome_tabela: dataframe} e salva no banco."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        for tabela, df in dfs_dict.items():
            df.to_sql(tabela, conn, if_exists='replace', index=False)
            print(f"✅ Tabela '{tabela}' salva no SQLite ({len(df)} registros).")

if __name__ == "__main__":
    print("🔄 Iniciando Transformação para o Modelo de ML...")
    
    # Processando cada entidade
    df_pkmn = transformar_pokemons()
    df_hab = transformar_habilidades()
    df_tipos = transformar_tipos()
    
    # Dicionário para carga em lote
    tabelas = {
        "features_pokemons": df_pkmn,
        "features_habilidades": df_hab,
        "features_tipos": df_tipos
    }
    
    salvar_no_sqlite(tabelas)
    print("\n ETL de Transformação concluído com sucesso!")