import pandas as pd
import json
import os

# Caminhos de pastas
RAW_PATH = 'data/raw/pokemons_detalhes.json'
PROCESSED_PATH = 'data/processed/'

def carregar_dados_brutos(caminho):
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)

def achatar_pokemon(lista_pokemons):
    """Transforma JSON aninhado em uma lista de dicionários planos para o DataFrame."""
    dados_planos = []
    
    for p in lista_pokemons:
        # Iniciamos com os dados básicos
        pokemon_flat = {
            "id": p.get("id"),
            "name": p.get("name"),
            "height": p.get("height"),
            "weight": p.get("weight"),
            "type_1": p.get("types")[0] if len(p.get("types", [])) > 0 else None,
            "type_2": p.get("types")[1] if len(p.get("types", [])) > 1 else None
        }
        
        # 'Achatamos' os stats: transformamos a chave do stat em uma coluna
        # Ex: de {"hp": 45} para uma coluna chamada 'hp' com valor 45
        stats = p.get("stats", {})
        pokemon_flat.update(stats)
        
        dados_planos.append(pokemon_flat)
    
    return pd.DataFrame(dados_planos)

if __name__ == "__main__":
    print("🔄 Iniciando transformação dos dados de Pokémon...")
    
    # 1. Carregar
    dados_brutos = carregar_dados_brutos(RAW_PATH)
    
    # 2. Transformar (Achatamento)
    df_pokemons = achatar_pokemon(dados_brutos)
    
    # 3. Visualizar o resultado (ajuda a validar se as colunas de stats apareceram)
    print("\nPreview do DataFrame achatado:")
    print(df_pokemons.head())
    
    # 4. Salvar (Criando a pasta processed se não existir)
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    df_pokemons.to_csv(f"{PROCESSED_PATH}pokemons_features.csv", index=False)
    
    print(f"\n✅ Transformação concluída! Arquivo salvo em: {PROCESSED_PATH}pokemons_features.csv")