import sqlite3
import pandas as pd
import os

# Caminhos dos arquivos
DB_PATH = 'data/database/pokemon_data.db'
COMBATS_PATH = 'data/raw/combats.csv'
OUTPUT_PATH = 'data/processed/train_data.csv'

def calcular_vantagem_tipo(p1_types, p2_types, df_tipos):
    """Calcula o saldo de vantagem de tipo do P1 sobre o P2."""
    score = 0
    for t1 in p1_types:
        if t1 == 'none' or t1 is None: continue
        
        # Localiza as relações de dano do tipo t1
        try:
            info_tipo = df_tipos[df_tipos['name'] == t1].iloc[0]
            
            for t2 in p2_types:
                if t2 == 'none' or t2 is None: continue
                
                # P1 causa dano dobrado em P2
                if t2 in str(info_tipo['double_damage_to']).split(','):
                    score += 1
                # P1 recebe dano dobrado de P2
                if t2 in str(info_tipo['double_damage_from']).split(','):
                    score -= 1
        except IndexError:
            continue
    return score

def preparar_dados_treino():
    print("🔌 Conectando ao banco de dados e carregando arquivos...")
    
    # Conecta ao SQLite para buscar os dados que você extraiu e transformou
    conn = sqlite3.connect(DB_PATH)
    df_pkmn = pd.read_sql("SELECT * FROM features_pokemons", conn)
    df_tipos = pd.read_sql("SELECT * FROM features_tipos", conn)
    conn.close()

    # Carrega os combates do Kaggle
    df_combats = pd.read_csv(COMBATS_PATH)

    # Cria um dicionário para busca rápida de informações dos Pokémons por ID
    pkmn_info = df_pkmn.set_index('id').to_dict('index')

    # --- TESTE DE MAPEAMENTO (Amostra de 5 lutas) ---
    print("\n--- TESTE DE SANIDADE (MAPEAMENTO) ---")
    for _, row in df_combats.head(5).iterrows():
        id1, id2 = row['First_pokemon'], row['Second_pokemon']
        if id1 in pkmn_info and id2 in pkmn_info:
            n1, n2 = pkmn_info[id1]['name'], pkmn_info[id2]['name']
            v_id = row['Winner']
            v_nome = pkmn_info[v_id]['name'] if v_id in pkmn_info else "Desconhecido"
            print(f"Luta: {n1} vs {n2} | Vencedor: {v_nome}")
    print("---------------------------------------\n")

    dados_processados = []

    print("🧬 Processando 50.000 batalhas e calculando vantagens...")
    for _, row in df_combats.iterrows():
        id1, id2 = row['First_pokemon'], row['Second_pokemon']
        
        # Garante que ambos os Pokémons existam no nosso banco de dados
        if id1 not in pkmn_info or id2 not in pkmn_info:
            continue
            
        p1 = pkmn_info[id1]
        p2 = pkmn_info[id2]
        
        # 1. Diferenciais de Stats (Diferença simples entre P1 e P2)
        registro = {
            'diff_hp': p1['hp'] - p2['hp'],
            'diff_attack': p1['attack'] - p2['attack'],
            'diff_defense': p1['defense'] - p2['defense'],
            'diff_sp_atk': p1['special-attack'] - p2['special-attack'],
            'diff_sp_def': p1['special-defense'] - p2['special-defense'],
            'diff_speed': p1['speed'] - p2['speed']
        }
        
        # 2. Vantagem de Tipo (Usando a lógica da tabela features_tipos)
        vantagem = calcular_vantagem_tipo(
            [p1['type_1'], p1['type_2']], 
            [p2['type_1'], p2['type_2']], 
            df_tipos
        )
        registro['type_advantage'] = vantagem
        
        # 3. Target (O objetivo do modelo: P1 venceu?)
        registro['p1_venceu'] = 1 if id1 == row['Winner'] else 0
        
        dados_processados.append(registro)

    # Converte a lista de dicionários em um DataFrame final
    df_final = pd.DataFrame(dados_processados)

    # Salva na pasta 'processed' para o treino do modelo
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_final.to_csv(OUTPUT_PATH, index=False)
    
    print(f"✅ Sucesso! Arquivo '{OUTPUT_PATH}' gerado com {len(df_final)} linhas.")

if __name__ == "__main__":
    preparar_dados_treino()