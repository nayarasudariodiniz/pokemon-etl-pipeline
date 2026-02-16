import streamlit as st
import pandas as pd
import sqlite3
import joblib
import os
import plotly.graph_objects as go

# 1. CONFIGURAÇÕES
st.set_page_config(page_title="PokéBattle Predictor", page_icon="⚔️", layout="wide")

# 2. CARREGAMENTO (Simplificado para a Raiz)
@st.cache_resource
def load_assets():
    # Pega o caminho da pasta onde este app.py está
    atual_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Procura os arquivos na raiz ou em suas pastas específicas
    db_path = os.path.join(atual_dir, 'pokemon_data.db') 
    model_path = os.path.join(atual_dir, 'models', 'pokemon_model.pkl')
    scaler_path = os.path.join(atual_dir, 'models', 'scaler.pkl')

    # Abre a conexão apenas para leitura (mode=ro) para evitar erros de permissão no servidor
    try:
        modelo = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        # URI=True permite passar o modo de leitura
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        return modelo, scaler, conn
    except Exception as e:
        st.error(f"Erro ao carregar: {e}")
        st.stop()

modelo, scaler, conn = load_assets()

# 3. FUNÇÕES
def get_pokemon_list():
    return pd.read_sql("SELECT id, name FROM features_pokemons ORDER BY name", conn)

def get_pokemon_stats(name):
    query = "SELECT * FROM features_pokemons WHERE name = ?"
    df = pd.read_sql(query, conn, params=(name,))
    return df.iloc[0] if not df.empty else None

# 4. INTERFACE
st.title("⚔️ PokéBattle Predictor")

df_names = get_pokemon_list()
col1, col2 = st.columns(2)

with col1:
    p1_n = st.selectbox("Pokémon 1", df_names['name'], key="p1")
    p1_d = get_pokemon_stats(p1_n)
    if p1_d is not None:
        st.image(f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{p1_d['id']}.png", width=200)

with col2:
    p2_n = st.selectbox("Pokémon 2", df_names['name'], key="p2")
    p2_d = get_pokemon_stats(p2_n)
    if p2_d is not None:
        st.image(f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{p2_d['id']}.png", width=200)

if st.button("BATALHA!"):
    diffs = pd.DataFrame([{
        'diff_hp': p1_d['hp'] - p2_d['hp'],
        'diff_attack': p1_d['attack'] - p2_d['attack'],
        'diff_defense': p1_d['defense'] - p2_d['defense'],
        'diff_sp_atk': p1_d['special-attack'] - p2_d['special-attack'],
        'diff_sp_def': p1_d['special-defense'] - p2_d['special-defense'],
        'diff_speed': p1_d['speed'] - p2_d['speed'],
        'type_advantage': 0
    }])
    
    pred = modelo.predict(scaler.transform(diffs))[0]
    vencedor = p1_n if pred == 1 else p2_n
    st.success(f"🏆 Vencedor: {vencedor.upper()}")