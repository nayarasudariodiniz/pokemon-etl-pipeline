import streamlit as st
import pandas as pd
import sqlite3
import joblib
import os
import plotly.graph_objects as go

# 1. CONFIGURAÇÕES DA PÁGINA
st.set_page_config(page_title="PokéBattle Predictor - Nayara Diniz", page_icon="⚔️", layout="wide")

# 2. ESTILIZAÇÃO (CSS)
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { 
        background-color: #ffcb05; color: #3d7dca; font-weight: bold; 
        border-radius: 10px; width: 100%; height: 3em; border: 2px solid #3d7dca;
    }
    </style>
""", unsafe_allow_html=True)

# 3. CARREGAMENTO DINÂMICO DE RECURSOS (Resolve o erro de Database File)
@st.cache_resource
def load_assets():
    # Descobre o caminho da pasta onde o app.py está rodando no servidor
    base_path = os.path.dirname(__file__)
    
    # Monta os caminhos completos
    db_path = os.path.join(base_path, 'data', 'database', 'pokemon_data.db')
    model_path = os.path.join(base_path, 'models', 'pokemon_model.pkl')
    scaler_path = os.path.join(base_path, 'models', 'scaler.pkl')

    # Carrega os arquivos
    modelo = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    
    return modelo, scaler, conn

# Executa o carregamento e define as variáveis globais (Resolve o NameError)
try:
    modelo, scaler, conn = load_assets()
except Exception as e:
    st.error(f"Erro ao carregar recursos: {e}")
    st.stop()

# 4. FUNÇÕES DE SUPORTE (Agora enxergam a variável 'conn' global)
def get_pokemon_list():
    return pd.read_sql("SELECT id, name FROM features_pokemons ORDER BY name", conn)

def get_pokemon_stats(name):
    query = "SELECT * FROM features_pokemons WHERE name = ?"
    df = pd.read_sql(query, conn, params=(name,))
    return df.iloc[0] if not df.empty else None

def criar_grafico_comparativo(p1_data, p2_data, p1_name, p2_name):
    stats = ['HP', 'Ataque', 'Defesa', 'Atq. Esp.', 'Def. Esp.', 'Velocidade']
    p1_vals = [p1_data['hp'], p1_data['attack'], p1_data['defense'], p1_data['special-attack'], p1_data['special-defense'], p1_data['speed']]
    p2_vals = [p2_data['hp'], p2_data['attack'], p2_data['defense'], p2_data['special-attack'], p2_data['special-defense'], p2_data['speed']]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=stats, x=p1_vals, name=p1_name, orientation='h', marker_color='#3d7dca'))
    fig.add_trace(go.Bar(y=stats, x=p2_vals, name=p2_name, orientation='h', marker_color='#ffcb05'))
    fig.update_layout(barmode='group', height=350, margin=dict(t=20, b=20))
    return fig

# 5. INTERFACE
st.title("⚔️ PokéBattle Predictor")
st.markdown("---")

df_names = get_pokemon_list()
c1, _, c2 = st.columns([1, 0.2, 1])

with c1:
    p1_n = st.selectbox("Pokémon 1:", df_names['name'], key="p1")
    p1_d = get_pokemon_stats(p1_n)
    if p1_d is not None:
        st.image(f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{p1_d['id']}.png", width=250)

with c2:
    p2_n = st.selectbox("Pokémon 2:", df_names['name'], key="p2")
    p2_d = get_pokemon_stats(p2_n)
    if p2_d is not None:
        st.image(f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{p2_d['id']}.png", width=250)

if st.button("PREVER VENCEDOR"):
    diffs = {
        'diff_hp': p1_d['hp'] - p2_d['hp'], 'diff_attack': p1_d['attack'] - p2_d['attack'],
        'diff_defense': p1_d['defense'] - p2_d['defense'], 'diff_sp_atk': p1_d['special-attack'] - p2_d['special-attack'],
        'diff_sp_def': p1_d['special-defense'] - p2_d['special-defense'], 'diff_speed': p1_d['speed'] - p2_d['speed'],
        'type_advantage': 0 
    }
    input_scaled = scaler.transform(pd.DataFrame([diffs]))
    pred = modelo.predict(input_scaled)[0]
    
    venc = p1_n if pred == 1 else p2_n
    st.success(f"🏆 Vencedor previsto: **{venc.upper()}**")
    st.plotly_chart(criar_grafico_comparativo(p1_d, p2_d, p1_n, p2_n), width='stretch')

# 6. RODAPÉ
st.markdown("---")
st.markdown("<div style='text-align: center; color: #707070;'>Projeto por <b>Nayara Diniz</b> | GitHub: nayarasudariodiniz</div>", unsafe_allow_html=True)