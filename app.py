import streamlit as st
import pandas as pd
import sqlite3
import joblib
import os
import plotly.graph_objects as go

# 1. CONFIGURAÇÕES
st.set_page_config(page_title="PokéBattle Predictor", page_icon="⚔️", layout="wide")

# 2. CARREGAMENTO (Mantendo a lógica que funcionou para o deploy)
@st.cache_resource
def load_assets():
    atual_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Caminhos baseados na raiz onde o banco agora reside
    db_path = os.path.join(atual_dir, 'pokemon_data.db') 
    model_path = os.path.join(atual_dir, 'models', 'pokemon_model.pkl')
    scaler_path = os.path.join(atual_dir, 'models', 'scaler.pkl')

    try:
        modelo = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        # Modo de leitura para evitar travamentos no servidor
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        return modelo, scaler, conn
    except Exception as e:
        st.error(f"Erro ao carregar recursos: {e}")
        st.stop()

modelo, scaler, conn = load_assets()

# 3. FUNÇÕES DE SUPORTE (Gráfico e Dados)
def get_pokemon_list():
    return pd.read_sql("SELECT id, name FROM features_pokemons ORDER BY name", conn)

def get_pokemon_stats(name):
    query = "SELECT * FROM features_pokemons WHERE name = ?"
    df = pd.read_sql(query, conn, params=(name,))
    return df.iloc[0] if not df.empty else None

def criar_grafico_comparativo(p1_data, p2_data, p1_name, p2_name):
    stats = ['HP', 'Ataque', 'Defesa', 'Atq. Esp.', 'Def. Esp.', 'Velocidade']
    p1_vals = [p1_data['hp'], p1_data['attack'], p1_data['defense'], 
               p1_data['special-attack'], p1_data['special-defense'], p1_data['speed']]
    p2_vals = [p2_data['hp'], p2_data['attack'], p2_data['defense'], 
               p2_data['special-attack'], p2_data['special-defense'], p2_data['speed']]

    fig = go.Figure()
    fig.add_trace(go.Bar(y=stats, x=p1_vals, name=p1_name, orientation='h', marker_color='#3d7dca'))
    fig.add_trace(go.Bar(y=stats, x=p2_vals, name=p2_name, orientation='h', marker_color='#ffcb05'))
    fig.update_layout(barmode='group', height=350, margin=dict(t=20, b=20))
    return fig

# 4. INTERFACE DO USUÁRIO
st.title("⚔️ PokéBattle Predictor")
st.markdown("---")

df_names = get_pokemon_list()
col1, space, col2 = st.columns([1, 0.2, 1])

with col1:
    p1_n = st.selectbox("Pokémon 1:", df_names['name'], key="p1")
    p1_d = get_pokemon_stats(p1_n)
    if p1_d is not None:
        st.image(f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{p1_d['id']}.png", width=250)

with col2:
    p2_n = st.selectbox("Pokémon 2:", df_names['name'], key="p2")
    p2_d = get_pokemon_stats(p2_n)
    if p2_d is not None:
        st.image(f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{p2_d['id']}.png", width=250)

st.markdown("---")

if st.button("PREVER VENCEDOR"):
    # Lógica de predição idêntica à anterior
    diffs = pd.DataFrame([{
        'diff_hp': p1_d['hp'] - p2_d['hp'],
        'diff_attack': p1_d['attack'] - p2_d['attack'],
        'diff_defense': p1_d['defense'] - p2_d['defense'],
        'diff_sp_atk': p1_d['special-attack'] - p2_d['special-attack'],
        'diff_sp_def': p1_d['special-defense'] - p2_d['special-defense'],
        'diff_speed': p1_d['speed'] - p2_d['speed'],
        'type_advantage': 0
    }])
    
    input_scaled = scaler.transform(diffs)
    pred = modelo.predict(input_scaled)[0]
    prob = modelo.predict_proba(input_scaled).max()
    
    vencedor = p1_n if pred == 1 else p2_n
    st.success(f"🏆 Vencedor previsto: **{vencedor.upper()}**")
    st.write(f"Confiança da IA: {prob:.2%}")
    
    # Exibição do gráfico comparativo
    st.plotly_chart(criar_grafico_comparativo(p1_d, p2_d, p1_n, p2_n), use_container_width=True)

# 5. RODAPÉ
st.markdown("---")
st.markdown("<div style='text-align: center; color: #707070;'>Projeto por <b>Nayara Diniz</b> | GitHub: nayarasudariodiniz</div>", unsafe_allow_html=True)