import streamlit as st
import pandas as pd
import sqlite3
import joblib
import os
import plotly.graph_objects as go

# 1. CONFIGURAÇÕES DA PÁGINA
st.set_page_config(
    page_title="PokéBattle Predictor - Nayara Diniz", 
    page_icon="⚔️", 
    layout="wide"
)

# 2. ESTILIZAÇÃO (CSS)
def local_css():
    st.markdown("""
        <style>
        .main { background-color: #f0f2f6; }
        .stButton>button { 
            background-color: #ffcb05; 
            color: #3d7dca; 
            font-weight: bold; 
            border-radius: 10px;
            width: 100%;
            height: 3em;
            border: 2px solid #3d7dca;
        }
        .stButton>button:hover {
            background-color: #3d7dca;
            color: #ffcb05;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# 3. CARREGAMENTO DE RECURSOS (Modelo, Scaler e Banco)
@st.cache_resource
def load_assets():
    # Caminhos relativos à raiz do projeto
    base_path = os.path.dirname(__file__)
    db_path = os.path.join(base_path, 'data', 'database', 'pokemon_data.db')
    model_path = os.path.join(base_path, 'models', 'pokemon_model.pkl')
    scaler_path = os.path.join(base_path, 'models', 'scaler.pkl')

    modelo = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    # A mágica está aqui: usar o caminho absoluto dinâmico
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return modelo, scaler, conn

# 4. FUNÇÕES DE SUPORTE
def get_pokemon_list():
    return pd.read_sql("SELECT id, name FROM features_pokemons ORDER BY name", conn)

def get_pokemon_stats(name):
    # Usamos parâmetros para evitar erros com nomes que possuem aspas (ex: Farfetch'd)
    query = "SELECT * FROM features_pokemons WHERE name = ?"
    df = pd.read_sql(query, conn, params=(name,))
    if df.empty:
        return None
    return df.iloc[0]

def criar_grafico_comparativo(p1_data, p2_data, p1_name, p2_name):
    stats = ['HP', 'Ataque', 'Defesa', 'Atq. Esp.', 'Def. Esp.', 'Velocidade']
    p1_vals = [p1_data['hp'], p1_data['attack'], p1_data['defense'], 
               p1_data['special-attack'], p1_data['special-defense'], p1_data['speed']]
    p2_vals = [p2_data['hp'], p2_data['attack'], p2_data['defense'], 
               p2_data['special-attack'], p2_data['special-defense'], p2_data['speed']]

    fig = go.Figure()
    fig.add_trace(go.Bar(y=stats, x=p1_vals, name=p1_name, orientation='h', marker_color='#3d7dca'))
    fig.add_trace(go.Bar(y=stats, x=p2_vals, name=p2_name, orientation='h', marker_color='#ffcb05'))
    fig.update_layout(barmode='group', height=400, margin=dict(t=30, b=0))
    return fig

# 5. INTERFACE DO USUÁRIO
st.title("⚔️ PokéBattle Predictor")
st.markdown("---")

df_names = get_pokemon_list()
col1, space, col2 = st.columns([1, 0.2, 1])

with col1:
    p1_name = st.selectbox("Selecione o Primeiro Pokémon:", df_names['name'], key="p1_sel")
    p1_data = get_pokemon_stats(p1_name)
    if p1_data is not None:
        st.image(f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{p1_data['id']}.png", width=300)
        st.write(f"**Tipo:** {p1_data['type_1'].upper()}")

with col2:
    p2_name = st.selectbox("Selecione o Segundo Pokémon:", df_names['name'], key="p2_sel")
    p2_data = get_pokemon_stats(p2_name)
    if p2_data is not None:
        st.image(f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{p2_data['id']}.png", width=300)
        st.write(f"**Tipo:** {p2_data['type_1'].upper()}")

st.markdown("---")

# 6. LÓGICA DE BATALHA
if st.button("PREVER VENCEDOR DA BATALHA"):
    if p1_data is not None and p2_data is not None:
        # Criamos o dicionário com os nomes exatos das colunas do treino
        diffs = {
            'diff_hp': p1_data['hp'] - p2_data['hp'],
            'diff_attack': p1_data['attack'] - p2_data['attack'],
            'diff_defense': p1_data['defense'] - p2_data['defense'],
            'diff_sp_atk': p1_data['special-attack'] - p2_data['special-attack'],
            'diff_sp_def': p1_data['special-defense'] - p2_data['special-defense'],
            'diff_speed': p1_data['speed'] - p2_data['speed'],
            'type_advantage': 0 
        }
        
        input_df = pd.DataFrame([diffs])
        
        try:
            input_scaled = scaler.transform(input_df)
            prediction = modelo.predict(input_scaled)[0]
            prob = modelo.predict_proba(input_scaled).max()

            vencedor = p1_name if prediction == 1 else p2_name
            cor = "#3d7dca" if prediction == 1 else "#ffcb05"
            
            st.markdown(f"<h2 style='text-align: center; color: {cor};'>🏆 {vencedor.upper()} VENCEU!</h2>", unsafe_allow_html=True)
            st.write(f"**Confiança da IA:** {prob:.2%}")
            st.plotly_chart(criar_grafico_comparativo(p1_data, p2_data, p1_name, p2_name), width='stretch')
            
        except Exception as e:
            st.error(f"Erro na predição: {e}")
    else:
        st.warning("Selecione Pokémons válidos para batalhar.")

# 7. RODAPÉ
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #707070; padding: 10px;'>
        Projeto desenvolvido por <b>Nayara Diniz</b>. 
        GitHub: <a href='https://github.com/nayarasudariodiniz' target='_blank' style='color: #3d7dca; text-decoration: none;'>nayarasudariodiniz</a>
    </div>
    """, 
    unsafe_allow_html=True
)