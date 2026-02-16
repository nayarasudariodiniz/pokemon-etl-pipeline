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
        .pokemon-card {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            border: 5px solid #ffcb05;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# 3. CARREGAMENTO DE RECURSOS (Modelo, Scaler e Banco)
@st.cache_resource
def load_assets():
    # Carrega o modelo XGBoost e o Scaler salvos anteriormente
    modelo = joblib.load('models/pokemon_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    # Conecta ao banco de dados SQLite
    conn = sqlite3.connect('data/database/pokemon_data.db', check_same_thread=False)
    return modelo, scaler, conn

try:
    modelo, scaler, conn = load_assets()
except Exception as e:
    st.error(f"Erro ao carregar modelos ou banco de dados: {e}")
    st.stop()

# 4. FUNÇÕES DE SUPORTE (Gráficos e Dados)
def get_pokemon_list():
    query = "SELECT id, name FROM features_pokemons ORDER BY name"
    return pd.read_sql(query, conn)

def get_pokemon_stats(name):
    query = f"SELECT * FROM features_pokemons WHERE name = '{name}'"
    return pd.read_sql(query, conn).iloc[0]

def criar_grafico_comparativo(p1_data, p2_data, p1_name, p2_name):
    stats = ['HP', 'Ataque', 'Defesa', 'Atq. Esp.', 'Def. Esp.', 'Velocidade']
    
    p1_vals = [p1_data['hp'], p1_data['attack'], p1_data['defense'], 
               p1_data['special-attack'], p1_data['special-defense'], p1_data['speed']]
    
    p2_vals = [p2_data['hp'], p2_data['attack'], p2_data['defense'], 
               p2_data['special-attack'], p2_data['special-defense'], p2_data['speed']]

    fig = go.Figure()
    fig.add_trace(go.Bar(y=stats, x=p1_vals, name=p1_name, orientation='h', marker_color='#3d7dca'))
    fig.add_trace(go.Bar(y=stats, x=p2_vals, name=p2_name, orientation='h', marker_color='#ffcb05'))

    fig.update_layout(
        title='Comparação Direta de Atributos',
        barmode='group',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# 5. INTERFACE DO USUÁRIO
st.title("⚔️ PokéBattle Predictor")
st.markdown("---")

df_names = get_pokemon_list()
col1, space, col2 = st.columns([1, 0.2, 1])

# Seleção do Pokémon 1
with col1:
    p1_name = st.selectbox("Selecione o Primeiro Pokémon:", df_names['name'], key="p1_select")
    p1_data = get_pokemon_stats(p1_name)
    st.image(f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{p1_data['id']}.png", width=300)
    st.markdown(f"**Tipo:** {p1_data['type_1'].upper()} {f'/ {p1_data['type_2'].upper()}' if p1_data['type_2'] != 'none' else ''}")

# Seleção do Pokémon 2
with col2:
    p2_name = st.selectbox("Selecione o Segundo Pokémon:", df_names['name'], key="p2_select")
    p2_data = get_pokemon_stats(p2_name)
    st.image(f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{p2_data['id']}.png", width=300)
    st.markdown(f"**Tipo:** {p2_data['type_1'].upper()} {f'/ {p2_data['type_2'].upper()}' if p2_data['type_2'] != 'none' else ''}")

st.markdown("---")

# 6. LÓGICA DE BATALHA
if st.button("PREVER VENCEDOR DA BATALHA"):
    # Prepara os dados de entrada para o modelo (Diferenciais)
    diffs = {
        'diff_hp': p1_data['hp'] - p2_data['hp'],
        'diff_attack': p1_data['attack'] - p2_data['attack'],
        'diff_defense': p1_data['defense'] - p2_data['defense'],
        'diff_sp_atk': p1_data['special-attack'] - p2_data['special-attack'],
        'diff_sp_def': p1_data['special-defense'] - p2_data['special-defense'],
        'diff_speed': p1_data['speed'] - p2_data['speed'],
        'type_advantage': 0 
    }
    
    # Transforma em DataFrame e aplica o Scaler
    input_df = pd.DataFrame([diffs])
    input_scaled = scaler.transform(input_df)
    
    # Realiza a predição
    prediction = modelo.predict(input_scaled)[0]
    probabilidade = modelo.predict_proba(input_scaled).max()

    # Exibe o resultado
    st.subheader("🏁 Resultado da Predição")
    vencedor = p1_name if prediction == 1 else p2_name
    st.success(f"O modelo prevê que o vencedor será: **{vencedor.upper()}**")
    st.write(f"Confiança da IA: {probabilidade:.2%}")

    # Exibe o gráfico comparativo
    st.plotly_chart(criar_grafico_comparativo(p1_data, p2_data, p1_name, p2_name), use_container_width=True)

# 7. RODAPÉ (Substituição da Barra Lateral)
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