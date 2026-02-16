import streamlit as st
import pandas as pd
import sqlite3
import joblib
import os

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="PokéBattle Predictor",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ESTILIZAÇÃO CSS ---
def local_css():
    st.markdown("""
        <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .stButton>button { 
            background-color: #ffcb05; 
            color: #2a75bb; 
            font-weight: 800; 
            border-radius: 12px;
            width: 100%;
            height: 3.5em;
            border: 3px solid #3c5aa6;
            font-size: 20px;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.02);
            background-color: #ffdb4d;
        }
        .stat-box {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 10px;
        }
        .vs-text {
            text-align: center;
            font-size: 50px;
            font-weight: 900;
            color: #cc0000;
            margin-top: 100px;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- CARREGAMENTO DE RECURSOS (CACHEADO) ---
@st.cache_resource
def load_resources():
    # Carrega modelos
    modelo = joblib.load('models/pokemon_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Conecta ao banco e carrega dados estáticos necessários para cálculo
    conn = sqlite3.connect('data/database/pokemon_data.db', check_same_thread=False)
    
    # Carregamos a tabela de tipos na memória para agilizar o cálculo de vantagem
    df_tipos = pd.read_sql("SELECT * FROM features_tipos", conn)
    
    return modelo, scaler, conn, df_tipos

try:
    modelo, scaler, conn, df_tipos = load_resources()
except Exception as e:
    st.error(f"Erro ao carregar recursos. Verifique se executou o `train_model.py` e se o banco existe. Detalhes: {e}")
    st.stop()

# --- FUNÇÕES DE LÓGICA (Replicando a lógica do prepare_ml_data.py) ---

def get_pokemon_list():
    """Retorna lista de nomes para o dropdown."""
    return pd.read_sql("SELECT id, name FROM features_pokemons ORDER BY name", conn)

def get_pokemon_data(name):
    """Busca os stats de um Pokémon específico."""
    # Nota: SQL pode ter problemas com hífens em nomes de colunas sem aspas. 
    # Trazemos tudo e filtramos no pandas para evitar erros de sintaxe SQL.
    query = f"SELECT * FROM features_pokemons WHERE name = '{name}'"
    df = pd.read_sql(query, conn)
    return df.iloc[0] if not df.empty else None

def calcular_vantagem_tipo_real(p1_types, p2_types, df_tipos_ref):
    """
    Mesma lógica exata do arquivo prepare_ml_data.py para garantir
    que o modelo receba o mesmo padrão de dados do treino.
    """
    score = 0
    for t1 in p1_types:
        if t1 == 'none' or t1 is None: continue
        
        try:
            # Localiza as relações de dano do tipo t1
            # Importante: O prepare_ml_data usa colunas strings 'double_damage_to' etc
            info_tipo = df_tipos_ref[df_tipos_ref['name'] == t1].iloc[0]
            
            for t2 in p2_types:
                if t2 == 'none' or t2 is None: continue
                
                # Verifica se t2 está na lista de double_damage_to (string separada por vírgula ou lista)
                # O seu prepare_ml_data faz split na string, vamos manter a consistência
                dd_to = str(info_tipo['double_damage_to']).split(',')
                dd_from = str(info_tipo['double_damage_from']).split(',')
                
                if t2 in dd_to:
                    score += 1
                if t2 in dd_from:
                    score -= 1
        except IndexError:
            continue
    return score

# --- INTERFACE DO USUÁRIO ---

st.title("⚡ PokéBattle AI Predictor")
st.markdown("Use Machine Learning para prever o vencedor de um duelo Pokémon!")

df_names = get_pokemon_list()

# Layout de Colunas
c1, c_mid, c2 = st.columns([1, 0.3, 1])

# --- SELEÇÃO POKÉMON 1 ---
with c1:
    st.markdown("### 🔴 Desafiante 1")
    p1_name = st.selectbox("Selecione o Pokémon", df_names['name'], key="p1", index=24) # Pikachu default
    
    if p1_name:
        p1_data = get_pokemon_data(p1_name)
        # Exibição de imagem oficial
        st.image(
            f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{p1_data['id']}.png", 
            use_container_width=True
        )
        
        # Card de Stats
        with st.container():
            st.markdown(f"""
                <div class='stat-box'>
                    <h3 style='text-align: center; color: #3c5aa6;'>{p1_name.capitalize()}</h3>
                    <p><b>Tipos:</b> {p1_data['type_1']} {f"/ {p1_data['type_2']}" if p1_data['type_2'] != 'none' else ''}</p>
                    <hr>
                    <p>❤️ HP: {p1_data['hp']}</p>
                    <p>⚔️ Atk: {p1_data['attack']} | 🛡️ Def: {p1_data['defense']}</p>
                    <p>🔮 Sp. Atk: {p1_data['special-attack']} | 🔮 Sp. Def: {p1_data['special-defense']}</p>
                    <p>⚡ Speed: {p1_data['speed']}</p>
                </div>
            """, unsafe_allow_html=True)

# --- VERSUS ---
with c_mid:
    st.markdown("<div class='vs-text'>VS</div>", unsafe_allow_html=True)

# --- SELEÇÃO POKÉMON 2 ---
with c2:
    st.markdown("### 🔵 Desafiante 2")
    p2_name = st.selectbox("Selecione o Pokémon", df_names['name'], key="p2", index=5) # Charizard default
    
    if p2_name:
        p2_data = get_pokemon_data(p2_name)
        st.image(
            f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{p2_data['id']}.png", 
            use_container_width=True
        )
        
        with st.container():
            st.markdown(f"""
                <div class='stat-box'>
                    <h3 style='text-align: center; color: #3c5aa6;'>{p2_name.capitalize()}</h3>
                    <p><b>Tipos:</b> {p2_data['type_1']} {f"/ {p2_data['type_2']}" if p2_data['type_2'] != 'none' else ''}</p>
                    <hr>
                    <p>❤️ HP: {p2_data['hp']}</p>
                    <p>⚔️ Atk: {p2_data['attack']} | 🛡️ Def: {p2_data['defense']}</p>
                    <p>🔮 Sp. Atk: {p2_data['special-attack']} | 🔮 Sp. Def: {p2_data['special-defense']}</p>
                    <p>⚡ Speed: {p2_data['speed']}</p>
                </div>
            """, unsafe_allow_html=True)

# --- BOTÃO E LÓGICA DE PREVISÃO ---
st.divider()
col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 2, 1])

with col_btn_2:
    botao_batalha = st.button("PREVER VENCEDOR 🥊")

if botao_batalha and p1_name and p2_name:
    if p1_name == p2_name:
        st.warning("Selecione Pokémons diferentes para a batalha!")
    else:
        with st.spinner('A IA está analisando estatísticas e vantagens de tipo...'):
            # 1. Calcular Vantagem de Tipo (CRUCIAL: Usando a lógica real agora)
            vantagem = calcular_vantagem_tipo_real(
                [p1_data['type_1'], p1_data['type_2']],
                [p2_data['type_1'], p2_data['type_2']],
                df_tipos
            )
            
            # 2. Montar o DataFrame de Input (Exatamente como no treino)
            # Atenção aos nomes das colunas de stats (hífen vs underscore)
            # No SQLite gerado pelo pandas, 'special-attack' costuma ser mantido como string se for text, 
            # mas vamos acessar via dicionário p1_data para garantir.
            input_data = {
                'diff_hp': p1_data['hp'] - p2_data['hp'],
                'diff_attack': p1_data['attack'] - p2_data['attack'],
                'diff_defense': p1_data['defense'] - p2_data['defense'],
                'diff_sp_atk': p1_data['special-attack'] - p2_data['special-attack'],
                'diff_sp_def': p1_data['special-defense'] - p2_data['special-defense'],
                'diff_speed': p1_data['speed'] - p2_data['speed'],
                'type_advantage': vantagem
            }
            
            df_input = pd.DataFrame([input_data])
            
            # 3. Escalar (Normalizar) os dados
            input_scaled = scaler.transform(df_input)
            
            # 4. Prever
            predicao = modelo.predict(input_scaled)[0]
            probabilidade = modelo.predict_proba(input_scaled).max()
            
            # --- EXIBIÇÃO DO RESULTADO ---
            st.markdown("---")
            winner_name = p1_name if predicao == 1 else p2_name
            loser_name = p2_name if predicao == 1 else p1_name
            winner_img = p1_data['id'] if predicao == 1 else p2_data['id']
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.image(
                    f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{winner_img}.png",
                    width=200
                )
            
            with res_col2:
                st.success(f"🏆 VENCEDOR: {winner_name.upper()}!")
                st.metric(label="Confiança do Modelo", value=f"{probabilidade:.1%}")
                
                # Feedback visual sobre o fator decisivo
                detalhes = []
                if vantagem > 0 and predicao == 1:
                    detalhes.append(f"A vantagem de tipo favoreceu {p1_name}.")
                elif vantagem < 0 and predicao == 0:
                    detalhes.append(f"A vantagem de tipo favoreceu {p2_name}.")
                
                if input_data['diff_speed'] > 0 and predicao == 1:
                    detalhes.append(f"{p1_name} é mais rápido.")
                elif input_data['diff_speed'] < 0 and predicao == 0:
                    detalhes.append(f"{p2_name} é mais rápido.")
                    
                if detalhes:
                    st.caption("Fatores chave: " + " | ".join(detalhes))