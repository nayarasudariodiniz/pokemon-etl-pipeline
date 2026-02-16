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

# --- ESTILIZAÇÃO CSS (CORRIGIDO PARA TEMA ESCURO) ---
def local_css():
    st.markdown("""
        <style>
        /* Força fundo escuro e texto claro globalmente */
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        
        /* Estilo dos Cards de Status */
        .stat-box {
            background-color: #262730; /* Fundo cinza escuro para contraste */
            padding: 20px;
            border-radius: 15px;
            border: 1px solid #41444d;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            margin-bottom: 20px;
            text-align: center;
        }
        
        /* Títulos dentro dos cards */
        .stat-box h3 {
            color: #ffcb05; /* Amarelo Pokémon */
            margin-bottom: 10px;
            font-size: 24px;
        }
        
        /* Texto dos status */
        .stat-box p {
            font-size: 16px;
            margin: 5px 0;
            color: #e0e0e0;
        }
        
        /* Linha divisória customizada */
        hr {
            margin: 10px 0;
            border-color: #41444d;
        }
        
        /* Botão de Batalha */
        .stButton>button { 
            background-color: #ffcb05; 
            color: #1a1c24; 
            font-weight: 800; 
            border-radius: 12px;
            width: 100%;
            height: 4em;
            border: none;
            font-size: 20px;
            text-transform: uppercase;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.02);
            background-color: #ffe066;
            color: #000;
            box-shadow: 0 0 15px rgba(255, 203, 5, 0.4);
        }
        
        /* Texto VS no meio */
        .vs-text {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            font-size: 60px;
            font-weight: 900;
            color: #ff4b4b;
            text-shadow: 0 0 10px rgba(255, 75, 75, 0.5);
            padding-top: 150px; /* Ajuste para alinhar com as imagens */
        }
        
        /* Controle de Imagem */
        .pokemon-img {
            max-height: 250px;
            object-fit: contain;
            margin-bottom: 15px;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- CARREGAMENTO DE RECURSOS (CACHEADO & BLINDADO) ---
@st.cache_resource
def load_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    db_path = os.path.join(base_dir, 'data', 'database', 'pokemon_data.db')
    modelo_path = os.path.join(base_dir, 'models', 'pokemon_model.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')

    if not os.path.exists(db_path):
        st.error(f"❌ ERRO CRÍTICO: Banco de dados não encontrado em:\n{db_path}")
        st.stop()

    if not os.path.exists(modelo_path):
        st.error(f"❌ Modelo não encontrado em:\n{modelo_path}")
        st.stop()

    modelo = joblib.load(modelo_path)
    scaler = joblib.load(scaler_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    
    try:
        df_tipos = pd.read_sql("SELECT * FROM features_tipos", conn)
    except Exception:
        st.error("Erro ao ler tabela 'features_tipos'. Verifique seu banco de dados.")
        st.stop()
    
    return modelo, scaler, conn, df_tipos

try:
    modelo, scaler, conn, df_tipos = load_resources()
except Exception as e:
    st.error(f"Erro fatal ao iniciar aplicação: {e}")
    st.stop()

# --- FUNÇÕES DE LÓGICA ---
def get_pokemon_list(conexao_db):
    return pd.read_sql("SELECT id, name FROM features_pokemons ORDER BY name", conexao_db)

def get_pokemon_data(name, conexao_db):
    query = f"SELECT * FROM features_pokemons WHERE name = '{name}'"
    df = pd.read_sql(query, conexao_db)
    return df.iloc[0] if not df.empty else None

def calcular_vantagem_tipo_real(p1_types, p2_types, df_tipos_ref):
    score = 0
    for t1 in p1_types:
        if t1 == 'none' or t1 is None: continue
        try:
            filtro = df_tipos_ref[df_tipos_ref['name'] == t1]
            if filtro.empty: continue
            
            info_tipo = filtro.iloc[0]
            
            for t2 in p2_types:
                if t2 == 'none' or t2 is None: continue
                
                dd_to = str(info_tipo['double_damage_to']).split(',')
                dd_from = str(info_tipo['double_damage_from']).split(',')
                
                if t2 in dd_to: score += 1
                if t2 in dd_from: score -= 1
        except Exception:
            continue
    return score

# --- INTERFACE DO USUÁRIO ---

st.title("⚡ PokéBattle AI Predictor")
st.markdown("Use Machine Learning para prever o vencedor de um duelo Pokémon!")
st.divider()

df_names = get_pokemon_list(conn)

# Layout Principal
c1, c_mid, c2 = st.columns([1, 0.2, 1])

# --- SELEÇÃO POKÉMON 1 ---
with c1:
    st.subheader("🔴 Desafiante 1")
    p1_name = st.selectbox("Selecione o Pokémon", df_names['name'], key="p1", index=24) # Pikachu
    
    if p1_name:
        p1_data = get_pokemon_data(p1_name, conn)
        
        # Imagem Centralizada
        col_img_1, _, _ = st.columns([1, 0.1, 0.1])
        with col_img_1:
             st.image(
                f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{p1_data['id']}.png", 
                use_container_width=True
            )
        
        # Card de Stats
        st.markdown(f"""
            <div class='stat-box'>
                <h3>{p1_name.capitalize()}</h3>
                <p style='color: #888;'>Tipos: <b style='color: #fff;'>{p1_data['type_1'].upper()}</b> {f"/ <b style='color: #fff;'>{p1_data['type_2'].upper()}</b>" if p1_data['type_2'] != 'none' else ''}</p>
                <hr>
                <p>❤️ HP: <b>{p1_data['hp']}</b></p>
                <p>⚔️ Atk: <b>{p1_data['attack']}</b> | 🛡️ Def: <b>{p1_data['defense']}</b></p>
                <p>⚡ Speed: <b>{p1_data['speed']}</b></p>
            </div>
        """, unsafe_allow_html=True)

# --- VERSUS ---
with c_mid:
    st.markdown("<div class='vs-text'>VS</div>", unsafe_allow_html=True)

# --- SELEÇÃO POKÉMON 2 ---
with c2:
    st.subheader("🔵 Desafiante 2")
    p2_name = st.selectbox("Selecione o Pokémon", df_names['name'], key="p2", index=5) # Charizard
    
    if p2_name:
        p2_data = get_pokemon_data(p2_name, conn)
        
        col_img_2, _, _ = st.columns([1, 0.1, 0.1])
        with col_img_2:
            st.image(
                f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{p2_data['id']}.png", 
                use_container_width=True
            )
        
        st.markdown(f"""
            <div class='stat-box'>
                <h3>{p2_name.capitalize()}</h3>
                <p style='color: #888;'>Tipos: <b style='color: #fff;'>{p2_data['type_1'].upper()}</b> {f"/ <b style='color: #fff;'>{p2_data['type_2'].upper()}</b>" if p2_data['type_2'] != 'none' else ''}</p>
                <hr>
                <p>❤️ HP: <b>{p2_data['hp']}</b></p>
                <p>⚔️ Atk: <b>{p2_data['attack']}</b> | 🛡️ Def: <b>{p2_data['defense']}</b></p>
                <p>⚡ Speed: <b>{p2_data['speed']}</b></p>
            </div>
        """, unsafe_allow_html=True)

# --- PREVISÃO ---
# --- PREVISÃO ---
st.divider()
col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 2, 1])

with col_btn_2:
    botao_batalha = st.button("PREVER VENCEDOR 🥊")

if botao_batalha and p1_name and p2_name:
    if p1_name == p2_name:
        st.warning("Selecione Pokémons diferentes!")
    else:
        try:
            with st.spinner('Simulando batalha...'):
                # 1. Lógica de Vantagem
                vantagem = calcular_vantagem_tipo_real(
                    [p1_data['type_1'], p1_data['type_2']],
                    [p2_data['type_1'], p2_data['type_2']],
                    df_tipos
                )
                
                # 2. Montagem dos Dados
                input_data = {
                    'diff_hp': float(p1_data['hp'] - p2_data['hp']), # Convertendo para float python puro
                    'diff_attack': float(p1_data['attack'] - p2_data['attack']),
                    'diff_defense': float(p1_data['defense'] - p2_data['defense']),
                    'diff_sp_atk': float(p1_data['special-attack'] - p2_data['special-attack']),
                    'diff_sp_def': float(p1_data['special-defense'] - p2_data['special-defense']),
                    'diff_speed': float(p1_data['speed'] - p2_data['speed']),
                    'type_advantage': float(vantagem)
                }
                
                df_input = pd.DataFrame([input_data])
                
                # 3. Previsão
                input_scaled = scaler.transform(df_input)
                predicao = int(modelo.predict(input_scaled)[0]) # Força int
                probabilidade = float(modelo.predict_proba(input_scaled).max()) # Força float
                
                winner_name = p1_name if predicao == 1 else p2_name
                winner_img = p1_data['id'] if predicao == 1 else p2_data['id']
                
                # 4. Exibição (Com tratamento de erro visual)
                st.markdown("---")
                st.success(f"🏆 O Vencedor provável é: **{str(winner_name).upper()}**")
                
                st.write(f"Confiança da IA: **{probabilidade:.1%}**")
                st.progress(probabilidade)
                
                # Detalhes do Motivo
                with st.expander("🔎 Por que a IA escolheu este vencedor?", expanded=True):
                    st.write("**Análise dos Fatores:**")
                    
                    # Vantagem de Tipo
                    if vantagem > 0:
                        st.info(f"✅ {p1_name} tem vantagem de tipo (+{vantagem})")
                    elif vantagem < 0:
                        st.info(f"✅ {p2_name} tem vantagem de tipo ({vantagem})")
                    else:
                        st.write("⚖️ Sem vantagem de tipo clara.")

                    # Velocidade
                    diff_speed = input_data['diff_speed']
                    if diff_speed > 0:
                        st.write(f"⚡ {p1_name} é mais rápido (+{int(diff_speed)} pts).")
                    elif diff_speed < 0:
                        st.write(f"⚡ {p2_name} é mais rápido ({int(diff_speed)} pts).")
                    
                    # Stats Físicos
                    st.caption(f"Diferença de Ataque: {input_data['diff_attack']}")
                    st.caption(f"Diferença de Defesa: {input_data['diff_defense']}")

        except Exception as e:
            st.error("Ocorreu um erro ao gerar os detalhes da previsão.")
            st.code(f"Erro técnico: {e}") # Isso vai mostrar o erro exato na tela