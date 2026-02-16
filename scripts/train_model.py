import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier  # Substituindo a importação
import joblib
import os

# Caminhos
INPUT_PATH = 'data/processed/train_data.csv'
MODEL_PATH = 'models/pokemon_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

def treinar_modelo_xgboost():
    # 1. Carregar os dados
    print("📊 Carregando matriz de treino...")
    df = pd.read_csv(INPUT_PATH)
    X = df.drop('p1_venceu', axis=1)
    y = df['p1_venceu']

    # 2. Dividir em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Normalização (Essencial para o XGBoost performar bem)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Criar e Treinar o XGBoost
    print("🚀 Treinando com XGBClassifier...")
    # n_estimators: número de rodadas de reforço
    # learning_rate: o "tamanho do passo" para não viciar o modelo
    modelo = XGBClassifier(
        n_estimators=1000, 
        learning_rate=0.05, 
        max_depth=6, 
        n_jobs=-1, 
        random_state=42
    )
    
    modelo.fit(X_train_scaled, y_train)

    # 5. Avaliação
    predicoes = modelo.predict(X_test_scaled)
    acuracia = accuracy_score(y_test, predicoes)
    
    print(f"\n✅ Treinamento concluído!")
    print(f"🎯 Nova Acurácia (XGBoost): {acuracia:.2%}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, predicoes))

    # 6. Salvar
    os.makedirs('models', exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"💾 Modelo e Scaler salvos com sucesso!")

if __name__ == "__main__":
    treinar_modelo_xgboost()