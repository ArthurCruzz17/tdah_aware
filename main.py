import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_rows', None)


print("--- Carregando os Dados ---")
try:
    features_df = pd.read_csv('features.csv', delimiter=';')
    patient_info = pd.read_csv('patient_info.csv', delimiter=';')
    cpt_ii_detailed = pd.read_csv('CPT_II_ConnersContinuousPerformanceTest.csv', delimiter=';')
    print("Dados carregados com sucesso!")
except FileNotFoundError as e:
    print(f"Erro ao carregar arquivo: {e}. Certifique-se de que os arquivos .csv estão no mesmo diretório do script.")
    exit()

print("Primeiras 5 linhas de features.csv:")
print(features_df.head())

print("Colunas disponíveis em patient_info.csv:")
print(patient_info.columns.tolist())

print("\nColunas disponíveis em CPT_II_ConnersContinuousPerformanceTest.csv:")
print(cpt_ii_detailed.columns.tolist())


if 'ADHD' in patient_info.columns:
    print("Valores únicos na coluna 'ADHD':", patient_info['ADHD'].unique())
    print("Contagem de valores na coluna 'ADHD':")
    print(patient_info['ADHD'].value_counts())
else:
    print("❌ Erro: A coluna 'ADHD' não foi encontrada em 'patient_info.csv'. Verifique o nome da coluna ou o delimitador do arquivo.")
    exit()

print("\n--- Realizando o merge dos DataFrames ---")
colunas_features_desejadas = ['ID', 'ACC__mean', 'ACC__variance', 'ACC__maximum']
colunas_features_existentes = [col for col in colunas_features_desejadas if col in features_df.columns]
features_df = features_df[colunas_features_existentes]


dados_completos = pd.merge(features_df, patient_info, on='ID', how='inner')


print("\n--- Adicionando dados detalhados do CPT_II_ConnersContinuousPerformanceTest.csv ---")

cpt_ii_features_to_add = [
    'ID',
    'Raw Score Omissions',
    'Raw Score Commissions',
    'Raw Score HitRT',
    'Raw Score VarSE', 
    'Raw Score DPrime'
]


cpt_ii_detailed_filtered = cpt_ii_detailed[[col for col in cpt_ii_features_to_add if col in cpt_ii_detailed.columns]]

dados_completos = pd.merge(dados_completos, cpt_ii_detailed_filtered, on='ID', how='inner')


if 'CPT_II' in dados_completos.columns:
    dados_completos = dados_completos.drop('CPT_II', axis=1)
    print("Coluna 'CPT_II' original (binária de patient_info.csv) removida.")


print(f"Dimensões do DataFrame combinado: {dados_completos.shape}")
print("Primeiras 5 linhas do DataFrame combinado (dados_completos):")
print(dados_completos.head())


print("\n--- Tratando Valores Ausentes (NaN) ---")
colunas_para_imputar = ['WURS', 'ASRS', 'MADRS', 'HADS_A', 'HADS_D', 'MDQ_POS']

for coluna in colunas_para_imputar:
    if coluna in dados_completos.columns:
        if dados_completos[coluna].isnull().any():
            mediana_coluna = dados_completos[coluna].median()
            dados_completos[coluna].fillna(mediana_coluna, inplace=True)
            print(f"Preenchido NaN na coluna '{coluna}' com a mediana: {mediana_coluna}")
    else:
        print(f"Aviso: Coluna '{coluna}' não encontrada para imputação.")


print("\n--- Tratando Valores Ausentes (NaN) nas novas colunas CPT_II ---")
for col_cpt in [col for col in cpt_ii_features_to_add if col != 'ID']:
    if col_cpt in dados_completos.columns:
        if dados_completos[col_cpt].isnull().any():
            mediana_cpt = dados_completos[col_cpt].median()
            dados_completos[col_cpt].fillna(mediana_cpt, inplace=True)
            print(f"Preenchido NaN na coluna '{col_cpt}' com a mediana: {mediana_cpt}")
    else:
        print(f"Aviso: Coluna '{col_cpt}' não encontrada para imputação.")


print("\nVerificando NaNs após imputação:")

features_para_x_updated = [
    'ACC__mean', 'ACC__variance', 'ACC__maximum',
    'AGE', 'SEX',
    'WURS', 'ASRS', 'MADRS', 'HADS_A', 'HADS_D', 'MDQ_POS'
] + [col for col in cpt_ii_features_to_add if col in dados_completos.columns and col != 'ID'] 

missing_cols = [col for col in features_para_x_updated if col not in dados_completos.columns]
if missing_cols:
    print(f"❌ Erro: As seguintes colunas de features não foram encontradas no DataFrame combinado: {missing_cols}")
    exit()

print(dados_completos[features_para_x_updated].isnull().sum())

print("\n--- Definindo Features (X) e Target (y) ---")
X = dados_completos[features_para_x_updated]
y = dados_completos['ADHD']

print(f"Dimensões de X (features): {X.shape}")
print(f"Dimensões de y (target): {y.shape}")

print("Primeiras 5 linhas de X:")
print(X.head())
print("\nPrimeiras 5 linhas de y:")
print(y.head())

print("\n--- Dividindo os dados em conjuntos de Treino e Teste ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Dimensões de X_train: {X_train.shape}")
print(f"Dimensões de X_test: {X_test.shape}")
print(f"Dimensões de y_train: {y_train.shape}")
print(f"Dimensões de y_test: {y_test.shape}")

print("\nContagem de classes em y_train:")
print(y_train.value_counts())
print("\nContagem de classes em y_test:")
print(y_test.value_counts())


print("\n--- Escalonando as Features (X) ---")
scaler = StandardScaler()

print("\nPrimeiras 5 linhas de X_train (antes do escalonamento):")
print(X_train.head())

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("\nPrimeiras 5 linhas de X_train_scaled (depois do escalonamento):")
print(X_train_scaled.head())

print("\nPrimeiras 5 linhas de X_test (antes do escalonamento):")
print(X_test.head())

print("\nPrimeiras 5 linhas de X_test_scaled (depois do escalonamento):")
print(X_test_scaled.head())


print("\n--- Treinando o Modelo (KNeighborsClassifier com GridSearchCV) ---")

knn = KNeighborsClassifier()

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 13],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

print("\nIniciando GridSearchCV para KNN...")
grid_search_knn.fit(X_train_scaled, y_train)

print("\nMelhores parâmetros encontrados para KNN:")
print(grid_search_knn.best_params_)
print("\nMelhor pontuação (accuracy) encontrada na validação cruzada para KNN:")
print(f"{grid_search_knn.best_score_:.4f}")

best_knn_model = grid_search_knn.best_estimator_

print("\n--- Avaliando o Desempenho do Modelo KNN no conjunto de teste ---")

y_pred_knn = best_knn_model.predict(X_test_scaled)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"\nAcurácia do modelo KNN no conjunto de teste: {accuracy_knn:.4f}")

cm_knn = confusion_matrix(y_test, y_pred_knn)
print("\nMatriz de Confusão para KNN:")
print(cm_knn)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Não-ADHD', 'ADHD'], yticklabels=['Não-ADHD', 'ADHD'])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão (KNN)')
plt.show()

print("\nAnálise da Matriz de Confusão (KNN):")
tn_knn, fp_knn, fn_knn, tp_knn = cm_knn.ravel()
print(f"True Negatives (TN): {tn_knn} (Casos de Não-ADHD corretamente previstos como Não-ADHD)")
print(f"False Positives (FP): {fp_knn} (Casos de Não-ADHD previstos incorretamente como ADHD)")
print(f"False Negatives (FN): {fn_knn} (Casos de ADHD previstos incorretamente como Não-ADHD)")
print(f"True Positives (TP): {tp_knn} (Casos de ADHD corretamente previstos como ADHD)")

precision_knn = precision_score(y_test, y_pred_knn, average='binary')
recall_knn = recall_score(y_test, y_pred_knn, average='binary')
f1_knn = f1_score(y_test, y_pred_knn, average='binary')

if hasattr(best_knn_model, "predict_proba"):
    y_pred_proba_knn = best_knn_model.predict_proba(X_test_scaled)[:, 1]
    roc_auc_knn = roc_auc_score(y_test, y_pred_proba_knn)
    print(f"\nPrecision (KNN): {precision_knn:.4f}")
    print(f"Recall (Sensibilidade) (KNN): {recall_knn:.4f}")
    print(f"F1-Score (KNN): {f1_knn:.4f}")
    print(f"ROC AUC Score (KNN): {roc_auc_knn:.4f}")
else:
    print("\nO modelo KNN não suporta predict_proba para ROC AUC.")

print("\n--- Passo 9: Treinar e Avaliar um Modelo RandomForestClassifier ---")

rf_classifier = RandomForestClassifier(random_state=42)

param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10]
}

grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

print("\nIniciando GridSearchCV para RandomForest...")
grid_search_rf.fit(X_train_scaled, y_train)

print("\nMelhores parâmetros encontrados para RandomForest:")
print(grid_search_rf.best_params_)
print("\nMelhor pontuação (accuracy) encontrada na validação cruzada para RandomForest:")
print(f"{grid_search_rf.best_score_:.4f}")

best_rf_model = grid_search_rf.best_estimator_

print("\n--- Avaliando o Desempenho do Modelo RandomForest no conjunto de teste ---")

y_pred_rf = best_rf_model.predict(X_test_scaled)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nAcurácia do modelo RandomForest no conjunto de teste: {accuracy_rf:.4f}")

cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nMatriz de Confusão para RandomForest:")
print(cm_rf)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Não-ADHD', 'ADHD'], yticklabels=['Não-ADHD', 'ADHD'])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão (RandomForest)')
plt.show()

print("\nAnálise da Matriz de Confusão (RandomForest):")
tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
print(f"True Negatives (TN): {tn_rf}")
print(f"False Positives (FP): {fp_rf}")
print(f"False Negatives (FN): {fn_rf}")
print(f"True Positives (TP): {tp_rf}")

precision_rf = precision_score(y_test, y_pred_rf, average='binary')
recall_rf = recall_score(y_test, y_pred_rf, average='binary')
f1_rf = f1_score(y_test, y_pred_rf, average='binary')

if hasattr(best_rf_model, "predict_proba"):
    y_pred_proba_rf = best_rf_model.predict_proba(X_test_scaled)[:, 1]
    roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
    print(f"\nPrecision (RandomForest): {precision_rf:.4f}")
    print(f"Recall (Sensibilidade) (RandomForest): {recall_rf:.4f}")
    print(f"F1-Score (RandomForest): {f1_rf:.4f}")
    print(f"ROC AUC Score (RandomForest): {roc_auc_rf:.4f}")
else:
    print("\nO modelo RandomForest não suporta predict_proba para ROC AUC.")

print("\nModelos avaliados com sucesso!")

print("\n--- Passo 10: Análise de Importância de Features (RandomForest) ---")

feature_importances = best_rf_model.feature_importances_

features_df_importance = pd.DataFrame({ 
    'Feature': X.columns,
    'Importance': feature_importances
})

features_df_importance = features_df_importance.sort_values(by='Importance', ascending=False)

print("\nImportância das Features no Modelo RandomForest:")
print(features_df_importance)

plt.figure(figsize=(10, 7))
sns.barplot(x='Importance', y='Feature', data=features_df_importance, palette='viridis')
plt.title('Importância das Features no Modelo RandomForest')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print("\nAnálise de importância de features concluída com sucesso!")

joblib.dump(grid_search_rf.best_estimator_, 'random_forest_model.pkl')
print("Modelo RandomForest salvo como 'random_forest_model.pkl'")

joblib.dump(scaler, 'scaler.pkl')
print("Scaler salvo como 'scaler.pkl'")

print("\nModelos avaliados com sucesso!")
print("Modelo e Scaler salvos para uso na aplicação Streamlit.")