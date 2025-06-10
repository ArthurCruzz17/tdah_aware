import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="TDAH Aware: Apoio Clínico Inteligente",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto"
)

st.warning("""
    **AVISO IMPORTANTE:** Esta ferramenta é um protótipo de Inteligência Artificial e foi desenvolvida para **auxiliar profissionais da saúde** (como psicólogos, psiquiatras e neurologistas) na **avaliação e identificação de indícios de TDAH** em seus pacientes.

    **Ela NÃO substitui o julgamento clínico, a experiência profissional ou o diagnóstico médico.** Os resultados devem ser interpretados **sempre** em conjunto com a avaliação clínica completa do paciente.
""")

st.title("🧠 TDAH Aware: Apoio Clínico Inteligente")
st.write("Insira os dados do paciente abaixo para obter uma análise computacional complementar.")

try:
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Erro: Modelo ou scaler não encontrados! Certifique-se de que 'random_forest_model.pkl' e 'scaler.pkl' estão no mesmo diretório.")
    st.stop()


st.header("Dados Demográficos e Clínicos do Paciente")


with st.form("patient_data_form"):
    st.markdown("---") 

    st.subheader("1. Informações Básicas")
    age = st.slider("Idade do Paciente", min_value=1, max_value=90, value=25)
    sex = st.radio("Sexo do Paciente", options=["Masculino", "Feminino", "Outro"])
    sex_encoded = 1 if sex == "Feminino" else (0 if sex == "Masculino" else -1) 

    st.markdown("---") 

    st.subheader("2. Pontuações de Questionários Validados")
    st.markdown("*(Insira as pontuações do paciente obtidas nos respectivos questionários de avaliação)*")

    st.write("**WURS (Wender Utah Rating Scale):** Pontuação referente aos sintomas de TDAH na infância (baseado nas recordações do paciente adulto).")
    wurs = st.slider("WURS Score (0-100)", min_value=0, max_value=100, value=40)

    st.write("**ASRS (Adult ADHD Self-Report Scale):** Pontuação total que avalia sintomas de desatenção e hiperatividade em adultos.")
    asrs = st.slider("ASRS Score (0-100)", min_value=0, max_value=100, value=42)

    st.write("**MADRS (Montgomery-Åsberg Depression Rating Scale):** Pontuação que avalia a gravidade da depressão no paciente.")
    madrs = st.slider("MADRS Score (0-60)", min_value=0, max_value=60, value=12)

    st.write("**HADS-A (Hospital Anxiety and Depression Scale - Ansiedade):** Pontuação de ansiedade do paciente.")
    hads_a = st.slider("HADS-A Score (0-21)", min_value=0, max_value=21, value=9)

    st.write("**HADS-D (Hospital Anxiety and Depression Scale - Depressão):** Pontuação de depressão do paciente.")
    hads_d = st.slider("HADS-D Score (0-21)", min_value=0, max_value=21, value=5)

    st.write("**MDQ_POS (Mood Disorder Questionnaire):** Indica se o paciente teve um resultado positivo no rastreamento para transtorno bipolar (1 para Sim, 0 para Não).")
    mdq_pos = st.radio("MDQ_POS (Resultado Positivo no Questionário MDQ para o paciente)", options=["Sim", "Não"])
    mdq_pos_encoded = 1 if mdq_pos == "Sim" else 0

    st.markdown("---") 

    st.subheader("3. Dados Objetivos de Desempenho e Atividade")
    st.markdown("*(Insira as pontuações do paciente obtidas em testes objetivos como o CPT-II e dados de sensores, se disponíveis)*")
    st.write("**CPT-II (Conners' Continuous Performance Test II):** Métricas objetivas de atenção e impulsividade do paciente.")
    raw_score_omissions = st.slider("CPT-II: Taxa de Omissões", min_value=0, max_value=20, value=5, help="Número de vezes que o paciente falhou em responder a um estímulo alvo.")
    raw_score_commissions = st.slider("CPT-II: Taxa de Comissões", min_value=0, max_value=20, value=5, help="Número de vezes que o paciente respondeu a um estímulo não-alvo.")
    raw_score_hitrt = st.slider("CPT-II: Tempo de Reação Médio (ms)", min_value=100, max_value=800, value=350, help="Tempo médio que o paciente levou para responder aos estímulos alvo.")
    raw_score_varse = st.slider("CPT-II: Variabilidade do Erro Padrão", min_value=0.0, max_value=100.0, value=10.0, help="Medida da consistência do tempo de reação do paciente.")
    raw_score_dprime = st.slider("CPT-II: Discriminação (D-Prime)", min_value=0.0, max_value=5.0, value=1.0, help="Capacidade do paciente de distinguir estímulos alvo de não-alvo.")

    st.write("**Dados de Acelerômetro (Comportamento Motor):** Valores médios derivados da análise de atividade do paciente durante um período (ex: em uma tarefa específica).")
    acc_mean = st.number_input("ACC__mean (Média da Aceleração)", value=180.0, format="%.4f", help="Valor médio da aceleração registrada.")
    acc_variance = st.number_input("ACC__variance (Variância da Aceleração)", value=80000.0, format="%.4f", help="Variabilidade da aceleração registrada, indicando irregularidade ou agitação.")
    acc_maximum = st.number_input("ACC__maximum (Pico de Aceleração)", value=2500.0, format="%.4f", help="Valor máximo de aceleração registrado.")

    submit_button = st.form_submit_button("Analisar Indícios no Paciente")

    user_input = pd.DataFrame([[
        acc_mean, acc_variance, acc_maximum, age, sex_encoded,
        wurs, asrs, madrs, hads_a, hads_d, mdq_pos_encoded,
        raw_score_omissions, raw_score_commissions, raw_score_hitrt,
        raw_score_varse, raw_score_dprime
    ]], columns=[
        'ACC__mean', 'ACC__variance', 'ACC__maximum', 'AGE', 'SEX',
        'WURS', 'ASRS', 'MADRS', 'HADS_A', 'HADS_D', 'MDQ_POS',
        'Raw Score Omissions', 'Raw Score Commissions', 'Raw Score HitRT',
        'Raw Score VarSE', 'Raw Score DPrime'
    ])

    user_input_scaled = scaler.transform(user_input) 

    if submit_button:
        prediction = model.predict(user_input_scaled)[0]
        prediction_proba = model.predict_proba(user_input_scaled)[0]
    else:
        
        prediction = None  
        prediction_proba = [0.5, 0.5] 


if submit_button:
    st.subheader("📊 Resultado da Análise Computacional do Paciente")

    if prediction == 1:
        st.error("❗ **INDÍCIOS SIGNIFICATIVOS DE TDAH DETECTADOS** ❗")
        st.write(f"A probabilidade de o paciente apresentar TDAH é de **{prediction_proba[1] * 100:.2f}%**.")
        st.markdown("""
        **Observações para o Profissional:**
        Com base nos dados inseridos, a ferramenta identificou um padrão de características que **sugerem fortemente a presença de TDAH** no paciente.
        """)
        st.markdown("""
        **Pontos de Destaque:**
        - **Questionários Clínicos (WURS, ASRS):** As pontuações elevadas nesses questionários são indicadores chaves, alinhando-se a relatos de sintomas persistentes de desatenção e hiperatividade/impulsividade.
        - **CPT-II (Omissões, Comissões, Variabilidade):** Métricas objetivas de desempenho de atenção e controle inibitório são cruciais. Padrões específicos de erros ou variabilidade no tempo de reação corroboram os indícios.
        - **Dados de Acelerômetro:** Podem fornecer insights adicionais sobre o comportamento motor e a agitação.

        **Sugestões para o Profissional:**
        * **Aprofundar a Anamnese:** Explorar detalhadamente o histórico de desenvolvimento, escolar e familiar, e o impacto dos sintomas nas diversas esferas da vida do paciente.
        * **Avaliações Complementares:** Considerar o uso de outras escalas validadas ou encaminhamento para neuropsicólogo para uma bateria de testes mais extensa, se ainda não realizado.
        * **Plano Terapêutico:** Este resultado pode apoiar a formulação de um plano de tratamento (farmacológico e/ou não farmacológico) e estratégias de manejo personalizadas.
        """)
    else:
        st.success("✅ **NENHUM INDÍCIO SIGNIFICATIVO DE TDAH DETECTADO POR ESTA ANÁLISE** ✅")
        st.write(f"A probabilidade de o paciente **não** apresentar TDAH é de **{prediction_proba[0] * 100:.2f}%**.")
        st.markdown("""
        **Observações para o Profissional:**
        De acordo com a análise computacional, os dados inseridos **não sugerem a presença de TDAH** por esta ferramenta.
        """)
        st.markdown("""
        **Sugestões para o Profissional:**
        * **Investigação Diferencial:** Se os sintomas do paciente persistirem, é fundamental continuar a investigação para outras possíveis causas (ex: ansiedade, depressão, outros transtornos de neurodesenvolvimento, condições médicas).
        * **Revisão Contextual:** Reavaliar o contexto de vida do paciente, fatores estressores ou outras condições que possam estar contribuindo para os sintomas.
        * **Monitoramento:** Manter o monitoramento dos sintomas e considerar reavaliações periódicas, se clinicamente indicado.
        """)

st.markdown("---")
st.warning("Lembre-se: Esta análise é um **auxílio** para o profissional. O **diagnóstico final e o plano de tratamento** são de responsabilidade exclusiva do médico ou profissional de saúde qualificado.")

st.markdown("---")
st.caption("Desenvolvido por Arthur Cruz para a matéria de Inteligência Artificial, do professor Wilson Andrade.")