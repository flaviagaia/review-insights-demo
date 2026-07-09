"""POC — Assistente de Análise de Reviews (Streamlit).

Execução:  streamlit run app/streamlit_app.py

Três abas:
1. Análise: KPIs, sentimento, aspectos, sumário LLM, shortlist de leitores.
2. Pergunte às reviews: Q&A (RAG-lite) com citação de fontes.
3. Monitoramento & FinOps: custo, latência, simulador e orçamento —
   o painel que em produção vira CloudWatch/Grafana.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_data
from src.llm_chain import LOG_PATH, usage_report
from src.nlp_pipeline import add_sentiment, aspect_sentiment, rank_reviewers
from src.qa import TfidfReviewRetriever, answer_question
from src.summarizer import summarize_entity

st.set_page_config(page_title="Review Insights — POC", layout="wide")

# US$/1M tokens (entrada, saída) para o simulador de custos
SIM_PRICING = {
    "Claude Haiku (Bedrock)": (0.80, 4.00),
    "Claude Sonnet (Bedrock)": (3.00, 15.00),
    "GPT-4o-mini (OpenAI)": (0.15, 0.60),
    "Llama 3 70B (Bedrock)": (0.72, 0.72),
}
USD_BRL = 5.5


@st.cache_data(show_spinner="Carregando e processando dados...")
def get_data():
    df = load_data()
    df, _ = add_sentiment(df)
    return df


@st.cache_resource(show_spinner="Indexando reviews para Q&A...")
def get_retriever():
    return TfidfReviewRetriever.from_dataframe(get_data())


df = get_data()

st.title("📚 Review Insights — POC")
st.caption("De 3 dias de análise manual para segundos. NLP clássico (custo zero) "
           "+ LLM sob demanda, com custo e qualidade monitorados.")

st.sidebar.header("Configuração")
provider_sel = st.sidebar.selectbox("Provedor LLM", ["mock (offline/demo)", "openai", "bedrock"])
provider = provider_sel.split()[0]
language = st.sidebar.selectbox("Idioma da resposta (LLM real)", ["português", "english"])

tab_analise, tab_qa, tab_monitor = st.tabs(
    ["📊 Análise", "💬 Pergunte às reviews", "📡 Monitoramento & FinOps"]
)

# ====================== ABA 1 — ANÁLISE ======================
with tab_analise:
    st.sidebar.header("Filtro de análise")
    mode = st.sidebar.radio("Analisar por", ["Autor", "Gênero", "Livro"])
    col = {"Autor": "author", "Gênero": "genre", "Livro": "Title"}[mode]
    options = df[col].value_counts()
    choice = st.sidebar.selectbox(
        f"{mode} ({len(options)} disponíveis)",
        options.index, format_func=lambda x: f"{x} ({options[x]} reviews)",
    )
    sub = df[df[col] == choice]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", f"{len(sub):,}")
    c2.metric("Nota média", f"{sub['review/score'].mean():.2f} ★")
    c3.metric("Sentimento do texto", f"{sub['sentiment'].mean():.0%} positivo")
    c4.metric("Leitores únicos", f"{sub['User_id'].nunique():,}")

    g1, g2 = st.columns(2)
    with g1:
        fig = px.histogram(sub, x="review/score", nbins=5, title="Distribuição de notas")
        st.plotly_chart(fig, width="stretch")
    with g2:
        ts = sub.groupby("review_year")["sentiment"].mean().reset_index()
        fig = px.line(ts, x="review_year", y="sentiment", title="Sentimento ao longo do tempo")
        st.plotly_chart(fig, width="stretch")

    st.subheader("O que os leitores elogiam e criticam")
    asp = aspect_sentiment(sub)
    if not asp.empty:
        agg = (asp[asp["polarity"] != 0].groupby("aspect")
               .agg(mencoes=("polarity", "size"), polaridade=("polarity", "mean"))
               .reset_index().sort_values("mencoes", ascending=False))
        fig = px.bar(agg, x="aspect", y="mencoes", color="polaridade",
                     color_continuous_scale="RdYlGn", range_color=[-1, 1],
                     title="Aspectos mencionados (cor = polaridade)")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Sumário executivo (LLM)")
    if st.button("🧠 Gerar sumário executivo", type="primary"):
        with st.spinner("Sumarizando reviews (map-reduce)..."):
            summary = summarize_entity(sub, f"{mode.lower()} '{choice}'", provider, language=language)
        st.markdown(summary)

    st.subheader("Leitores recomendados para entrevista")
    st.dataframe(
        rank_reviewers(sub if len(sub) > 200 else df, top_k=8)
        [["profileName", "n_reviews", "total_helpful_votes", "avg_len",
          "avg_score", "relevance_score"]],
        width="stretch",
    )

# ====================== ABA 2 — Q&A (RAG-lite) ======================
with tab_qa:
    st.subheader("Pergunte em linguagem natural — respostas com fontes")
    st.caption("Recuperação TF-IDF (offline) + LLM. Toda resposta cita as reviews-fonte "
               "[R0], [R1]... — a defesa estrutural contra alucinação. Em produção: "
               "embeddings + Bedrock Knowledge Bases, mesmo contrato de citação.")
    examples = [
        "Quais as críticas recorrentes aos livros de negócios?",
        "O que os leitores acham do ritmo dos livros de ficção?",
        "Algum autor tem problema de formatação nas edições Kindle?",
    ]
    ex = st.selectbox("Exemplos de pergunta", ["(escreva a sua)"] + examples)
    question = st.text_input("Sua pergunta", value="" if ex.startswith("(") else ex)
    if st.button("🔎 Responder", type="primary") and question.strip():
        with st.spinner("Recuperando reviews e gerando resposta..."):
            answer, sources = answer_question(question, get_retriever(), provider, language=language)
        st.markdown(answer)
        if not sources.empty:
            with st.expander(f"📎 Fontes utilizadas ({len(sources)} reviews)"):
                view = sources[["id", "author", "Title", "review/score", "review/text",
                                "similarity"]].copy()
                view["review/text"] = view["review/text"].str[:200] + "..."
                st.dataframe(view, width="stretch")

# ====================== ABA 3 — MONITORAMENTO & FINOPS ======================
with tab_monitor:
    st.subheader("Observabilidade da POC")
    st.caption("Cada chamada LLM é logada em logs/llm_usage.jsonl (provedor, modelo, "
               "tokens, custo estimado, latência). Em produção, o mesmo telemetria vai "
               "para CloudWatch (dashboards/alarmes) ou Grafana; MLflow versiona os "
               "experimentos do modelo de sentimento.")

    rep = usage_report()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Chamadas LLM", rep["calls"])
    m2.metric("Tokens (in+out)", f"{rep['tokens_in'] + rep['tokens_out']:,}")
    m3.metric("Custo acumulado", f"US$ {rep['cost_usd']:.4f}")
    m4.metric("Latência média", f"{rep['avg_latency_s']:.2f}s")

    if LOG_PATH.exists():
        log = pd.DataFrame(
            json.loads(line) for line in LOG_PATH.read_text().splitlines() if line
        )
        log["ts"] = pd.to_datetime(log["ts"])
        log["custo acumulado (US$)"] = log["cost_usd"].cumsum()
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(px.line(log, x="ts", y="custo acumulado (US$)",
                                    title="Custo acumulado por chamada"), width="stretch")
        with g2:
            st.plotly_chart(px.scatter(log, x="ts", y="latency_s", color="model",
                                       title="Latência por chamada (s)"), width="stretch")

    st.divider()
    st.subheader("Simulador de custo mensal por modelo")
    s1, s2, s3 = st.columns(3)
    analises_mes = s1.slider("Análises/mês", 50, 2000, 500, step=50)
    tokens_in = s2.number_input("Tokens de entrada/análise", 5_000, 100_000, 20_000, step=5_000)
    tokens_out = s3.number_input("Tokens de saída/análise", 500, 10_000, 2_000, step=500)

    sim = pd.DataFrame([
        {"Modelo": nome,
         "Custo/análise (US$)": tokens_in / 1e6 * pin + tokens_out / 1e6 * pout}
        for nome, (pin, pout) in SIM_PRICING.items()
    ])
    sim["Custo mensal (US$)"] = sim["Custo/análise (US$)"] * analises_mes
    sim["Custo mensal (R$)"] = sim["Custo mensal (US$)"] * USD_BRL
    st.plotly_chart(px.bar(sim, x="Modelo", y="Custo mensal (R$)",
                           title=f"Custo LLM mensal para {analises_mes} análises",
                           text_auto=".2f"), width="stretch")
    st.caption("Estratégia de cascata (padrão validado em produção): modelo barato como "
               "principal e escalonamento ao modelo premium apenas quando a confiança cai — "
               "aqui, Haiku no map e Sonnet só no reduce. Referência: análise manual custa ~R$ 714.")

    st.divider()
    st.subheader("Controle de orçamento (kill switch)")
    b1, b2 = st.columns([1, 2])
    budget = b1.number_input("Orçamento mensal de LLM (US$)", 10.0, 5000.0, 100.0, step=10.0)
    consumo = rep["cost_usd"]
    pct = min(consumo / budget, 1.0) if budget else 0
    b2.progress(pct, text=f"Consumido: US$ {consumo:.4f} de US$ {budget:.0f} ({pct:.1%})")
    if pct >= 0.8:
        st.error("⛔ Acima de 80% do orçamento: em produção, alarme via AWS Budgets + "
                 "SNS e bloqueio automático de novas chamadas (kill switch).")
    else:
        st.success("✅ Dentro do orçamento. Em produção: AWS Budgets + tags de custo por "
                   "feature, com alarme em 80% e bloqueio em 100%.")
