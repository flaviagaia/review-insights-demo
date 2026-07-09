# 📚 Review Insights — MVP

**Demo ao vivo:** [flaviagaia.github.io/review-insights-demo](https://flaviagaia.github.io/review-insights-demo/)

Análise inteligente de avaliações de livros com NLP e IA Generativa: de dias de leitura manual
para segundos, com custo, segurança e qualidade monitorados desde o primeiro dia.

**Dados:** [Amazon Books Reviews (Kaggle)](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews) —
3 milhões de reviews públicas, processadas por amostragem aleatória (~378 mil) e exibidas com
identificadores de leitores **pseudonimizados** (hash SHA-256).

## 🧠 Mapa mental — o que foi usado para construir este MVP

```mermaid
mindmap
  root((Review Insights MVP))
    Dados
      Kaggle Amazon Books Reviews
        3M reviews, 2.7GB
      pyarrow streaming
        amostra de 13% em 10s
      pandas
        limpeza, dedup, HTML residual
      Checkpoints parquet
        pipeline em estágios
      Loader robusto a schemas
    NLP clássico, custo zero
      Sentimento por supervisão fraca
        TF-IDF e LogReg balanceada
        recall de críticas 0.82
      Tópicos NMF
      Aspectos por dicionário
        polaridade por sentença
      Ranking de leitores
        score composto
    Camada LLM sob demanda
      Cliente plugável
        mock, OpenAI, Bedrock
      Sumarização map-reduce
      RAG-lite
        respostas com fontes citadas
      Log de custos
        tokens, dólares, latência
    Aplicação
      POC Streamlit e Plotly
        3 abas testadas
      Demo HTML standalone
        Chart.js
        JSON embutido
        simulador de custo
        kill switch de orçamento
    Design padrão Mira Animator
      Temas light-minimal e mira-dark
      Glassmorphism e Tailwind
      AOS para entradas animadas
      D3.js
        loops contínuos
        metáfora orbital
    Publicação e segurança
      GitHub Pages
      Pseudonimização LGPD
      noindex, zero credenciais
      Fallback de CDN
```

## 🗺 Como navegar na demo

1. **📊 Análise** — escolha autor, gênero ou livro: KPIs animados, distribuição de notas,
   sentimento ao longo do tempo, aspectos elogiados/criticados e sumário executivo gerado
   sobre as reviews selecionadas.
2. **💬 Pergunte às reviews** — perguntas em linguagem natural com resposta ancorada em
   trechos citados `[R0]`, `[R1]`... (contrato anti-alucinação).
3. **📡 Monitoramento & FinOps** — custo acumulado por chamada, simulador de custo mensal
   por modelo (Haiku, Sonnet, GPT-4o-mini, Llama) e controle de orçamento com kill switch.

## 🔒 Segurança dos dados

- Reviews são públicas (dataset Kaggle acima); ainda assim, nenhum nome de usuário é exibido:
  IDs viram pseudônimos por hash ("Leitor EA91BB").
- Página com `noindex, nofollow`; nenhuma credencial ou chave de API no código.
- Nenhuma chamada externa além dos CDNs de bibliotecas (Tailwind, D3, Chart.js, AOS).

## ⚙️ Stack em uma linha

`pyarrow · pandas · scikit-learn (TF-IDF, LogReg, NMF) · estratégia map-reduce para LLM ·
Streamlit/Plotly (POC) · HTML + Chart.js + D3 + AOS + Tailwind (demo) · GitHub Pages`

---
*Desenvolvido por Flávia Guimarães Gaia Paula.*
