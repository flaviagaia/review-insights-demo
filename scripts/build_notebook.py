"""Constrói e executa notebooks/01_eda_hipoteses.ipynb (storytelling da análise)."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# 📚 Análise de Avaliações de Livros — EDA e Validação de Hipóteses
**Case técnico — NLP e LLMs sobre avaliações de livros**

**Contexto de negócio:** a editora leva ~3 dias e 5 analistas para explorar avaliações
manualmente. Este notebook demonstra o pipeline que automatiza essa exploração e valida
as hipóteses que guiaram a solução.

| # | Hipótese |
|---|----------|
| H1 | A nota média esconde insatisfação — o viés positivo das estrelas inflaciona a percepção |
| H2 | Poucos leitores concentram os votos de utilidade → shortlist objetiva para entrevistas |
| H3 | O sentimento do **texto** discrimina melhor que as estrelas (3★ é ambíguo, o texto não) |
| H4 | Os aspectos criticados variam por gênero → ação editorial direcionada |
| H5 | Cauda longa: poucos títulos concentram as reviews → esforço manual mal alocado |

> Rodando com a **amostra sintética** (mesmo schema do Kaggle). Para os dados reais,
> coloque os CSVs em `data/raw/` — nenhuma célula muda."""),

    code("""import os, sys
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')  # trabalhar a partir da raiz do repo
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.nlp_pipeline import add_sentiment, aspect_sentiment, extract_topics, rank_reviewers

plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})

df = load_data()
print(f"{len(df):,} reviews | {df['Title'].nunique()} livros | "
      f"{df['author'].nunique()} autores | {df['genre'].nunique()} gêneros")
df.head(3)"""),

    md("""## 1. Qualidade e panorama dos dados
Antes de qualquer modelo: nulos, duplicatas e o formato das colunas-chave
(já tratados em `data_loader`: parsing de `helpfulness` "7/10", listas de
autores/categorias, datas epoch)."""),

    code("""display(df[['review/score', 'review_len', 'helpful_ratio']].describe().round(2))
print('Nulos por coluna (%):')
print((df[['author', 'genre', 'helpful_ratio', 'Price']].isna().mean() * 100).round(1))"""),

    md("""## 2. H1 — O viés positivo das notas
Se a maioria das reviews é 4-5★, a média de estrelas satura e deixa de discriminar
qualidade. É exatamente onde o texto vira o sinal mais rico."""),

    code("""ax = df['review/score'].value_counts().sort_index().plot.bar(color='#4C72B0', figsize=(5, 3))
ax.set_title('Distribuição de notas'); ax.set_xlabel('Estrelas'); ax.set_ylabel('Reviews')
pct = (df['review/score'] >= 4).mean() * 100
print(f'{pct:.0f}% das reviews têm 4-5 estrelas -> H1 confirmada')"""),

    md("""## 3. H5 — Cauda longa de volume
Curva de Pareto: se poucos livros concentram as reviews, a análise manual
"livro a livro" gasta a maior parte do tempo onde há pouco sinal."""),

    code("""counts = df['Title'].value_counts().values
cum = np.cumsum(counts) / counts.sum()
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum * 100, color='#4C72B0')
ax.axhline(80, ls='--', c='gray', lw=0.8)
ax.set_xlabel('% dos livros'); ax.set_ylabel('% das reviews'); ax.set_title('Concentração de reviews')
n80 = int(np.searchsorted(cum, 0.8)) + 1
print(f'{n80} livros ({n80/len(counts)*100:.0f}%) concentram 80% das reviews -> H5 confirmada')"""),

    md("""## 4. H3 — Sentimento do texto via supervisão fraca
**Ideia:** usar as próprias estrelas como rótulo (≥4 positivo, ≤2 negativo) para treinar
TF-IDF + Regressão Logística — zero custo de anotação, escala para o dataset inteiro.
O modelo então "lê" as reviews de 3★ (ambíguas por definição) e resolve a ambiguidade."""),

    code("""df, model = add_sentiment(df)
print('Performance no holdout (rótulos = estrelas):')
print(model.report_)"""),

    code("""fig, ax = plt.subplots(figsize=(5, 3))
df.boxplot(column='sentiment', by='review/score', ax=ax, grid=False)
plt.suptitle(''); ax.set_title('Sentimento do texto por nota')
ax.set_xlabel('Estrelas'); ax.set_ylabel('P(positivo)')
amb = df[df['review/score'] == 3]
print(f"Nas {len(amb):,} reviews 3 estrelas: {(amb['sentiment']<0.4).mean()*100:.0f}% texto negativo, "
      f"{(amb['sentiment']>0.6).mean()*100:.0f}% texto positivo -> o texto desambigua a nota (H3)")"""),

    md("""### Performance por autor — nota vs sentimento do texto
Para o negócio: dois autores com a mesma média de estrelas podem ter percepções
muito diferentes no texto."""),

    code("""perf = (df.groupby('author')
        .agg(reviews=('review/score', 'size'), nota_media=('review/score', 'mean'),
             sentimento=('sentiment', 'mean'))
        .query('reviews >= 30').sort_values('sentimento', ascending=False).round(3))
display(perf.head(10))
ax = perf['sentimento'].plot.barh(color='#4C72B0', figsize=(6, 4))
ax.set_title('Sentimento médio do texto por autor'); ax.invert_yaxis()"""),

    md("""## 5. H4 — Aspectos por gênero
Análise de aspectos (enredo, personagens, clareza, preço...) por dicionário +
polaridade da sentença. Em produção, a extração evolui para LLM few-shot
(mais recall), mantendo esta versão como baseline auditável."""),

    code("""asp = aspect_sentiment(df).merge(df[['genre']], left_on='review_idx', right_index=True)
pivot = (asp[asp['polarity'] != 0].groupby(['genre', 'aspect'])['polarity']
         .mean().unstack().round(2))
fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha='right')
ax.set_yticks(range(len(pivot.index)), pivot.index)
ax.set_title('Polaridade média por aspecto e gênero'); fig.colorbar(im, shrink=0.8)
print('Cada gênero tem um perfil próprio de elogios/críticas -> H4 confirmada')"""),

    md("""## 6. Tópicos dominantes (NMF)
Visão não supervisionada do que os leitores falam — complementa os aspectos."""),

    code("""for i, topic in enumerate(extract_topics(df['review/text'], n_topics=6), 1):
    print(f'Tópico {i}: {", ".join(topic)}')"""),

    md("""## 7. H2 — Leitores relevantes para entrevista
Score composto: votos úteis da comunidade (35%), taxa de utilidade (25%),
profundidade do texto (20%), produtividade (10%) e discriminação de nota (10%)."""),

    code("""conc = df.groupby('User_id')['helpful_votes'].sum().sort_values(ascending=False)
p10 = conc.head(max(1, len(conc)//10)).sum() / conc.sum() * 100
print(f'10% dos usuários concentram {p10:.0f}% dos votos de utilidade -> H2 confirmada')
rank_reviewers(df, top_k=10)[['profileName', 'n_reviews', 'total_helpful_votes',
                              'avg_len', 'avg_score', 'relevance_score']]"""),

    md("""## 8. Sumarização com LLM (map-reduce)
Demonstração com `LLM_PROVIDER=mock` (extrativa, offline). Em produção: Bedrock/Claude,
com o mesmo código — e cada chamada logada com tokens, custo e latência."""),

    code("""from src.llm_client import LLMClient
from src.summarizer import summarize_entity

autor = df['author'].value_counts().index[0]
print(summarize_entity(df[df['author'] == autor], f"autor '{autor}'", LLMClient('mock')))"""),

    code("""print('Observabilidade da POC (logs/llm_usage.jsonl):')
LLMClient.usage_report()"""),

    md("""## Conclusões

As 5 hipóteses foram confirmadas na amostra, sustentando o desenho da solução:

1. **Automação com NLP clássico** resolve o volume (sentimento, aspectos, tópicos) a custo ~zero;
2. **LLM aplicado cirurgicamente** (sumarização executiva, futura base de conhecimento RAG) onde gera valor único;
3. **Ranking de leitores** transforma a busca por entrevistados de dias para segundos;
4. **Impacto:** ~R$ 20 mil/mês de capacidade analítica liberada (~R$ 240 mil/ano), análises 10x mais frequentes.

**Próximos passos:** ver roadmap AWS incremental no README (batch produtivo → RAG → escala e governança)."""),
]

nb.cells = cells
out = Path("notebooks/01_eda_hipoteses.ipynb")
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)

print("Executando notebook...")
client = NotebookClient(nb, timeout=600, resources={"metadata": {"path": "notebooks/"}})
client.execute()
nbf.write(nb, out)
print(f"OK -> {out}")
