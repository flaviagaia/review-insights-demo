"""Sumarização de reviews com LangChain (LCEL) — estratégia map-reduce.

Por que map-reduce: um autor popular pode ter milhares de reviews — não
cabem em uma janela de contexto e enviá-las inteiras custaria caro.
1. MAP: blocos de reviews são resumidos em paralelo (`chain.batch`);
2. REDUCE: os resumos parciais viram um sumário executivo estruturado.

Otimização de custo: as reviews enviadas ao LLM são pré-selecionadas por
relevância (votos úteis + tamanho), não aleatórias.
"""
from __future__ import annotations

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .llm_chain import get_llm

MAP_PROMPT = ChatPromptTemplate.from_template(
    """Você é um analista de uma editora. Resuma em até 5 bullets os pontos
recorrentes (elogios e críticas) das avaliações de leitores abaixo.
Seja factual e cite aspectos concretos (enredo, personagens, clareza, preço etc).

### REVIEWS
{reviews}"""
)

REDUCE_PROMPT = ChatPromptTemplate.from_template(
    """Você é um analista sênior de uma editora. Com base nos resumos parciais
de avaliações de leitores sobre {entity}, escreva um sumário executivo com:
1. Percepção geral (1-2 frases)
2. Pontos fortes recorrentes
3. Críticas recorrentes
4. Recomendações acionáveis para a editora

### REVIEWS (resumos parciais)
{reviews}"""
)


def _select_reviews(df: pd.DataFrame, max_reviews: int = 60) -> pd.DataFrame:
    """Prioriza reviews com mais votos úteis e texto substantivo."""
    ranked = df.assign(
        _rank=df["helpful_votes"].fillna(0) + df["review_len"].clip(upper=300) / 100
    ).sort_values("_rank", ascending=False)
    return ranked.head(max_reviews)


def summarize_entity(df: pd.DataFrame, entity_name: str,
                     provider: str | None = None,
                     chunk_size: int = 20, max_reviews: int = 60) -> str:
    """Gera sumário executivo das reviews de um autor/gênero/livro."""
    llm, callbacks = get_llm(provider)
    cfg = {"callbacks": callbacks}

    selected = _select_reviews(df, max_reviews)
    texts = [
        f"[{r['review/score']:.0f}★] {r['review/summary']}: {r['review/text'][:600]}"
        for _, r in selected.iterrows()
    ]
    if not texts:
        return "Sem reviews suficientes para sumarizar."

    map_chain = MAP_PROMPT | llm | StrOutputParser()
    reduce_chain = REDUCE_PROMPT | llm | StrOutputParser()

    # Modo mock: uma única passada extrativa (map-reduce só agrega com LLM real)
    if getattr(llm, "_llm_type", "") == "extractive-mock":
        return map_chain.invoke({"reviews": "\n\n".join(texts)}, config=cfg)

    # MAP em paralelo (LangChain gerencia concorrência e retries)
    chunks = ["\n\n".join(texts[i:i + chunk_size])
              for i in range(0, len(texts), chunk_size)]
    partials = map_chain.batch([{"reviews": c} for c in chunks], config=cfg)

    # REDUCE
    return reduce_chain.invoke(
        {"entity": entity_name, "reviews": "\n\n".join(partials)}, config=cfg
    )
