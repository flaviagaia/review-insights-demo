"""Q&A sobre as reviews (RAG) com LangChain: retriever custom + chain LCEL.

O retriever TF-IDF implementa a interface `BaseRetriever` do LangChain —
em produção, trocar por embeddings + vector store (Bedrock Knowledge Bases,
pgvector) é substituir ESTA classe, sem tocar no chain. O contrato se
mantém: resposta SEMPRE ancorada em trechos citáveis (anti-alucinação).
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .llm_chain import get_llm

# Expansão de consulta PT->EN: reviews em inglês + retriever lexical na POC.
# Em produção, embeddings multilíngues (Titan v2) eliminam esta ponte.
PT_EN = {
    "negócios": "business economics", "negocios": "business economics",
    "ficção": "fiction story novel", "ficcao": "fiction story novel",
    "romance": "romance", "mistério": "mystery thriller", "misterio": "mystery thriller",
    "história": "history", "historia": "history", "autoajuda": "self-help",
    "tecnologia": "computers technology", "ritmo": "pacing slow fast paced",
    "personagens": "characters character", "escrita": "writing prose written",
    "enredo": "plot storyline", "final": "ending conclusion",
    "críticas": "negative criticism problem disappointing bad",
    "criticas": "negative criticism problem disappointing bad",
    "elogios": "praise great excellent loved", "problema": "problem issue broken",
    "problemas": "problems issues errors", "preço": "price expensive value",
    "preco": "price expensive value", "caro": "expensive price",
    "formatação": "formatting format edition", "formatacao": "formatting format edition",
    "kindle": "kindle edition formatting", "exemplos": "examples case studies",
    "clareza": "clarity clear confusing", "profundidade": "depth superficial thorough",
    "erros": "errors wrong inaccurate", "leitores": "readers reviewers",
    "recorrentes": "recurring common", "livros": "books book",
}

QA_PROMPT = ChatPromptTemplate.from_template(
    """Você é um analista de uma editora. Responda a pergunta do usuário
usando SOMENTE os trechos de reviews abaixo. Cite as fontes pelo id entre
colchetes (ex: [R12]). Se os trechos não sustentarem uma resposta, diga
explicitamente que não há evidência suficiente na base.

PERGUNTA: {question}

### REVIEWS
{context}"""
)


def _expand_query(question: str) -> str:
    extra = [en for pt, en in PT_EN.items() if pt in question.lower()]
    return question + " " + " ".join(extra)


class TfidfReviewRetriever(BaseRetriever):
    """Retriever lexical sobre as reviews (interface LangChain).

    Trocável por Bedrock Knowledge Bases / pgvector em produção,
    mantendo o mesmo chain de resposta.
    """

    df: Any = None
    vectorizer: Any = None
    matrix: Any = None
    k: int = 8

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, k: int = 8) -> "TfidfReviewRetriever":
        df = df.reset_index(drop=True)
        vec = TfidfVectorizer(max_features=30_000, ngram_range=(1, 2),
                              stop_words="english", min_df=2)
        corpus = (df["Title"].fillna("") + " " + df["author"].fillna("")
                  + " " + df["genre"].fillna("") + " "
                  + df["review/summary"].fillna("") + " " + df["review/text"])
        matrix = vec.fit_transform(corpus)
        return cls(df=df, vectorizer=vec, matrix=matrix, k=k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        q = self.vectorizer.transform([_expand_query(query)])
        sims = cosine_similarity(q, self.matrix).ravel()
        top = sims.argsort()[::-1][: self.k]
        docs = []
        for rank, i in enumerate(top):
            if sims[i] <= 0.01:
                continue
            r = self.df.iloc[i]
            docs.append(Document(
                page_content=str(r["review/text"])[:400],
                metadata={"id": f"R{rank}", "author": r["author"],
                          "title": r["Title"], "score": float(r["review/score"]),
                          "similarity": float(sims[i])},
            ))
        return docs


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata['id']}] ({d.metadata['author']} — {d.metadata['title']}, "
        f"{d.metadata['score']:.0f}★) {d.page_content}"
        for d in docs
    )


def answer_question(question: str, retriever: TfidfReviewRetriever,
                    provider: str | None = None) -> tuple[str, pd.DataFrame]:
    """Retorna (resposta, fontes usadas). Chain: retrieve -> prompt -> llm."""
    llm, callbacks = get_llm(provider)

    docs = retriever.invoke(question)
    if not docs:
        return "Não há evidência suficiente na base para responder.", pd.DataFrame()

    chain = QA_PROMPT | llm | StrOutputParser()
    answer = chain.invoke(
        {"question": question, "context": _format_docs(docs)},
        config={"callbacks": callbacks},
    )
    sources = pd.DataFrame([
        {"id": d.metadata["id"], "author": d.metadata["author"],
         "Title": d.metadata["title"], "review/score": d.metadata["score"],
         "review/text": d.page_content, "similarity": round(d.metadata["similarity"], 3)}
        for d in docs
    ])
    return answer, sources
