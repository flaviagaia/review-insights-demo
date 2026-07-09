"""Pipeline de NLP clássico (100% offline, custo zero).

Componentes:
1. Sentimento por supervisão fraca: as estrelas rotulam o treino
   (>=4 positivo, <=2 negativo) de um TF-IDF + Regressão Logística.
   O modelo então estima o sentimento DO TEXTO, inclusive das reviews
   de 3 estrelas — onde a nota não diz nada, mas o texto diz muito.
2. Tópicos via NMF sobre TF-IDF.
3. Análise de aspectos por dicionário + sentimento da sentença.

Racional da arquitetura: tarefas de alto volume (sentimento, aspectos,
tópicos) rodam com NLP clássico a custo ~zero; o LLM (src/llm_client.py)
é reservado para o que ele faz de único — sumarização e Q&A — reduzindo
o custo total da solução em ordens de magnitude.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# aspectos e termos-gatilho (extensível; em produção: extração via LLM few-shot)
ASPECT_KEYWORDS = {
    "enredo": ["plot", "story", "storyline", "twist", "narrative"],
    "personagens": ["character", "characters", "protagonist", "villain", "hero"],
    "ritmo": ["pacing", "pace", "slow", "fast paced", "drags"],
    "escrita": ["writing", "prose", "written", "sentence", "editing", "typos"],
    "final": ["ending", "conclusion", "finale"],
    "clareza": ["clarity", "clear", "confusing", "explained", "organized", "structured"],
    "exemplos": ["example", "examples", "case studies", "case study"],
    "praticidade": ["practical", "actionable", "apply", "applied", "advice"],
    "profundidade": ["depth", "thorough", "superficial", "surface", "research"],
    "precisão": ["accurate", "accuracy", "errors", "wrong", "factual"],
    "edição/formato": ["edition", "format", "formatting", "print", "kindle", "code samples"],
    "preço": ["price", "expensive", "value", "money", "worth"],
}

_NEG_WORDS = re.compile(
    r"\b(not|no|never|poorly|bad|terrible|awful|waste|disappoint\w*|boring|"
    r"weak|worst|confusing|shallow|slow|broken|wrong|errors?|flat|predictable|"
    r"repetitive|rushed|letdown|outdated|derivative|clumsy|unreadable|superficial)\b",
    re.I,
)
_POS_WORDS = re.compile(
    r"\b(great|excellent|wonderful|brilliant|masterpiece|loved?|best|gripping|"
    r"beautiful\w*|perfect|satisfying|memorable|immersive|rich|clever|elegant|"
    r"remarkable|impressive|solid|useful|actionable|recommend\w*|hooked|original)\b",
    re.I,
)


def bayes_smooth(mean, n, prior_mean, strength: float = 50.0):
    """Suavização bayesiana (empirical Bayes) para médias com n pequeno.

    Um autor com 12 reviews não deve competir em ranking com um de 1.200:
    a média encolhe em direção ao prior (ex.: média do gênero) proporcionalmente
    à escassez de evidência. `strength` é o pseudo-n do prior.
    """
    return (mean * n + prior_mean * strength) / (n + strength)


class SentimentModel:
    """Classificador de sentimento treinado por supervisão fraca (estrelas).

    `calibrate=True` embrulha a LogReg em CalibratedClassifierCV (sigmoid):
    probabilidades calibradas custam ~3x o tempo de treino, mas tornam o
    score interpretável como probabilidade real — recomendado em produção.
    """

    def __init__(self, calibrate: bool = False) -> None:
        from sklearn.calibration import CalibratedClassifierCV

        self.vectorizer = TfidfVectorizer(
            max_features=20_000, ngram_range=(1, 2), stop_words="english", min_df=2
        )
        # class_weight="balanced": a base real é ~87% positiva e o caso de uso
        # é justamente encontrar críticas — recall da classe negativa importa.
        base = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        self.clf = (CalibratedClassifierCV(base, method="sigmoid", cv=3)
                    if calibrate else base)
        self.report_: str | None = None

    def fit(self, df: pd.DataFrame) -> "SentimentModel":
        labeled = df[df["review/score"] != 3]
        y = (labeled["review/score"] >= 4).astype(int)
        x_tr, x_te, y_tr, y_te = train_test_split(
            labeled["review/text"], y, test_size=0.2, random_state=42, stratify=y
        )
        self.clf.fit(self.vectorizer.fit_transform(x_tr), y_tr)
        y_pred = self.clf.predict(self.vectorizer.transform(x_te))
        self.report_ = classification_report(y_te, y_pred, target_names=["neg", "pos"])
        return self

    def score(self, texts: pd.Series) -> np.ndarray:
        """Probabilidade de sentimento positivo em [0, 1]."""
        return self.clf.predict_proba(self.vectorizer.transform(texts))[:, 1]


def add_sentiment(df: pd.DataFrame) -> tuple[pd.DataFrame, SentimentModel]:
    model = SentimentModel().fit(df)
    df = df.copy()
    df["sentiment"] = model.score(df["review/text"])
    df["sentiment_label"] = pd.cut(
        df["sentiment"], [0, 0.4, 0.6, 1.0], labels=["negativo", "neutro", "positivo"]
    )
    return df, model


def extract_topics(texts: pd.Series, n_topics: int = 6, n_words: int = 8) -> list[list[str]]:
    """Tópicos dominantes via NMF (interpretáveis para o time de negócio)."""
    vec = TfidfVectorizer(max_features=5000, stop_words="english", min_df=3)
    x = vec.fit_transform(texts)
    n_topics = min(n_topics, max(2, x.shape[0] // 50))
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=400)
    nmf.fit(x)
    vocab = np.array(vec.get_feature_names_out())
    return [list(vocab[np.argsort(comp)[::-1][:n_words]]) for comp in nmf.components_]


def _sentence_polarity(sentence: str) -> int:
    pos = len(_POS_WORDS.findall(sentence))
    neg = len(_NEG_WORDS.findall(sentence))
    return int(np.sign(pos - neg))


def aspect_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada review, detecta aspectos mencionados e a polaridade local
    (sentença onde o aspecto aparece). Retorna long-format:
    [index da review, aspecto, polaridade -1/0/+1]."""
    records = []
    for idx, text in df["review/text"].items():
        for sentence in re.split(r"(?<=[.!?])\s+", str(text)):
            low = sentence.lower()
            for aspect, kws in ASPECT_KEYWORDS.items():
                if any(kw in low for kw in kws):
                    records.append(
                        {"review_idx": idx, "aspect": aspect,
                         "polarity": _sentence_polarity(sentence)}
                    )
    return pd.DataFrame(records)


def rank_reviewers(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Ranqueia usuários com opiniões relevantes para entrevista.

    Score combina: utilidade percebida pela comunidade (votos úteis),
    produtividade (nº de reviews), profundidade (tamanho médio do texto)
    e criticidade equilibrada (variância de nota — quem só dá 5★ ou só 1★
    informa menos do que quem discrimina qualidade).
    """
    g = df.groupby(["User_id", "profileName"]).agg(
        n_reviews=("review/text", "size"),
        avg_helpful_ratio=("helpful_ratio", "mean"),
        total_helpful_votes=("helpful_votes", "sum"),
        avg_len=("review_len", "mean"),
        score_std=("review/score", "std"),
        avg_score=("review/score", "mean"),
    ).reset_index()
    g = g[g["n_reviews"] >= 2].copy()
    g["score_std"] = g["score_std"].fillna(0)

    def norm(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng else s * 0

    g["relevance_score"] = (
        0.35 * norm(np.log1p(g["total_helpful_votes"]))
        + 0.25 * norm(g["avg_helpful_ratio"].fillna(0))
        + 0.20 * norm(np.log1p(g["avg_len"]))
        + 0.10 * norm(np.log1p(g["n_reviews"]))
        + 0.10 * norm(g["score_std"])
    ).round(3)
    return g.sort_values("relevance_score", ascending=False).head(top_k)
