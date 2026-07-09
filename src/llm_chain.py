"""Camada LLM com LangChain: modelos plugáveis + observabilidade via callback.

Provedores (variável LLM_PROVIDER):
- mock    -> ExtractiveMockLLM (offline, custo zero — demo sem API key)
- openai  -> ChatOpenAI       (pip install langchain-openai)
- bedrock -> ChatBedrock      (pip install langchain-aws)

Por que LangChain aqui:
- o MESMO chain (prompt | llm | parser) roda com qualquer provedor;
- observabilidade idiomática: um BaseCallbackHandler loga tokens, custo
  estimado e latência de toda chamada em logs/llm_usage.jsonl (FinOps);
- em produção, plugar LangSmith para traces é uma variável de ambiente
  (LANGCHAIN_TRACING_V2=true), sem tocar no código.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.llms import LLM

LOG_PATH = Path("logs/llm_usage.jsonl")

# ------------------------------------------------------- segurança de input
# Reviews são dado NÃO CONFIÁVEL injetado em prompts. Duas defesas em camadas:
# 1. sanitize_untrusted(): remove padrões de prompt-injection conhecidos;
# 2. os prompts envolvem o material em delimitadores e instruem o modelo a
#    tratar tudo dentro deles como DADOS, nunca como instruções.
_INJECTION_RE = re.compile(
    r"(?i)(ignore\s+(all|any|the|previous|prior|above)[\s\S]{0,40}?"
    r"(instruction|prompt|rule)s?|disregard\s+(the\s+)?(system|previous|above)|"
    r"you\s+are\s+now\s+|act\s+as\s+(if|a|an)\s|new\s+instructions?\s*:|"
    r"system\s+prompt|<\s*/?\s*(system|assistant|instruction)\s*>)"
)


def sanitize_untrusted(text: str) -> str:
    """Neutraliza tentativas de injeção vindas do conteúdo das reviews."""
    return _INJECTION_RE.sub("[conteúdo removido por segurança]", str(text))

# US$ por 1M tokens (entrada, saída)
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "anthropic.claude-3-5-haiku-20241022-v1:0": (0.80, 4.00),
    "mock": (0.0, 0.0),
}


# ------------------------------------------------------------------ callback
class CostLoggingCallback(BaseCallbackHandler):
    """Loga cada chamada LLM (tokens estimados, custo, latência) em JSONL.

    Em produção o mesmo papel é do LangSmith/CloudWatch; manter o log local
    garante a trilha de FinOps mesmo na POC offline.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._t0: float = 0.0
        self._prompt_chars: int = 0

    def on_llm_start(self, serialized: dict, prompts: list[str], **kw: Any) -> None:
        self._t0 = time.time()
        self._prompt_chars = sum(len(p) for p in prompts)

    def on_llm_end(self, response: Any, **kw: Any) -> None:
        out_chars = sum(len(g.text) for gens in response.generations for g in gens)
        tokens_in = max(1, self._prompt_chars // 4)   # ~4 chars/token
        tokens_out = max(1, out_chars // 4)
        price_in, price_out = PRICING.get(self.model_name, (0.0, 0.0))
        LOG_PATH.parent.mkdir(exist_ok=True)
        with LOG_PATH.open("a") as f:
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "model": self.model_name,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": round(tokens_in / 1e6 * price_in
                                  + tokens_out / 1e6 * price_out, 6),
                "latency_s": round(time.time() - self._t0, 2),
            }) + "\n")


def usage_report() -> dict:
    """Agrega o log de uso — base do painel de FinOps do app."""
    if not LOG_PATH.exists():
        return {"calls": 0, "tokens_in": 0, "tokens_out": 0,
                "cost_usd": 0.0, "avg_latency_s": 0.0}
    rows = [json.loads(l) for l in LOG_PATH.read_text().splitlines() if l]
    return {
        "calls": len(rows),
        "tokens_in": sum(r["tokens_in"] for r in rows),
        "tokens_out": sum(r["tokens_out"] for r in rows),
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
        "avg_latency_s": round(float(np.mean([r["latency_s"] for r in rows])), 2),
    }


# ------------------------------------------------------------------ mock LLM
class ExtractiveMockLLM(LLM):
    """LLM fake para demo offline: sumarização extrativa por centralidade
    TF-IDF, com deduplicação e separação elogios/críticas.

    Implementa a interface `LLM` do LangChain — o restante do código não
    sabe (nem precisa saber) que não há um modelo real por trás.
    """

    @property
    def _llm_type(self) -> str:
        return "extractive-mock"

    def _call(self, prompt: str, stop: list[str] | None = None, **kw: Any) -> str:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        neg_re = re.compile(
            r"\b(not|poorly|bad|terrible|awful|waste|disappoint\w*|boring|weak|"
            r"confusing|shallow|slow|broken|wrong|errors?|flat|predictable|"
            r"repetitive|rushed|letdown|outdated|clumsy|superficial|"
            r"too (?:much|many|expensive)|hard to|almost no|struggled|gave up|"
            r"below my|expected more|no real|generic advice|magazine article)\b", re.I)

        material = prompt.split("### REVIEWS")[-1]
        seen, sentences = set(), []
        for s in re.split(r"(?<=[.!?])\s+|\n+", material):
            s = s.strip().lstrip("- ").strip()
            # remove cabeçalhos "[5★] Título:" em QUALQUER posição da frase
            s = re.sub(r"\[\d★\]\s*[^:]{0,80}:\s*", " ", s).strip()
            if 30 < len(s) < 320 and s.lower() not in seen:
                seen.add(s.lower())
                sentences.append(s)
        if len(sentences) < 3:
            return material[:800]

        vec = TfidfVectorizer(stop_words="english")
        x = vec.fit_transform(sentences)
        centrality = np.asarray((x @ x.T).mean(axis=1)).ravel()
        sim = cosine_similarity(x)

        def diverse_top(indices: list[int], k: int) -> list[int]:
            chosen: list[int] = []
            for i in sorted(indices, key=lambda i: -centrality[i]):
                if all(sim[i, j] < 0.6 for j in chosen):
                    chosen.append(i)
                if len(chosen) == k:
                    break
            return chosen

        neg_idx = [i for i, s in enumerate(sentences) if neg_re.search(s)]
        pos_idx = [i for i in range(len(sentences)) if i not in set(neg_idx)]
        pos = [sentences[i] for i in diverse_top(pos_idx, 4)]
        neg = [sentences[i] for i in diverse_top(neg_idx, 4)]

        parts = ["[MODO DEMO — LLM extrativo offline; em produção, Claude via "
                 "Bedrock com o mesmo chain]", "", "**Pontos fortes recorrentes:**"]
        parts += [f"- {s}" for s in pos] or ["- (nenhum destaque)"]
        parts += ["", "**Críticas recorrentes:**"]
        parts += [f"- {s}" for s in neg] or ["- (nenhuma crítica relevante)"]
        return "\n".join(parts)


# ------------------------------------------------------------------ factory
def get_llm(provider: str | None = None):
    """Devolve (llm, callbacks) prontos para compor chains LCEL."""
    provider = (provider or os.getenv("LLM_PROVIDER", "mock")).lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0.2), [CostLoggingCallback(model)]

    if provider == "bedrock":
        from langchain_aws import ChatBedrock
        model = os.getenv("BEDROCK_MODEL_ID",
                          "anthropic.claude-3-5-haiku-20241022-v1:0")
        return (ChatBedrock(model_id=model, model_kwargs={"temperature": 0.2},
                            region_name=os.getenv("AWS_REGION", "us-east-1")),
                [CostLoggingCallback(model)])

    return ExtractiveMockLLM(), [CostLoggingCallback("mock")]


def get_cascade_llm(cheap: str | None = None, premium: str | None = None):
    """Cascata custo→qualidade (padrão validado em produção no setor público):
    o modelo barato atende por padrão e o premium assume em caso de falha.

    Implementação idiomática LangChain: `with_fallbacks`. Em produção, o gate
    também pode ser por confiança (o barato autoavalia a resposta e escala ao
    premium quando a confiança cai — ex.: Llama -> Claude Sonnet), adicionando
    um passo de verificação antes do fallback.
    """
    llm_cheap, cb1 = get_llm(cheap)
    llm_premium, cb2 = get_llm(premium or cheap)
    return llm_cheap.with_fallbacks([llm_premium]), cb1 + cb2
