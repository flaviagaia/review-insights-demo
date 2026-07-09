"""Guardrails: travas de entrada e saída para o Q&A generativo.

Três riscos, três gates — todos ANTES ou DEPOIS do LLM, nunca dependendo
só do prompt (defesa em profundidade):

1. ENTRADA / conteúdo   — pergunta com conteúdo impróprio (ofensas, pedidos
   de dados pessoais, instruções de injeção) é recusada sem chamar o LLM.
2. ENTRADA / escopo     — pergunta sem aderência ao domínio (similaridade
   máxima do retriever abaixo do piso) é recusada: zero tokens gastos e
   zero chance de resposta inventada.
3. SAÍDA / alucinação   — citações [Rn] da resposta são validadas contra as
   fontes recuperadas; citação inexistente derruba a resposta inteira
   (é mais seguro recusar do que entregar evidência fabricada).

Em produção AWS, o gate de conteúdo é reforçado pelo Amazon Bedrock
Guardrails (filtros gerenciados de toxicidade/PII) — estas funções
continuam valendo como segunda camada, portáveis para qualquer provedor.
"""
from __future__ import annotations

import re

from .llm_chain import _INJECTION_RE

# Mensagens padronizadas (o app exibe como estão)
REFUSAL_CONTENT = ("Não posso ajudar com esse tipo de conteúdo. "
                   "Reformule a pergunta sobre os livros e as reviews da base.")
REFUSAL_SCOPE = ("Fora do escopo: esta ferramenta responde apenas perguntas "
                 "sobre os livros e reviews da base analisada.")
REFUSAL_UNGROUNDED = ("Não há evidência suficiente na base para responder "
                      "com segurança (resposta descartada por citar fontes "
                      "inexistentes).")

# Conteúdo impróprio (PT/EN): ofensas, sexual explícito, violência,
# tentativa de extrair dados pessoais dos leitores.
_IMPROPER_RE = re.compile(
    r"(?i)\b(porn\w*|sexo\s+expl[ií]cito|nude\w*|nsfw|"
    r"como\s+(?:fazer|construir)\s+(?:uma?\s+)?(?:bomba|arma|veneno)|"
    r"how\s+to\s+(?:make|build)\s+a?\s*(?:bomb|weapon|poison)|"
    r"matar|kill\s+(?:him|her|them|someone)|suicid\w*|"
    r"cpf|cart[ãa]o\s+de\s+cr[eé]dito|credit\s+card|senha|password|"
    r"endere[çc]o\s+(?:do|da|de)\s|telefone\s+(?:do|da|de)\s|"
    r"e-?mail\s+(?:do|da|de)\s|nome\s+real\s+(?:do|da|de)\s)"
)

_CITATION_RE = re.compile(r"\[R(\d+)\]")

# Âncoras de domínio (PT/EN): num corpus de livros, quase qualquer substantivo
# aparece em alguma review ("capital", "chocolate", "França"...), então
# similaridade sozinha não detecta pergunta fora do escopo. Exigimos que a
# pergunta mencione o domínio (livros, autores, leitores, opiniões, aspectos).
_DOMAIN_ANCHOR_RE = re.compile(
    r"(?i)\b(livros?|autor(?:es|a|as)?|leitor(?:es|a|as)?|reviews?|resenha\w*|"
    r"avalia\w+|leitura|obras?|edi[çc][ãa]o|edi[çc][õo]es|kindle|escrita|ritmo|"
    r"personag\w+|enredo|final|cap[ií]tulo\w*|g[êe]nero\w*|fic[çc][ãa]o|"
    r"romances?|mist[ée]rios?|neg[óo]cios|autoajuda|elogi\w+|cr[ií]tica\w*|"
    r"opini[ãa]o|opini[õo]es|sentimento\w*|aspecto\w*|acham?\b|"
    r"books?|authors?|readers?|reading|editions?|writing|pacing|characters?|"
    r"plots?|endings?|chapters?|genres?|fiction|novels?|business|praise\w*|"
    r"criticisms?|complaints?|opinions?|think)\b")


def has_domain_anchor(question: str) -> bool:
    """A pergunta menciona o domínio (livros/reviews/opiniões)?"""
    return bool(_DOMAIN_ANCHOR_RE.search(str(question)))

# Piso de similaridade TF-IDF: abaixo disso o retriever "achou" documentos
# por acaso lexical, não por aderência real ao tema.
MIN_TOP_SIMILARITY = 0.05


def guard_question(question: str) -> tuple[bool, str]:
    """Gate de entrada. Retorna (ok, motivo_da_recusa)."""
    q = str(question).strip()
    if not q or len(q) > 500:
        return False, REFUSAL_SCOPE
    if _INJECTION_RE.search(q):
        # tentativa de injeção na PERGUNTA (não só nas reviews): recusa
        return False, REFUSAL_CONTENT
    if _IMPROPER_RE.search(q):
        return False, REFUSAL_CONTENT
    return True, ""


def guard_scope(docs: list, min_similarity: float = MIN_TOP_SIMILARITY) -> bool:
    """Gate de escopo: exige pelo menos 1 fonte com similaridade real."""
    if not docs:
        return False
    top = max(float(d.metadata.get("similarity", 0.0)) for d in docs)
    return top >= min_similarity


def guard_answer(answer: str, docs: list) -> str:
    """Gate de saída: toda citação [Rn] deve existir nas fontes recuperadas.

    Resposta que cita fonte inexistente é evidência fabricada — descartamos
    a resposta inteira em vez de "consertar" (fail-closed).
    """
    valid_ids = {d.metadata.get("id", "") for d in docs}
    cited = {f"R{m}" for m in _CITATION_RE.findall(str(answer))}
    if cited and not cited.issubset(valid_ids):
        return REFUSAL_UNGROUNDED
    return answer
