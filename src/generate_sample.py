"""Gera uma amostra sintética com o MESMO schema do dataset Kaggle
"Amazon Books Reviews" (Books_rating.csv + books_data.csv).

Uso: permite desenvolver e demonstrar a POC sem os ~2.7GB do dataset real.
Para usar os dados reais, basta colocar os CSVs do Kaggle em data/raw/ —
todo o pipeline funciona sem alteração.

Execução:
    python -m src.generate_sample --n-reviews 6000 --out data/sample
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

GENRES = {
    "Fiction": ["plot", "characters", "pacing", "writing"],
    "Mystery & Thriller": ["plot", "pacing", "characters", "ending"],
    "Romance": ["characters", "writing", "pacing", "ending"],
    "Science Fiction": ["plot", "worldbuilding", "characters", "writing"],
    "Business & Economics": ["clarity", "examples", "practicality", "depth"],
    "Self-Help": ["practicality", "clarity", "repetition", "depth"],
    "History": ["depth", "writing", "accuracy", "pacing"],
    "Computers & Technology": ["clarity", "examples", "accuracy", "edition"],
}

ASPECT_SENTENCES = {
    "plot": {
        "pos": ["The plot is gripping and full of clever twists.",
                "A masterfully constructed story that kept me hooked until the last page.",
                "The storyline is original and beautifully executed."],
        "neg": ["The plot is predictable and full of holes.",
                "The story drags and goes nowhere for entire chapters.",
                "A confusing storyline that never really comes together."],
    },
    "characters": {
        "pos": ["The characters are deep, believable and easy to care about.",
                "Wonderful character development throughout the book.",
                "The protagonist is one of the most memorable I have read in years."],
        "neg": ["The characters feel flat and one-dimensional.",
                "I could not connect with any of the characters.",
                "The villain's motivations make absolutely no sense."],
    },
    "pacing": {
        "pos": ["The pacing is perfect, I finished it in two sittings.",
                "Fast paced and impossible to put down."],
        "neg": ["The pacing is painfully slow, especially in the middle.",
                "The first half drags so much I almost gave up."],
    },
    "writing": {
        "pos": ["The prose is elegant and a pleasure to read.",
                "Beautifully written, every sentence feels crafted."],
        "neg": ["The writing is clumsy and repetitive.",
                "Poor editing, full of typos and awkward sentences."],
    },
    "ending": {
        "pos": ["The ending is satisfying and ties everything together.",
                "A brilliant twist ending I never saw coming."],
        "neg": ["The ending feels rushed and unearned.",
                "After such a buildup, the ending was a huge letdown."],
    },
    "worldbuilding": {
        "pos": ["The worldbuilding is rich and utterly immersive.",
                "An imaginative universe with consistent internal logic."],
        "neg": ["The worldbuilding is shallow and derivative.",
                "The universe raises questions the author never answers."],
    },
    "clarity": {
        "pos": ["Complex ideas are explained with remarkable clarity.",
                "Very well structured, each chapter builds on the previous one."],
        "neg": ["The explanations are confusing and poorly organized.",
                "The author assumes too much prior knowledge."],
    },
    "examples": {
        "pos": ["Great real-world examples that make the concepts stick.",
                "The case studies are relevant and up to date."],
        "neg": ["The examples are outdated and hard to relate to.",
                "Too much theory, almost no practical examples."],
    },
    "practicality": {
        "pos": ["Full of actionable advice I applied immediately.",
                "This book genuinely changed how I work day to day."],
        "neg": ["Nice ideas but nothing you can actually apply.",
                "Generic advice you can find in any blog post."],
    },
    "depth": {
        "pos": ["Impressive depth of research and analysis.",
                "Goes far beyond the surface, a truly thorough treatment."],
        "neg": ["Superficial treatment of an important topic.",
                "It reads like a long magazine article, no real depth."],
    },
    "repetition": {
        "pos": ["Concise and to the point, no filler.",
                "Every chapter adds something new."],
        "neg": ["The same idea is repeated over and over.",
                "Could have been a 20-page essay, extremely repetitive."],
    },
    "accuracy": {
        "pos": ["Meticulously researched and factually solid.",
                "The technical content is accurate and current."],
        "neg": ["Contains several factual errors that undermine trust.",
                "Some technical sections are simply wrong."],
    },
    "edition": {
        "pos": ["This edition is well formatted with clear code samples.",
                "Great print quality and useful diagrams."],
        "neg": ["The code samples in this edition are broken.",
                "Terrible formatting on the kindle edition, tables are unreadable."],
    },
    "price": {
        "pos": ["Excellent value for the price.",
                "Worth every penny."],
        "neg": ["Way too expensive for what it delivers.",
                "Not worth the cover price, borrow it instead."],
    },
}

OPENERS = {
    5: ["Absolutely loved this book.", "One of the best books I have read this year.",
        "A must-read.", "Five stars without hesitation."],
    4: ["Really enjoyed this one.", "A very good read overall.", "Solid book with minor flaws."],
    3: ["A mixed experience for me.", "Decent, but I expected more.", "Some great parts, some weak ones."],
    2: ["Quite disappointing.", "I struggled to finish this.", "Below my expectations."],
    1: ["A complete waste of time.", "I rarely give one star, but this earned it.",
        "Do not waste your money."],
}

SUMMARIES = {
    5: ["Outstanding", "A masterpiece", "Loved it", "Highly recommended"],
    4: ["Very good read", "Enjoyable", "Recommended with minor caveats"],
    3: ["Mixed feelings", "Average", "Good ideas, uneven execution"],
    2: ["Disappointing", "Expected more", "Hard to finish"],
    1: ["Terrible", "Avoid", "Waste of money"],
}

FIRST = ["John", "Mary", "Susan", "David", "Karen", "Michael", "Linda", "Robert", "Patricia",
         "James", "Jennifer", "William", "Elizabeth", "Richard", "Barbara", "Thomas", "Nancy",
         "Daniel", "Laura", "Paul", "Amy", "Mark", "Julie", "Steven", "Anna"]
LAST = ["Smith", "Johnson", "Brown", "Miller", "Davis", "Wilson", "Moore", "Taylor",
        "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Clark"]

AUTHOR_NAMES = [
    "Alice Warren", "Brian Cole", "Clara Whitfield", "Daniel Reyes", "Elena Marsh",
    "Frank Delgado", "Grace Holloway", "Henry Aldridge", "Isabel Navarro", "Jonas Beck",
    "Katherine Pryce", "Liam Foster", "Marina Vidal", "Nathan Brooks", "Olivia Sterling",
    "Peter Lang", "Quinn Harper", "Rachel Osei", "Samuel Trent", "Tessa Morrow",
]

PUBLISHERS = ["Northlight Press", "Vellum House", "Bluebird Books", "Crown & Quill", "Meridian"]


def _make_books(rng: random.Random, n_books: int) -> pd.DataFrame:
    """Cria o catálogo (books_data.csv). Autores têm 'qualidade latente' distinta
    para que hipóteses sobre performance de autor sejam verificáveis na amostra."""
    rows = []
    genres = list(GENRES)
    for i in range(n_books):
        author = AUTHOR_NAMES[i % len(AUTHOR_NAMES)]
        genre = genres[hash(author) % len(genres)] if rng.random() < 0.7 else rng.choice(genres)
        title = (f"{rng.choice(['The', 'A', 'Beyond the', 'Secrets of the', 'Last'])} "
                 f"{rng.choice(['Silent', 'Hidden', 'Golden', 'Broken', 'Infinite', 'Practical', 'Digital'])} "
                 f"{rng.choice(['Garden', 'Empire', 'Algorithm', 'Horizon', 'Letter', 'Manager', 'City', 'Code'])}"
                 f" (Book {i + 1})")
        rows.append({
            "Title": title,
            "description": f"A {genre.lower()} book by {author}.",
            "authors": f"['{author}']",
            "image": "", "previewLink": "", "infoLink": "",
            "publisher": rng.choice(PUBLISHERS),
            "publishedDate": str(rng.randint(1995, 2012)),
            "categories": f"['{genre}']",
            "ratingsCount": 0,  # preenchido depois
        })
    return pd.DataFrame(rows)


def _review_text(rng: random.Random, genre: str, score: int) -> str:
    aspects = GENRES[genre] + (["price"] if rng.random() < 0.25 else [])
    n_aspects = rng.randint(2, min(4, len(aspects)))
    chosen = rng.sample(aspects, n_aspects)
    p_pos = {1: 0.08, 2: 0.25, 3: 0.5, 4: 0.8, 5: 0.95}[score]
    sents = [rng.choice(OPENERS[score])]
    for a in chosen:
        pol = "pos" if rng.random() < p_pos else "neg"
        sents.append(rng.choice(ASPECT_SENTENCES[a][pol]))
    return " ".join(sents)


def generate(n_reviews: int = 6000, n_books: int = 60, n_users: int = 800,
             out_dir: str = "data/sample") -> None:
    rng = random.Random(SEED)
    np.random.seed(SEED)

    books = _make_books(rng, n_books)

    # qualidade latente por autor -> distribuições de nota distintas
    authors = books["authors"].str.strip("[]'").unique()
    quality = {a: np.clip(rng.gauss(3.9, 0.55), 2.2, 4.9) for a in authors}

    users = []
    for u in range(n_users):
        users.append({
            "User_id": f"A{u:06d}XYZ",
            "profileName": f"{rng.choice(FIRST)} {rng.choice(LAST)} \"{rng.choice(['reader', 'bookworm', 'critic', 'reviewer'])}\"",
            # personas: críticos escrevem mais e recebem mais votos úteis
            "persona": rng.choices(["critic", "fan", "casual"], weights=[0.1, 0.3, 0.6])[0],
        })

    # popularidade long-tail (Pareto): poucos livros concentram reviews
    popularity = np.random.pareto(1.2, size=len(books)) + 1
    popularity = popularity / popularity.sum()

    rows = []
    for i in range(n_reviews):
        b = int(np.random.choice(len(books), p=popularity))
        book = books.iloc[b]
        author = book["authors"].strip("[]'")
        genre = book["categories"].strip("[]'")
        u = rng.choice(users)

        mu = quality[author] + {"critic": -0.5, "fan": 0.4, "casual": 0.0}[u["persona"]]
        score = int(np.clip(round(rng.gauss(mu, 0.9)), 1, 5))
        text = _review_text(rng, genre, score)
        if u["persona"] == "critic":  # críticos escrevem reviews mais longas
            text += " " + _review_text(rng, genre, score)

        base_votes = {"critic": 18, "fan": 6, "casual": 2}[u["persona"]]
        total_votes = max(0, int(rng.gauss(base_votes, base_votes / 2)))
        helpful = int(total_votes * rng.uniform(0.55, 0.98)) if total_votes else 0

        rows.append({
            "Id": f"{b:010d}",
            "Title": book["Title"],
            "Price": round(rng.uniform(5, 45), 2) if rng.random() < 0.4 else np.nan,
            "User_id": u["User_id"],
            "profileName": u["profileName"],
            "review/helpfulness": f"{helpful}/{total_votes}",
            "review/score": float(score),
            "review/time": rng.randint(852076800, 1362355200),  # 1997–2013 (epoch)
            "review/summary": rng.choice(SUMMARIES[score]),
            "review/text": text,
        })

    ratings = pd.DataFrame(rows)
    counts = ratings.groupby("Title").size()
    books["ratingsCount"] = books["Title"].map(counts).fillna(0).astype(int)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(out / "Books_rating.csv", index=False)
    books.to_csv(out / "books_data.csv", index=False)
    print(f"OK: {len(ratings)} reviews, {len(books)} livros -> {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reviews", type=int, default=6000)
    ap.add_argument("--n-books", type=int, default=60)
    ap.add_argument("--out", default="data/sample")
    args = ap.parse_args()
    generate(args.n_reviews, args.n_books, out_dir=args.out)
