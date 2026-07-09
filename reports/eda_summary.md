# EDA (dados reais, amostra aleatória de 378,455 de 3M reviews) — 90,609 livros, 54,601 autores
- 80% das reviews têm 4-5★ (nota média satura → o texto é o sinal).
- 31% dos livros concentram 80% das reviews → esforço manual mal alocado na cauda longa.
- Reviews 3★ (32,182): o modelo separa 38% como negativas e 42% como positivas — informação invisível na nota.
- Aspectos criticados variam por gênero → ação editorial direcionada (ex.: ritmo em ficção; clareza/atualização em técnicos).
- 10% dos usuários escrevem 45% das reviews; variante do dataset sem coluna helpfulness → ranking usa profundidade, produtividade e discriminação de nota.

## Tópicos dominantes (NMF)

- Tópico 1: book, recommend, reading, information, excellent, written, highly, author
- Tópico 2: story, life, novel, world, characters, time, man, people
- Tópico 3: read, ve, time, book, times, years, best, loved
- Tópico 4: great, story, condition, loved, classic, price, fun, gift
- Tópico 5: good, like, really, just, think, don, didn, know
- Tópico 6: books, series, love, best, reading, favorite, characters, ve

Modelo de sentimento:
              precision    recall  f1-score   support

         neg       0.58      0.84      0.69      2364
         pos       0.97      0.91      0.94     15907

    accuracy                           0.90     18271
   macro avg       0.78      0.87      0.81     18271
weighted avg       0.92      0.90      0.91     18271
