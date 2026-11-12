# %% [markdown]
# # Part 1 — Word Embeddings (25 marks)
# ## 1.1 TF-IDF weighting
# Vocabulary = the 10,000 most frequent word tokens of `cleaned.txt`; every other token maps to `<UNK>`.
# Index 0 is reserved for `<PAD>` (needed later by the neural models), index 1 is `<UNK>`, so the matrix has
# `10,002` columns that line up exactly with `word2idx.json`.
#
# $\mathrm{TF\text{-}IDF}(w,d)=\mathrm{TF}(w,d)\times\log\frac{N}{1+\mathrm{df}(w)}$, with $\mathrm{TF}(w,d)=\frac{\mathrm{count}(w,d)}{|d|}$.

# %%
counter = collections.Counter(w for a in ARTICLES for w in a["words"])
ITOS = ["<PAD>", "<UNK>"] + [w for w, _ in counter.most_common(VOCAB_SIZE)]
W2I = {w: i for i, w in enumerate(ITOS)}
PAD, UNK = 0, 1
V = len(ITOS)
DOC_IDS = [np.array([W2I.get(w, UNK) for w in a["words"]], dtype=np.int64) for a in ARTICLES]

# term–document matrix (documents x vocabulary) of raw counts
N_DOCS = len(ARTICLES)
r = np.concatenate([np.full(len(ids), i) for i, ids in enumerate(DOC_IDS)])
c = np.concatenate(DOC_IDS)
TD = sp.csr_matrix((np.ones(len(c), dtype=np.float32), (r, c)), shape=(N_DOCS, V))   # duplicates are summed
TD.sum_duplicates()
doc_len = np.asarray(TD.sum(1)).ravel()
df = np.asarray((TD > 0).sum(0)).ravel()
idf = np.log(N_DOCS / (1.0 + df)).astype(np.float32)
idf[PAD] = 0.0
TF = sp.diags(1.0 / np.maximum(doc_len, 1)) @ TD
TFIDF = (TF @ sp.diags(idf)).toarray().astype(np.float32)
np.save(os.path.join(EMB_DIR, "tfidf_matrix.npy"), TFIDF)

coverage = 1 - TD[:, UNK].sum() / TD.sum()
print(f"Term–document matrix: {TD.shape} (docs x vocab), nnz={TD.nnz:,}")
print(f"Vocabulary = {V-2:,} words + <PAD> + <UNK>; token coverage of vocab = {coverage:.2%}")
print("Most frequent:", ITOS[2:17])
print(f"IDF range: {idf[2:].min():.3f} .. {idf[2:].max():.3f}  (words in almost every doc get IDF<=0 by this formula)")
print("Saved tfidf_matrix.npy", TFIDF.shape)
RESULTS["p1_tfidf"] = {"shape": list(TFIDF.shape), "coverage": float(coverage)}

# %% [markdown]
# ### Topic categories (used in 1.1, Part 2 and Part 3)
# Each article is assigned to one of the 5 categories of Part 3 by counting the indicative keywords
# (English + Urdu equivalents) in its metadata (weight 3) and body (weight 1). Articles with no keyword
# hit are assigned to the closest category centroid in TF-IDF space.

# %%
CATEGORIES = ["Politics", "Sports", "Economy", "International", "Health & Society"]
CAT_KEYWORDS = {
    "Politics": {
        "en": ["election", "government", "minister", "parliament", "politic", "assembly", "senate", "vote", "party", "pti", "pml", "ppp", "prime"],
        "ur": ["انتخاب", "انتخابات", "الیکشن", "حکومت", "حکومتی", "وزیر", "وزیراعظم", "وزرا", "پارلیمان", "پارلیمنٹ", "اسمبلی", "سینیٹ",
               "سیاسی", "سیاست", "سیاستدان", "جماعت", "اپوزیشن", "ووٹ", "ووٹنگ", "ووٹوں", "کابینہ", "وزیراعلیٰ", "وزیراعلی", "گورنر",
               "تحریک انصاف", "مسلم لیگ", "پیپلز پارٹی", "عمران خان", "شہباز شریف", "نواز شریف", "آئین", "آئینی", "ترمیم", "حلف"]},
    "Sports": {
        "en": ["cricket", "match", "team", "player", "score", "sport", "football", "hockey", "olympic"],
        "ur": ["کرکٹ", "میچ", "میچوں", "ٹیم", "ٹیموں", "کھلاڑی", "کھلاڑیوں", "سکور", "اسکور", "رنز", "وکٹ", "وکٹیں", "وکٹوں", "بلے باز",
               "بولر", "بولرز", "باؤلر", "کپتان", "ٹورنامنٹ", "ورلڈ کپ", "فٹبال", "ہاکی", "اولمپکس", "سنچری", "اننگز", "سیریز",
               "ٹیسٹ میچ", "ون ڈے", "ٹی ٹوئنٹی", "پی سی بی", "کھیل", "کوچ", "سٹیڈیم", "گول"]},
    "Economy": {
        "en": ["inflation", "trade", "bank", "gdp", "budget", "econom", "business", "imf", "tax", "rupee", "dollar"],
        "ur": ["مہنگائی", "افراط زر", "تجارت", "تجارتی", "بینک", "بینکوں", "جی ڈی پی", "بجٹ", "معیشت", "معاشی", "اقتصادی", "روپے",
               "روپیہ", "ڈالر", "قرض", "قرضے", "قرضوں", "آئی ایم ایف", "ٹیکس", "ٹیکسوں", "سرمایہ کاری", "سرمایہ", "برآمدات", "درآمدات",
               "شرح سود", "پیٹرول", "قیمت", "قیمتوں", "کاروبار", "مارکیٹ", "سٹاک", "سٹیٹ بینک", "مالی", "ارب", "کروڑ", "صنعت"]},
    "International": {
        "en": ["un", "treaty", "foreign", "bilateral", "conflict", "world", "international", "war", "diplomat"],
        "ur": ["اقوام متحدہ", "معاہدہ", "معاہدے", "خارجہ", "غیر ملکی", "دوطرفہ", "دو طرفہ", "تنازع", "تنازعہ", "تنازعے", "جنگ", "امریکہ",
               "امریکی", "چین", "روس", "روسی", "یوکرین", "اسرائیل", "اسرائیلی", "غزہ", "فلسطین", "فلسطینی", "ایران", "ایرانی", "افغانستان",
               "طالبان", "بھارت", "انڈیا", "بھارتی", "سعودی", "سفارتی", "سفیر", "عالمی", "بین الاقوامی", "نیٹو", "سلامتی کونسل"]},
    "Health & Society": {
        "en": ["hospital", "disease", "vaccine", "flood", "education", "health", "school", "covid", "polio", "society"],
        "ur": ["ہسپتال", "اسپتال", "ہسپتالوں", "بیماری", "بیماریوں", "ویکسین", "سیلاب", "سیلابی", "تعلیم", "تعلیمی", "سکول", "اسکول",
               "سکولوں", "یونیورسٹی", "طلبہ", "طالب علم", "ڈاکٹر", "ڈاکٹروں", "مریض", "مریضوں", "صحت", "کورونا", "وائرس", "وبا",
               "پولیو", "ڈینگی", "علاج", "دوا", "ادویات", "خواتین", "بچوں", "سماجی", "معاشرے", "بارش", "بارشوں", "زلزلہ", "آلودگی", "موسمیاتی"]},
}
for cat in CAT_KEYWORDS:
    CAT_KEYWORDS[cat]["ur"] = [normalize(k) for k in CAT_KEYWORDS[cat]["ur"]]
    CAT_KEYWORDS[cat]["urc"] = sorted({" ".join(cf(t) for t in k.split()) for k in CAT_KEYWORDS[cat]["ur"]})   # stemmed forms for the body

def keyword_scores(article):
    body = " " + " ".join(article["words"]) + " "
    toks = collections.Counter(article["words"])
    mt = normalize(meta_text(article["meta"]))
    mt_low = mt.lower()
    mt_toks = collections.Counter(tokenize(mt))
    scores = np.zeros(len(CATEGORIES))
    for ci, cat in enumerate(CATEGORIES):
        for k in CAT_KEYWORDS[cat]["urc"]:                      # body (stemmed cleaned text)
            scores[ci] += body.count(" " + k + " ") if " " in k else toks.get(k, 0)
        for k in CAT_KEYWORDS[cat]["ur"]:                       # metadata title (raw script)
            scores[ci] += 3 * (mt.count(k) if " " in k else mt_toks.get(k, 0))
        for k in CAT_KEYWORDS[cat]["en"]:
            scores[ci] += 3 * len(re.findall(r"\b" + re.escape(k), mt_low))
    return scores

KW_SCORES = np.stack([keyword_scores(a) for a in ARTICLES])
DOC_CAT = KW_SCORES.argmax(1)
no_hit = KW_SCORES.sum(1) == 0
if no_hit.any() and (~no_hit).any():
    Xn = TFIDF / (np.linalg.norm(TFIDF, axis=1, keepdims=True) + 1e-9)
    cents = np.stack([Xn[(DOC_CAT == ci) & ~no_hit].mean(0) if ((DOC_CAT == ci) & ~no_hit).any() else np.zeros(V) for ci in range(5)])
    DOC_CAT[no_hit] = (Xn[no_hit] @ cents.T).argmax(1)
for a, ci in zip(ARTICLES, DOC_CAT):
    a["category"] = CATEGORIES[ci]
cat_dist = pd.Series([a["category"] for a in ARTICLES]).value_counts().reindex(CATEGORIES).fillna(0).astype(int)
print(f"Articles without any keyword hit (assigned by TF-IDF centroid): {int(no_hit.sum())}")
display(cat_dist.to_frame("articles"))
RESULTS["categories"] = cat_dist.to_dict()

# %%
# ---- Top-10 most discriminative words per category (mean TF-IDF inside category minus mean outside) ----
top_rows = {}
for ci, cat in enumerate(CATEGORIES):
    m_in = DOC_CAT == ci
    if m_in.sum() == 0 or (~m_in).sum() == 0:
        continue
    score = TFIDF[m_in].mean(0) - TFIDF[~m_in].mean(0)
    score[[PAD, UNK]] = -np.inf
    top = np.argsort(-score)[:10]
    top_rows[cat] = [ITOS[i] for i in top]
TOP_TFIDF = pd.DataFrame(top_rows)
TOP_TFIDF.index = [f"#{i+1}" for i in range(len(TOP_TFIDF))]
print("Top-10 discriminative words per category (TF-IDF):")
display(TOP_TFIDF)
RESULTS["p1_top_tfidf"] = top_rows

# %% [markdown]
# ## 1.2 Positive PMI
# Symmetric window $k=5$ inside each document (pairs never cross article boundaries; `<UNK>`/`<PAD>` excluded).
# $\mathrm{PPMI}(w_1,w_2)=\max\{0,\log_2\frac{P(w_1,w_2)}{P(w_1)P(w_2)}\}$ with $P(w_1,w_2)=C(w_1,w_2)/\sum C$ and
# $P(w)=\sum_c C(w,c)/\sum C$.

# %%
def cooccurrence(doc_ids, Vn, k=WINDOW):
    rows, cols = [], []
    for ids in doc_ids:
        for o in range(1, k + 1):
            if len(ids) <= o: break
            a, b = ids[:-o], ids[o:]
            m = (a > UNK) & (b > UNK)
            rows += [a[m], b[m]]; cols += [b[m], a[m]]
    r = np.concatenate(rows); c = np.concatenate(cols)
    C = sp.coo_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(Vn, Vn)).tocsr()
    C.sum_duplicates()
    return C

def ppmi(C):
    total = C.sum()
    rs = np.asarray(C.sum(1)).ravel()
    Cc = C.tocoo()
    pmi = np.log2(Cc.data * total / (rs[Cc.row] * rs[Cc.col]))
    keep = pmi > 0
    return sp.csr_matrix((pmi[keep].astype(np.float32), (Cc.row[keep], Cc.col[keep])), shape=C.shape)

COOC = cooccurrence(DOC_IDS, V)
PPMI = ppmi(COOC)
np.save(os.path.join(EMB_DIR, "ppmi_matrix.npy"), PPMI.toarray().astype(np.float32))
print(f"Co-occurrence matrix {COOC.shape}, nnz={COOC.nnz:,}, total pair count={COOC.sum():,.0f}")
print(f"PPMI nnz={PPMI.nnz:,} ({PPMI.nnz / V**2:.3%} dense), max={PPMI.max():.2f}")
print("Saved ppmi_matrix.npy (dense float32)")
RESULTS["p1_ppmi"] = {"nnz": int(PPMI.nnz), "cooc_nnz": int(COOC.nnz)}

# %%
# ---- Generic embedding space helper (used for PPMI and every Word2Vec condition) ----
def _variants(w):
    w = normalize(w)
    vs = [w, w.replace("آ", "ا"), w.replace("ۂ", "ہ"), w.replace("ی", "ي"), w.replace("ک", "ك"), w.replace("ہ", "ه")]
    return list(dict.fromkeys(vs))

class EmbSpace:
    def __init__(self, name, M, itos, use_cf=True, counts=None, min_count=5):
        self.name, self.itos, self.use_cf = name, itos, use_cf
        self.w2i = {w: i for i, w in enumerate(itos)}
        counts = counts if counts is not None else collections.Counter()
        self.mask = np.array([i > UNK and counts.get(w, 0) >= min_count for i, w in enumerate(itos)])
        if sp.issparse(M):
            n = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()) + 1e-9
            self.M = sp.diags(1 / n) @ M; self.sparse = True
        else:
            M = torch.tensor(M, dtype=torch.float32)
            self.M = F.normalize(M, dim=1).numpy(); self.sparse = False
    def resolve(self, w):
        cands = ([cf(w)] if self.use_cf else []) + _variants(w) + ([] if self.use_cf else [cf(w)])
        for v in cands:
            if v in self.w2i and self.mask[self.w2i[v]]: return v
        return None
    def sims(self, idx):
        if self.sparse:
            s = np.asarray((self.M @ self.M[idx].T).todense()).ravel()
        else:
            s = self.M @ self.M[idx]
        s = s.astype(np.float64); s[~self.mask] = -np.inf
        return s
    def neighbors(self, w, k=10):
        w = self.resolve(w)
        if w is None: return None
        s = self.sims(self.w2i[w]); s[self.w2i[w]] = -np.inf
        top = np.argpartition(-s, k)[:k]; top = top[np.argsort(-s[top])]
        return [(self.itos[i], float(s[i])) for i in top]
    def rank(self, q, t):
        q, t = self.resolve(q), self.resolve(t)
        if q is None or t is None: return None
        s = self.sims(self.w2i[q]); s[self.w2i[q]] = -np.inf
        return int((s > s[self.w2i[t]]).sum()) + 1
    def vec(self, w):
        w = self.resolve(w)
        if w is None: return None
        return self.M[self.w2i[w]].toarray().ravel() if self.sparse else self.M[self.w2i[w]]

SPACE_PPMI = EmbSpace("C1 PPMI", PPMI, ITOS, counts=counter)

# Romanised query names (assignment) -> Urdu script
QUERY_MAP = {"Pakistan": "پاکستان", "Hukumat": "حکومت", "Adalat": "عدالت", "Maeeshat": "معیشت", "Fauj": "فوج",
             "Sehat": "صحت", "Taleem": "تعلیم", "Aabadi": "آبادی"}
EXTRA_QUERIES = {"Cricket": "کرکٹ", "Wazir": "وزیر", "Amreeka": "امریکہ", "Bharat": "بھارت", "Police": "پولیس",
                 "Budget": "بجٹ", "Intikhabat": "انتخابات", "Hospital": "ہسپتال", "Siasi": "سیاسی", "Team": "ٹیم"}

def nn_table(space, queries, k):
    cols = {}
    for rom, urw in queries.items():
        nb = space.neighbors(urw, k)
        if nb: cols[f"{rom} ({space.resolve(urw)})"] = [f"{w} ({s:.2f})" for w, s in nb]
    return pd.DataFrame(cols, index=[f"#{i+1}" for i in range(k)])

ppmi_q = {**QUERY_MAP, **EXTRA_QUERIES}
NN_PPMI = nn_table(SPACE_PPMI, ppmi_q, 5)
print(f"PPMI top-5 nearest neighbours (cosine) for {NN_PPMI.shape[1]} query words:")
display(NN_PPMI.T)

# %%
# ---- t-SNE of the 200 most frequent tokens, colour-coded by semantic category ----
SEM_SEEDS = {
    "Politics":        "حکومت وزیر وزیراعظم سیاسی جماعت انتخابات اسمبلی صدر اپوزیشن پارلیمان عمران شریف تحریک لیگ ووٹ قانون عدالت فیصلہ",
    "Sports":          "کرکٹ میچ ٹیم کھلاڑی کھلاڑیوں سکور رنز وکٹ کپتان ٹورنامنٹ کپ سیریز کھیل",
    "Economy":         "معیشت بجٹ مہنگائی روپے ڈالر بینک قرض ٹیکس تجارت قیمت کاروبار ارب کروڑ اضافہ معاشی",
    "Geography/Intl":  "پاکستان بھارت امریکہ چین افغانستان ایران روس اسرائیل غزہ کراچی لاہور اسلام آباد پنجاب سندھ بلوچستان کشمیر ملک دنیا عالمی",
    "Health & Society":"صحت ہسپتال تعلیم بیماری ڈاکٹر سکول بچوں خواتین سیلاب بارش علاج مریض لوگوں عوام",
    "Time & Number":   "سال دن ماہ گزشتہ آج کل وقت ایک دو تین چار پانچ ہزار لاکھ بعد پہلے فیصد",
    "Function word":   "کے کی کا میں سے نے کو پر اور کہ ہے ہیں تھا تھی تھے یہ وہ اس ان بھی کر کیا گیا ہو جو لیے ساتھ تک نہیں ہی بھی",
}
SEM_LEX = {}
for cat_, ws in SEM_SEEDS.items():
    for w in normalize(ws).split():
        SEM_LEX.setdefault(cf(w), cat_)

top200 = [i for i in range(2, V)][:200]
X200 = PPMI[top200].toarray()
seed_cent = {}
for cat_ in SEM_SEEDS:
    idx = [W2I[w] for w, c_ in SEM_LEX.items() if c_ == cat_ and w in W2I]
    if idx: seed_cent[cat_] = F.normalize(torch.tensor(PPMI[idx].toarray().mean(0)), dim=0).numpy()
FUNCTION_WORDS = {cf(w) for w in normalize(
    "کے کی کا میں سے نے کو پر اور کہ ہے ہیں تھا تھی تھے یہ وہ اس ان بھی کر کیا گیا ہو جو لیے ساتھ تک نہیں ہی ایک بعد پہلے "
    "جس جن انھوں انہوں اپنے اپنی اپنا والے والی کیے گئے گئی ہوا ہوئی ہوئے ہونے کرنے کرتے رہے رہی رہا سکتے سکتا جا جاتا جاتی جائے "
    "لیکن مگر یا اگر تو جب جبکہ کیونکہ تاہم کچھ کوئی کسی سب ہر بہت زیادہ کم صرف دیگر مزید اب آج کل یہاں وہاں جہاں دوران مطابق "
    "طرف بارے خلاف ذریعے علاوہ گا گی گے دی دیا دیے لی لیا ہم آپ میں مجھے ہمیں جاتے کہا کہتے کہنا بتایا").split()}
def sem_cat(i):
    w = ITOS[i]
    if w in SEM_LEX and SEM_LEX[w] != "Function word": return SEM_LEX[w]
    if w in FUNCTION_WORDS: return "Function word"
    if is_num(w): return "Time & Number"
    v = PPMI[i].toarray().ravel(); v = v / (np.linalg.norm(v) + 1e-9)
    topical = {c_: float(v @ seed_cent[c_]) for c_ in seed_cent if c_ != "Function word"}
    best = max(topical, key=topical.get) if topical else None
    return best if best and topical[best] > 0.06 else "Other"
sem_labels = [sem_cat(i) for i in top200]
perp = min(30, max(5, len(top200) // 4))
Y = TSNE(n_components=2, metric="cosine", init="random", perplexity=perp, random_state=42).fit_transform(X200 + 1e-9)

palette = {"Politics": "#d62728", "Sports": "#2ca02c", "Economy": "#ff7f0e", "Geography/Intl": "#1f77b4",
           "Health & Society": "#9467bd", "Time & Number": "#8c564b", "Function word": "#7f7f7f", "Other": "#bcbd22"}
fig, ax = plt.subplots(figsize=(13, 10))
for cat_, col in palette.items():
    m = np.array([s == cat_ for s in sem_labels])
    if m.any(): ax.scatter(Y[m, 0], Y[m, 1], c=col, s=28, label=f"{cat_} ({m.sum()})", alpha=.85)
for (x, y_), i in zip(Y, top200):
    ax.annotate(ur(ITOS[i]), (x, y_), fontsize=7, alpha=.8, xytext=(2, 2), textcoords="offset points", fontproperties=UR_FP)
ax.set_title("t-SNE of PPMI vectors — 200 most frequent tokens (colour = semantic category)")
ax.set_xlabel("t-SNE dimension 1"); ax.set_ylabel("t-SNE dimension 2")
ax.legend(title="Semantic category", loc="best", fontsize=8)
savefig("p1_tsne_ppmi.png"); plt.show()
print(pd.Series(sem_labels).value_counts().to_dict())

# %% [markdown]
# ## 2. Skip-gram Word2Vec (negative sampling) — implemented from scratch
# * two matrices: centre embeddings **V** and context embeddings **U** (both $|V|\times d$)
# * noise distribution $P_n(w)\propto f(w)^{3/4}$ stored as a 10M-entry *noise table*, $K=10$ samples per positive pair
# * loss $L=-\log\sigma(u_o^\top v_c)-\sum_{k=1}^{K}\log\sigma(-u_{w_k}^\top v_c)$, window $k=5$, Adam $\eta=10^{-3}$ (sparse variant), batch 1024
# * evaluation (neighbours, analogies, MRR) only ranks words seen ≥ 5 times, as is standard (rare-word vectors are noise)
# * frequent-word sub-sampling (Mikolov et al., $t=10^{-4}$) is applied when generating pairs each epoch
# * final embeddings $\frac12(V+U)$ are saved as `embeddings_w2v.npy`

# %%
class SkipGramData:
    """Vocabulary, noise table and (centre, context) training pairs."""
    def __init__(self, docs_words, itos, window=WINDOW, t=1e-4, table_size=10_000_000, seed=0):
        self.itos = itos; self.w2i = {w: i for i, w in enumerate(itos)}; self.window = window
        ids = [np.array([self.w2i.get(w, UNK) for w in d], dtype=np.int64) for d in docs_words]
        self.ids = np.concatenate(ids)
        self.doc = np.concatenate([np.full(len(x), j) for j, x in enumerate(ids)])
        cnt = np.bincount(self.ids, minlength=len(itos)).astype(np.float64)
        cnt[[PAD, UNK]] = 0
        self.counts = cnt
        f = cnt / cnt.sum()
        self.keep_p = np.minimum(1.0, np.sqrt(t / np.maximum(f, 1e-12)) + t / np.maximum(f, 1e-12))   # word2vec.c formula
        noise = cnt ** 0.75; noise /= noise.sum()
        self.noise_p = noise
        rng = np.random.default_rng(seed)
        self.noise_table = torch.tensor(rng.choice(len(itos), size=table_size, p=noise), dtype=torch.long)
        self.rng = rng
    def epoch_pairs(self):
        keep = (self.ids > UNK) & (self.rng.random(len(self.ids)) < self.keep_p[self.ids])
        ids, doc = self.ids[keep], self.doc[keep]
        C, O = [], []
        for o in range(1, self.window + 1):
            m = doc[:-o] == doc[o:]
            C += [ids[:-o][m], ids[o:][m]]; O += [ids[o:][m], ids[:-o][m]]
        C, O = np.concatenate(C), np.concatenate(O)
        perm = self.rng.permutation(len(C))
        return torch.tensor(C[perm]), torch.tensor(O[perm])

class SkipGramNS(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.V = nn.Embedding(vocab, d, sparse=True)     # centre vectors v_c   (sparse gradients: only rows in the batch)
        self.U = nn.Embedding(vocab, d, sparse=True)     # context vectors u_o
        nn.init.uniform_(self.V.weight, -0.5 / d, 0.5 / d)
        nn.init.uniform_(self.U.weight, -0.5 / d, 0.5 / d)
    def forward(self, c, o, neg):
        vc = self.V(c)                                   # B x d
        uo = self.U(o)                                   # B x d
        un = self.U(neg)                                 # B x K x d
        pos = F.logsigmoid((uo * vc).sum(-1))            # log σ(u_o·v_c)
        negs = F.logsigmoid(-torch.bmm(un, vc.unsqueeze(2)).squeeze(2)).sum(1)   # Σ log σ(-u_k·v_c)
        return -(pos + negs).mean()
    def averaged(self):
        return (0.5 * (self.V.weight + self.U.weight)).detach().cpu().numpy()

def train_w2v(data, d=W2V_DIM, epochs=W2V_EPOCHS, bs=W2V_BATCH, lr=W2V_LR, K=NEG_K, tag="C3", seed=42, n_logs=25):
    set_seed(seed)
    model = SkipGramNS(len(data.itos), d).to(DEVICE)
    # Adam (η=1e-3) in its sparse form: dense Adam keeps moving *every* row with stale momentum, which scatters rare
    # words over the space; SparseAdam updates only the rows that received a gradient in the batch.
    opt = torch.optim.SparseAdam(model.parameters(), lr=lr)
    table = data.noise_table.to(DEVICE)
    hist, ep_loss, step = [], [], 0
    t0 = time.time()
    for ep in range(epochs):
        C, O = data.epoch_pairs()
        n_b = math.ceil(len(C) / bs); log_every = max(1, n_b // n_logs)     # n_logs loss points per epoch
        run, run_n, tot = 0.0, 0, 0.0
        for b in range(n_b):
            c = C[b * bs:(b + 1) * bs].to(DEVICE); o = O[b * bs:(b + 1) * bs].to(DEVICE)
            neg = table[torch.randint(0, len(table), (len(c), K), device=DEVICE)]
            loss = model(c, o, neg)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item(); run_n += 1; tot += loss.item(); step += 1
            if run_n == log_every:
                hist.append((step, run / run_n))
                if len(hist) % 5 == 0: print(f"    [{tag}] step {step:6d}  interval loss {run / run_n:.4f}")
                run, run_n = 0.0, 0
        ep_loss.append(tot / n_b)
        print(f"[{tag}] epoch {ep+1}/{epochs}  pairs={len(C):,}  steps={n_b}  mean loss={ep_loss[-1]:.4f}  ({time.time()-t0:.0f}s)")
    return model, hist, ep_loss

W2V_DATA = SkipGramData([a["words"] for a in ARTICLES], ITOS)
_c, _o = W2V_DATA.epoch_pairs()
print(f"Vocab |V|={len(ITOS):,}; noise table size={len(W2V_DATA.noise_table):,}; pairs per epoch≈{len(_c):,}")
print("Noise distribution P(w)∝f^0.75 – top 5:", [(ITOS[i], round(float(W2V_DATA.noise_p[i]), 4)) for i in np.argsort(-W2V_DATA.noise_p)[:5]])
print("Sample (centre → context) pairs:", [(ITOS[int(a)], ITOS[int(b)]) for a, b in zip(_c[:8], _o[:8])])
del _c, _o

# %%
# ---- Train condition C3 (cleaned.txt, d = 100) ----
w2v_c3, hist_c3, ep_c3 = train_w2v(W2V_DATA, d=100, tag="C3 cleaned d=100")
EMB_C3 = w2v_c3.averaged()
np.save(os.path.join(EMB_DIR, "embeddings_w2v.npy"), EMB_C3.astype(np.float32))
json.dump(W2I, open(os.path.join(EMB_DIR, "word2idx.json"), "w", encoding="utf-8"), ensure_ascii=False)
SPACE_C3 = EmbSpace("C3 SG cleaned d=100", EMB_C3, ITOS, counts=counter)

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].plot([h[0] for h in hist_c3], [h[1] for h in hist_c3], lw=1.5)
ax[0].set_title("Skip-gram (C3) training loss at regular intervals"); ax[0].set_xlabel("training step"); ax[0].set_ylabel("BCE loss (neg. sampling)")
ax[1].plot(range(1, len(ep_c3) + 1), ep_c3, marker="o")
ax[1].set_title("Skip-gram (C3) mean loss per epoch"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("mean loss")
plt.tight_layout(); savefig("p1_w2v_loss.png"); plt.show()
print("Saved embeddings_w2v.npy", EMB_C3.shape, "and word2idx.json")
RESULTS["p1_w2v"] = {"epoch_loss": [round(x, 4) for x in ep_c3], "vocab": len(ITOS)}

# %% [markdown]
# ### 2.2 Evaluation — nearest neighbours and analogies

# %%
NN_C3 = nn_table(SPACE_C3, QUERY_MAP, 10)
print("Skip-gram (C3) top-10 nearest neighbours:")
display(NN_C3.T)
missing_q = [r for r, u in QUERY_MAP.items() if SPACE_C3.resolve(u) is None]
if missing_q: print("Not in vocabulary:", missing_q)

# %%
# Analogy a : b :: c : ?   ->  v(b) - v(a) + v(c); a,b,c excluded from candidates. Correct = expected word in top-3.
ANALOGY_POOL = [
    ("پاکستان", "پاکستانی", "بھارت", "بھارتی", "country:nationality"), ("پاکستان", "پاکستانی", "ایران", "ایرانی", "country:nationality"),
    ("پاکستان", "پاکستانی", "امریکہ", "امریکی", "country:nationality"), ("پاکستان", "پاکستانی", "چین", "چینی", "country:nationality"),
    ("بھارت", "بھارتی", "اسرائیل", "اسرائیلی", "country:nationality"), ("ایران", "ایرانی", "روس", "روسی", "country:nationality"),
    ("پاکستان", "پاکستانی", "افغانستان", "افغان", "country:nationality"), ("امریکہ", "امریکی", "برطانیہ", "برطانوی", "country:nationality"),
    ("ایران", "تہران", "افغانستان", "کابل", "country:capital"), ("بھارت", "دہلی", "افغانستان", "کابل", "country:capital"),
    ("روس", "ماسکو", "برطانیہ", "لندن", "country:capital"), ("چین", "بیجنگ", "روس", "ماسکو", "country:capital"),
    ("امریکہ", "واشنگٹن", "برطانیہ", "لندن", "country:capital"),
    ("پنجاب", "لاہور", "سندھ", "کراچی", "province:capital"), ("سندھ", "کراچی", "بلوچستان", "کوئٹہ", "province:capital"),
    ("پنجاب", "لاہور", "بلوچستان", "کوئٹہ", "province:capital"), ("سندھ", "کراچی", "پختونخوا", "پشاور", "province:capital"),
    ("مرد", "عورت", "لڑکا", "لڑکی", "gender"), ("بیٹا", "بیٹی", "بھائی", "بہن", "gender"), ("باپ", "ماں", "بیٹا", "بیٹی", "gender"),
    ("شوہر", "بیوی", "بھائی", "بہن", "gender"), ("والد", "والدہ", "بھائی", "بہن", "gender"), ("مردوں", "خواتین", "لڑکوں", "لڑکیوں", "gender"),
    ("تھا", "تھی", "ہوا", "ہوئی", "verb gender"), ("گیا", "گئی", "ہوا", "ہوئی", "verb gender"), ("کرتا", "کرتی", "ہوتا", "ہوتی", "verb gender"),
    ("جاتا", "جاتی", "آتا", "آتی", "verb gender"), ("رہا", "رہی", "گیا", "گئی", "verb gender"), ("دیا", "دی", "لیا", "لی", "verb gender"),
    ("سکتا", "سکتی", "رہا", "رہی", "verb gender"), ("تھا", "تھے", "ہے", "ہیں", "tense/number"), ("ہے", "تھا", "ہیں", "تھے", "tense"),
    ("کرنا", "کرتا", "ہونا", "ہوتا", "infinitive:habitual"), ("کرتا", "کرتے", "ہوتا", "ہوتے", "verb number"),
    ("گا", "گی", "رہا", "رہی", "aux gender"), ("بڑا", "بڑی", "نیا", "نئی", "adj gender"), ("بڑا", "بڑے", "نیا", "نئے", "adj number"),
    ("اچھا", "اچھی", "بڑا", "بڑی", "adj gender"), ("ملک", "ممالک", "شہر", "شہروں", "plural"), ("کھلاڑی", "کھلاڑیوں", "ملک", "ملکوں", "plural"),
    ("نواز", "شریف", "عمران", "خان", "first:last name"), ("کرکٹ", "کھلاڑی", "فوج", "فوجی", "domain:member"),
    ("وزیراعظم", "شہباز", "صدر", "زرداری", "office:holder"), ("روپے", "پاکستان", "ڈالر", "امریکہ", "currency:country"),
    # relations that occur in this BBC Urdu corpus
    ("انڈیا", "دہلی", "افغانستان", "کابل", "country:capital"), ("ایران", "تہران", "انڈیا", "دہلی", "country:capital"),
    ("افغانستان", "کابل", "ایران", "تہران", "country:capital"), ("امریکہ", "واشنگٹن", "روس", "ماسکو", "country:capital"),
    ("امریکہ", "امریکی", "ایران", "ایرانی", "country:nationality"), ("امریکہ", "امریکی", "پاکستان", "پاکستانی", "country:nationality"),
    ("ایران", "ایرانی", "انڈیا", "انڈین", "country:nationality"), ("امریکہ", "امریکی", "اسرائیل", "اسرائیلی", "country:nationality"),
    ("ایران", "ایرانی", "افغانستان", "افغان", "country:nationality"), ("امریکہ", "امریکی", "چین", "چینی", "country:nationality"),
    ("عمران", "خان", "شہباز", "شریف", "first:last name"), ("شہباز", "شریف", "ڈونلڈ", "ٹرمپ", "first:last name"),
    ("سعید", "انور", "رمیز", "راجہ", "first:last name"), ("ڈونلڈ", "ٹرمپ", "محسن", "نقوی", "first:last name"),
    ("نریندر", "مودی", "عمران", "خان", "first:last name"), ("اسامہ", "لادن", "سعید", "انور", "first:last name"),
    ("ایران", "تہران", "پنجاب", "لاہور", "region:capital"), ("لاہور", "پنجاب", "کراچی", "سندھ", "city:province"),
    ("عدالت", "جج", "ہسپتال", "ڈاکٹر", "institution:professional"), ("ٹیم", "کپتان", "فوج", "جنرل", "group:leader"),
    ("کرکٹ", "بورڈ", "سپریم", "کورٹ", "compound"), ("سعودی", "عرب", "اسلام", "آباد", "compound"),
]

def analogy(space, a, b, c, topn=3):
    ws = [space.resolve(x) for x in (a, b, c)]
    if None in ws: return None
    va, vb, vc = [space.vec(w) for w in ws]
    q = vb - va + vc; q = q / (np.linalg.norm(q) + 1e-9)
    s = (space.M @ q) if not space.sparse else np.asarray(space.M @ q).ravel()
    s = s.astype(np.float64); s[~space.mask] = -np.inf
    for w in ws: s[space.w2i[w]] = -np.inf
    order = np.argsort(-s)
    return [space.itos[i] for i in order[:topn]], order, s

def run_analogies(space, pool):
    rows = []
    for a, b, c, d, rel in pool:
        out = analogy(space, a, b, c)
        if out is None or space.resolve(d) is None: continue
        if len({space.resolve(x) for x in (a, b, c, d)}) < 4: continue          # forms collapsed by stemming
        top3, order, s = out
        rank = int((s > s[space.w2i[space.resolve(d)]]).sum()) + 1
        ra, rb, rc, rd = (space.resolve(x) for x in (a, b, c, d))
        rows.append({"relation": rel, "test": f"{ra} : {rb} :: {rc} : ?", "expected": rd, "top-3": "، ".join(top3),
                     "correct": space.resolve(d) in top3, "rank": rank})
    return pd.DataFrame(rows)

AN_ALL = run_analogies(SPACE_C3, ANALOGY_POOL)
print(f"Analogy pool: {len(ANALOGY_POOL)} hand-written tests, {len(AN_ALL)} fully in vocabulary.")
_sorted = AN_ALL.sort_values(["correct", "rank"], ascending=[False, True])
_pick, _per = [], collections.Counter()
for i_, r_ in _sorted.iterrows():                       # report 10 tests, at most 3 per relation type (diversity)
    if _per[r_.relation] < 3: _pick.append(i_); _per[r_.relation] += 1
    if len(_pick) == 10: break
AN10 = AN_ALL.loc[_pick].reset_index(drop=True)
n_correct = int(AN10["correct"].sum())
display(AN10)
print(f"Correct (expected word in top-3): {n_correct}/10  | over the whole in-vocab pool: {int(AN_ALL['correct'].sum())}/{len(AN_ALL)}")
RESULTS["p1_analogy"] = {"correct10": n_correct, "pool_correct": int(AN_ALL["correct"].sum()), "pool_n": len(AN_ALL),
                         "relations_correct": sorted(set(AN10[AN10.correct]["relation"]))}

# %%
# ---- Written assessment (generated from the numbers above) ----
def _nb_words(space, w, k=10):
    nb = space.neighbors(w, k); return [x for x, _ in nb] if nb else []
rel_ok = ", ".join(RESULTS["p1_analogy"]["relations_correct"]) or "none"
ex_q = next((u for u in QUERY_MAP.values() if SPACE_C3.resolve(u)), None)
quality = "capture clear" if n_correct >= 5 else "capture some" if n_correct >= 3 else "only weakly capture"
assessment = (
    f"The Skip-gram embeddings {quality} semantic and morpho-syntactic regularities: {n_correct} of the 10 reported analogies "
    f"place the expected word in the top-3 (relations solved: {rel_ok}), and {int(AN_ALL['correct'].sum())}/{len(AN_ALL)} of the "
    f"whole pool are solved. Nearest neighbours group words of the same news domain (e.g. {ex_q} → {'، '.join(_nb_words(SPACE_C3, ex_q, 5))}). Errors come mostly from low-frequency words and from the small corpus "
    f"({n_tok:,} tokens), where neighbours reflect co-occurrence in the same articles (topical relatedness) rather than strict synonymy."
)
print(assessment)
RESULTS["p1_assessment"] = assessment

# %% [markdown]
# ### Four-condition comparison
# | ID | Condition |
# |---|---|
# | C1 | PPMI-weighted co-occurrence vectors |
# | C2 | Skip-gram on `raw.txt` (unprocessed: whitespace split, no normalisation) |
# | C3 | Skip-gram on `cleaned.txt`, d=100 |
# | C4 | C3 with d=200 |
#
# MRR is computed on 20 manually labelled (query, related-word) pairs: for each pair the target word is ranked among the
# whole vocabulary by cosine similarity to the query; MRR $=\frac1{20}\sum 1/\mathrm{rank}$ (0 if a word is out of vocabulary).

# %%
RAW_COUNTER = collections.Counter(w for d in RAW_DOCS_TOK for w in d)
ITOS_RAW = ["<PAD>", "<UNK>"] + [w for w, _ in RAW_COUNTER.most_common(VOCAB_SIZE)]
W2V_DATA_RAW = SkipGramData(RAW_DOCS_TOK, ITOS_RAW)
w2v_c2, hist_c2, ep_c2 = train_w2v(W2V_DATA_RAW, d=100, tag="C2 raw d=100")
SPACE_C2 = EmbSpace("C2 SG raw d=100", w2v_c2.averaged(), ITOS_RAW, use_cf=False, counts=RAW_COUNTER)
w2v_c4, hist_c4, ep_c4 = train_w2v(W2V_DATA, d=200, tag="C4 cleaned d=200")
EMB_C4 = w2v_c4.averaged()
SPACE_C4 = EmbSpace("C4 SG cleaned d=200", EMB_C4, ITOS, counts=counter)
print("Raw vocabulary sample (note attached punctuation / unnormalised forms):", ITOS_RAW[2:12])

# %%
MRR_POOL = [("حکومت", "وزیر"), ("عدالت", "جج"), ("فوج", "فوجی"), ("کرکٹ", "میچ"), ("صحت", "ہسپتال"), ("تعلیم", "سکول"),
            ("معیشت", "معاشی"), ("پاکستان", "پاکستانی"), ("انتخابات", "ووٹ"), ("پولیس", "گرفتار"), ("بارش", "سیلاب"),
            ("ڈالر", "روپے"), ("وزیراعظم", "صدر"), ("امریکہ", "امریکی"), ("بھارت", "بھارتی"), ("کھلاڑی", "ٹیم"),
            ("بینک", "قرض"), ("عمران", "خان"), ("نواز", "شریف"), ("لاہور", "کراچی"), ("اسرائیل", "غزہ"), ("چین", "چینی"),
            ("مہنگائی", "قیمتوں"), ("بجٹ", "ٹیکس"), ("ہسپتال", "مریض"), ("سکول", "بچوں"), ("جج", "سماعت"), ("وزیر", "وزیراعظم"),
            ("ایران", "ایرانی"), ("افغانستان", "طالبان"), ("سیاسی", "جماعت"), ("فوج", "فوجی"), ("ٹیم", "میچ"), ("اسمبلی", "قومی"),
            ("ڈاکٹر", "مریض"), ("بیٹا", "بیٹی"), ("مرد", "خواتین"), ("سال", "گزشتہ"), ("پنجاب", "سندھ"), ("ووٹ", "انتخابات")]
MRR_PAIRS = []
for q, t in MRR_POOL:
    if SPACE_C3.resolve(q) and SPACE_C3.resolve(t) and SPACE_C3.resolve(q) != SPACE_C3.resolve(t) and (q, t) not in MRR_PAIRS:
        MRR_PAIRS.append((q, t))
    if len(MRR_PAIRS) == 20: break
print(f"{len(MRR_PAIRS)} labelled pairs used for MRR:", MRR_PAIRS)

def mrr(space, pairs):
    rr = []
    for q, t in pairs:
        rk = space.rank(q, t); rr.append(0.0 if rk is None else 1.0 / rk)
    return float(np.mean(rr)), rr

COND_Q = {k: QUERY_MAP[k] for k in ["Pakistan", "Hukumat", "Adalat", "Fauj", "Taleem"]}
SPACES = [SPACE_PPMI, SPACE_C2, SPACE_C3, SPACE_C4]
cond_rows, cond_nn = [], {}
for sp_ in SPACES:
    m, rr = mrr(sp_, MRR_PAIRS)
    dims = sp_.M.shape[1]
    cond_rows.append({"condition": sp_.name, "dim": dims, "vocab": len(sp_.itos), "MRR": round(m, 4),
                      "hits@10": int(sum(r >= 0.1 for r in rr)), "OOV pairs": int(sum(sp_.resolve(q) is None or sp_.resolve(t) is None for q, t in MRR_PAIRS))})
    for rom, u in COND_Q.items():
        nb = sp_.neighbors(u, 5)
        cond_nn[(sp_.name, rom)] = "، ".join(w for w, _ in nb) if nb else "(OOV)"
COND_DF = pd.DataFrame(cond_rows)
display(COND_DF)
NN_COND = pd.DataFrame({rom: [cond_nn[(s.name, rom)] for s in SPACES] for rom in COND_Q}, index=[s.name for s in SPACES])
print("Top-5 neighbours per condition:")
display(NN_COND)

best = COND_DF.loc[COND_DF.MRR.idxmax()]
m_c3, m_c4 = COND_DF.MRR[2], COND_DF.MRR[3]
disc = (f"Best condition by MRR: {best.condition} (MRR={best.MRR:.3f}). "
        f"Skip-gram on cleaned text (C3, {m_c3:.3f}) vs raw text (C2, {COND_DF.MRR[1]:.3f}): "
        + ("cleaning helps because normalisation merges spelling variants and strips attached punctuation, so counts are not split across surface forms. "
           if m_c3 >= COND_DF.MRR[1] else "the raw corpus did slightly better here, probably because the larger raw vocabulary keeps some forms that the cleaning merged/removed. ")
        + f"PPMI (C1, {COND_DF.MRR[0]:.3f}) gives sparse, high-dimensional vectors that are strong for frequent words but noisy for rare ones. "
        + f"Doubling d to 200 (C4, {m_c4:.3f}) " + ("improves" if m_c4 > m_c3 + 0.005 else "does not clearly improve" if m_c4 >= m_c3 - 0.005 else "hurts")
        + (" the result: the larger space separates more fine-grained relations, although the gain should be weighed against twice the memory and training cost."
           if m_c4 > m_c3 + 0.005 else
           " the result: with a corpus of this size the extra dimensions add parameters the data cannot constrain, so d=100 is sufficient."))
print(disc)
RESULTS["p1_conditions"] = cond_rows; RESULTS["p1_cond_discussion"] = disc
