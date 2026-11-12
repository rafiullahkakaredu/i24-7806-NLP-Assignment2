# %% [markdown]
# # DS 5007 — Natural Language Processing · Assignment 2
# ## Neural NLP Pipeline — continuation of the BBC Urdu NLP Pipeline
#
# | | |
# |---|---|
# | **Name** | Rafiullah |
# | **Roll No.** | i24-7806 |
# | **GitHub** | https://github.com/rafiullahkakaredu/i24-7806-NLP-Assignment2 |
#
# **Restrictions honoured:** no pretrained models, no Gensim, no HuggingFace. Every model (Skip-gram,
# BiLSTM-CRF, Transformer encoder) is written from scratch in PyTorch. `nn.Transformer`,
# `nn.MultiheadAttention` and `nn.TransformerEncoder` are **not** used anywhere.
#
# **How to run on Google Colab**
# 1. `Runtime → Change runtime type → GPU` (T4 is enough).
# 2. Upload `cleaned.txt`, `raw.txt` and `Metadata.json` (from Assignment 1) to `/content`
#    (the left *Files* panel), or let the upload cell below ask for them.
# 3. `Runtime → Run all`. Every cell prints output. The last cell writes `report.pdf` and a zip
#    with the required submission layout.

# %%
# ===================== 0. Setup: packages, fonts, imports =====================
import os, sys, subprocess, re, json, math, time, random, collections, unicodedata, shutil, glob, warnings
warnings.filterwarnings("ignore")
IN_COLAB = "google.colab" in sys.modules

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

for mod, pkg in [("arabic_reshaper", "arabic-reshaper"), ("bidi", "python-bidi"), ("reportlab", "reportlab")]:
    try:
        __import__(mod)
    except ImportError:
        _pip(pkg)

import numpy as np
import pandas as pd
import matplotlib
if not IN_COLAB and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import arabic_reshaper
from bidi.algorithm import get_display

try:
    display
except NameError:          # running as a plain .py script
    def display(x):
        print(x.to_string() if hasattr(x, "to_string") else x)

# ---- Urdu-capable font for matplotlib (labels in t-SNE / heatmaps) ----
def _find_urdu_font():
    cands = glob.glob("/usr/share/fonts/**/NotoNaskhArabic-Regular.ttf", recursive=True) + \
            glob.glob("/usr/share/fonts/**/NotoNastaliqUrdu-Regular.ttf", recursive=True) + \
            glob.glob("./fonts/*.ttf")
    return cands[0] if cands else None

URDU_FONT = _find_urdu_font()
if URDU_FONT is None and IN_COLAB:
    subprocess.run("apt-get -qq install -y fonts-noto-core > /dev/null 2>&1", shell=True)
    URDU_FONT = _find_urdu_font()
if URDU_FONT is None:
    try:
        import urllib.request
        os.makedirs("fonts", exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/fonts/NotoNaskhArabic/hinted/ttf/NotoNaskhArabic-Regular.ttf",
            "fonts/NotoNaskhArabic-Regular.ttf")
        URDU_FONT = "fonts/NotoNaskhArabic-Regular.ttf"
    except Exception as e:
        print("Could not fetch an Urdu font (labels may show as boxes):", e)
if URDU_FONT:
    fm.fontManager.addfont(URDU_FONT)
    _fname = fm.FontProperties(fname=URDU_FONT).get_name()
    plt.rcParams["font.family"] = ["DejaVu Sans", _fname]
UR_FP = fm.FontProperties(fname=URDU_FONT) if URDU_FONT else None

_MPL_NATIVE_RTL = tuple(int(x) for x in re.findall(r"\d+", matplotlib.__version__)[:2]) >= (3, 11)   # mpl>=3.11 shapes RTL itself
def ur(s):
    """Shape + reorder Urdu text so matplotlib draws it correctly (right-to-left, joined letters)."""
    if _MPL_NATIVE_RTL:
        return str(s)
    try:
        return get_display(arabic_reshaper.reshape(str(s)))
    except Exception:
        return str(s)

plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 11, "axes.labelsize": 10})

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Python", sys.version.split()[0], "| torch", torch.__version__, "| device:", DEVICE)
print("Colab:", IN_COLAB, "| Urdu font:", URDU_FONT)

# %%
# ===================== 0.1 Configuration =====================
STUDENT_NAME = "Rafiullah"
ROLL_NO      = "i24-7806"
SECTION      = "X"                  # <-- put your section letter here (e.g. "A")
GITHUB_URL   = f"https://github.com/rafiullahkakaredu/{ROLL_NO}-NLP-Assignment2"

FAST = os.environ.get("A2_FAST", "0") == "1"      # quick smoke-test mode (never used for submission)

DATA_DIR = os.environ.get("A2_DATA_DIR", "/content" if IN_COLAB else ".")
SUB_NAME = f"{ROLL_NO}_Assignment2_DS-{SECTION}"
OUT_DIR  = os.environ.get("A2_OUT_DIR", os.path.join("/content" if IN_COLAB else ".", SUB_NAME))
EMB_DIR, MODEL_DIR, DATA_OUT, FIG_DIR = [os.path.join(OUT_DIR, d) for d in ("embeddings", "models", "data", "figures")]
for d in (EMB_DIR, MODEL_DIR, DATA_OUT, FIG_DIR):
    os.makedirs(d, exist_ok=True)

# Hyper-parameters fixed by the assignment
VOCAB_SIZE   = 10_000      # most frequent tokens kept; rest -> <UNK>
WINDOW       = 5           # k
NEG_K        = 10          # K noise samples
W2V_DIM      = 100         # d
W2V_LR       = 1e-3        # Adam
W2V_BATCH    = 1024        # >= 512
W2V_EPOCHS   = 2 if FAST else 10      # >= 5
SEQ_MAX_EPOCHS = 4 if FAST else 60    # early stopping (patience 5) decides the real stop
TR_EPOCHS    = 3 if FAST else 20
MAX_LEN      = 256

RESULTS = {"student": STUDENT_NAME, "roll": ROLL_NO, "github": GITHUB_URL}   # everything for the report
def savefig(name):
    p = os.path.join(FIG_DIR, name); plt.savefig(p, bbox_inches="tight", dpi=150); return p
print("Output folder:", OUT_DIR, "| FAST mode:", FAST)

# %%
# ===================== 0.2 Locate / upload the Assignment-1 files =====================
REQUIRED = ["cleaned.txt", "raw.txt", "Metadata.json"]

def find_file(name):
    roots = [DATA_DIR, ".", "/content", "/content/drive/MyDrive", "/content/data"]
    for r in roots:
        if not os.path.isdir(r):
            continue
        for f in os.listdir(r):
            if f.lower() == name.lower():
                return os.path.join(r, f)
    hits = [p for p in glob.glob(f"/content/**/{name}", recursive=True)] if IN_COLAB else []
    return hits[0] if hits else None

paths = {n: find_file(n) for n in REQUIRED}
if IN_COLAB and any(v is None for v in paths.values()):
    from google.colab import files
    print("Please upload:", [n for n, v in paths.items() if v is None])
    files.upload()               # files land in the current folder (/content)
    paths = {n: find_file(n) for n in REQUIRED}
for n, p in paths.items():
    print(f"{n:15s} -> {p}  ({os.path.getsize(p)/1e6:.2f} MB)" if p else f"{n:15s} -> MISSING")
assert paths["cleaned.txt"] and paths["Metadata.json"], "cleaned.txt and Metadata.json are required"
if not paths["raw.txt"]:
    print("WARNING: raw.txt missing – condition C2 will reuse cleaned.txt without cleaning-normalisation.")

# %%
# ===================== 0.3 Urdu text utilities (normaliser, tokeniser, stemmer/lemmatiser from A1) =====================
_DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_CHAR_MAP = str.maketrans({
    "\u064A": "\u06CC", "\u0649": "\u06CC",   # Arabic yeh / alef maksura -> Farsi yeh
    "\u0643": "\u06A9",                                               # Arabic kaf -> keheh
    "\u0647": "\u06C1", "\u06D5": "\u06C1", "\u0629": "\u06C1",       # Arabic heh / teh marbuta -> gol heh
    "\u200c": " ", "\u200b": "", "\u200d": "", "\u00a0": " ", "\ufeff": "",
    **{chr(0x06F0 + i): str(i) for i in range(10)}, **{chr(0x0660 + i): str(i) for i in range(10)},
})
PUNCT_CHARS = set("۔،؟!؛٪\"'()[]{}<>:;,.?«»…-–—/\\|*+=#@&%^~`’‘“”•·")

def normalize(text):
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("آ", "آ")      # alef + madda -> آ
    text = text.translate(_CHAR_MAP)
    text = _DIAC.sub("", text)
    return re.sub(r"[ \t]+", " ", text)

_TOK = re.compile(r"<NUM>|\w+|[^\w\s]", re.UNICODE)      # <NUM> = number placeholder used by the A1 cleaner
def tokenize(text, norm=True):
    return _TOK.findall(normalize(text) if norm else text)

def is_punct(t): return all((c in PUNCT_CHARS) or unicodedata.category(c).startswith("P") or unicodedata.category(c).startswith("S") for c in t)
def is_num(t):   return t == "<NUM>" or bool(re.fullmatch(r"\d+([.,]\d+)?", t))
def is_latin(t): return bool(re.search(r"[A-Za-z]", t))
def is_word(t):  return any(c.isalpha() for c in t) or is_num(t)

# ---- Rule-based Urdu stemmer + lemmatiser (Assignment 1 style: longest-suffix stripping + exceptions) ----
SUFFIXES = sorted(["یاں", "یوں", "وں", "یں", "ات", "یات", "ئیں", "ئے", "ئی", "یا", "تا", "تی", "تے", "نا", "نے", "نی",
                   "گا", "گی", "گے", "ے", "ی", "ا", "و"], key=len, reverse=True)
LEMMA_EXCEPTIONS = {"گیا": "جا", "گئی": "جا", "گئے": "جا", "گئیں": "جا", "کیا": "کر", "کی": "کر", "کیے": "کر", "کئے": "کر",
                    "دیا": "دے", "دی": "دے", "دیے": "دے", "دئیے": "دے", "لیا": "لے", "لی": "لے", "لیے": "لے",
                    "ہوا": "ہو", "ہوئی": "ہو", "ہوئے": "ہو", "ہوئیں": "ہو", "تھا": "ہے", "تھی": "ہے", "تھے": "ہے", "ہیں": "ہے"}
def stem(w, min_len=2):
    for s in SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= min_len:
            return w[: -len(s)]
    return w
def lemmatize(w):
    return LEMMA_EXCEPTIONS.get(w, stem(w))

_demo = "پاکستان کی حکومت نے بدھ کو نئے بجٹ کا اعلان کیا۔ وزیراعظم نے کہا کہ معیشت بہتر ہو رہی ہے!"
print(tokenize(_demo))
print([(w, stem(w), lemmatize(w)) for w in ["حکومتوں", "کھلاڑیوں", "کرتے", "گئی", "انتخابات", "ممالک"]])

# %%
# ===================== 0.4 Load corpus + metadata (format auto-detection) =====================
def read_text(p):
    with open(p, encoding="utf-8-sig", errors="replace") as f:
        return f.read()

_MARKERS = [
    r"^\s*=+\s*ARTICLE\s+(\d+)\s*=+\s*()$",                                    # ===== ARTICLE 12 =====  (A1 format)
    r"^\s*\[\s*(\d+)\s*\]\s*(.*)$",                                         # [12] ...
    r"^\s*(?:article|doc|document|id|مضمون)\s*[#:\-]?\s*(\d+)\s*[:\-\)]?\s*(.*)$",  # Article 12: ...
    r"^\s*<doc[^>]*?(?:id\s*=\s*\"?(\d+)\"?)?[^>]*>\s*(.*)$",                # <doc id="12">
    r"^\s*#{1,6}\s*(\d+)\s*(.*)$",                                          # ## 12
    r"^\s*(\d+)\s*[\.\):]\s*$",                                             # 12.
]
_SEP = re.compile(r"^\s*(?:={3,}|-{3,}|\*{3,}|_{3,}|#{3,}|~{3,}|</doc>|=+\s*END\s+ARTICLE\s*\d*\s*=+)\s*$", re.I)

def candidate_splits(text):
    lines = text.splitlines()
    cands = []
    for pat in _MARKERS:
        rx = re.compile(pat, re.I)
        hits = [(i, rx.match(l)) for i, l in enumerate(lines) if rx.match(l)]
        if len(hits) >= 3:
            ids, docs = [], []
            for j, (i, m) in enumerate(hits):
                end = hits[j + 1][0] if j + 1 < len(hits) else len(lines)
                body = "\n".join(([m.group(2)] if m.lastindex and m.lastindex >= 2 and m.group(2) else []) + lines[i + 1:end])
                body = "\n".join(l for l in body.splitlines() if not _SEP.match(l)).strip()
                ids.append(m.group(1) if m.group(1) else str(j + 1)); docs.append(body)
            cands.append(("marker:" + pat[:18], ids, docs))
    if sum(bool(_SEP.match(l)) for l in lines) >= 3:
        blocks, cur = [], []
        for l in lines:
            if _SEP.match(l):
                if "\n".join(cur).strip(): blocks.append("\n".join(cur).strip())
                cur = []
            else:
                cur.append(l)
        if "\n".join(cur).strip(): blocks.append("\n".join(cur).strip())
        cands.append(("separator", [str(i + 1) for i in range(len(blocks))], blocks))
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    cands.append(("blank-line", [str(i + 1) for i in range(len(blocks))], blocks))
    ls = [l.strip() for l in lines if l.strip()]
    cands.append(("line", [str(i + 1) for i in range(len(ls))], ls))
    return cands

def split_documents(text, n_expected=None, force=None):
    cands = candidate_splits(text)
    if force:
        cands = [c for c in cands if c[0].startswith(force)] or cands
    if n_expected:
        best = min(cands, key=lambda c: abs(len(c[2]) - n_expected))
    else:
        best = cands[0]
    return best

def load_metadata(p):
    m = json.load(open(p, encoding="utf-8-sig"))
    if isinstance(m, dict):
        for key in ("articles", "data", "items", "records", "docs", "documents"):
            if key in m and isinstance(m[key], (list, dict)):
                m = m[key]; break
    out = []
    if isinstance(m, dict):
        for k, v in m.items():
            v = v if isinstance(v, dict) else {"value": v}
            out.append({"_id": str(k), **v})
    else:
        for i, v in enumerate(m):
            v = v if isinstance(v, dict) else {"value": v}
            vid = next((v[k] for k in ("id", "article_id", "doc_id", "article_number", "number", "index") if k in v), i + 1)
            out.append({"_id": str(vid), **v})
    return out

def meta_text(entry):
    parts = []
    for k, v in entry.items():
        if k == "_id": continue
        if isinstance(v, str): parts.append(v)
        elif isinstance(v, list): parts.extend(str(x) for x in v if isinstance(x, (str, int)))
    return " ".join(parts)

META = load_metadata(paths["Metadata.json"])
DOC_SPLIT = None          # set to "marker" / "separator" / "blank-line" / "line" to force a format
how, doc_ids, clean_docs = split_documents(read_text(paths["cleaned.txt"]), len(META), DOC_SPLIT)
print(f"Metadata entries: {len(META)} | cleaned.txt split by '{how}' -> {len(clean_docs)} documents")
print("Metadata keys of first entry:", list(META[0].keys()))

# align documents with metadata: by id when ids overlap, else by order
meta_by_id = {e["_id"]: e for e in META}
overlap = sum(i in meta_by_id for i in doc_ids)
ARTICLES = []
if overlap >= 0.5 * len(doc_ids):
    for i, d in zip(doc_ids, clean_docs):
        ARTICLES.append({"id": i, "meta": meta_by_id.get(i, {}), "text": d})
    print(f"Aligned by id ({overlap}/{len(doc_ids)} ids matched)")
else:
    for j, d in enumerate(clean_docs):
        ARTICLES.append({"id": doc_ids[j], "meta": META[j] if j < len(META) else {}, "text": d})
    print("Aligned by order")
# ---- remove boilerplate: lines repeated in >= 5% of articles (BBC footer, "playback not supported", ...) ----
_line_df = collections.Counter(l for a in ARTICLES for l in {x.strip() for x in a["text"].splitlines() if x.strip()})
BOILER = {l for l, n in _line_df.items() if n >= max(3, 0.05 * len(ARTICLES))}
for a in ARTICLES:
    a["text"] = "\n".join(x for x in a["text"].splitlines() if x.strip() and x.strip() not in BOILER)
print(f"Removed {len(BOILER)} boilerplate line types (e.g. {next(iter(BOILER), '')[:70]!r})")
# ---- drop exact duplicate articles ----
_seen, _dedup = set(), []
for a in ARTICLES:
    key = a["text"].strip()
    if key and key not in _seen: _seen.add(key); _dedup.append(a)
print(f"Duplicate articles removed: {len(ARTICLES) - len(_dedup)}")
ARTICLES = [a for a in _dedup if len(tokenize(a["text"])) >= 5]
for a in ARTICLES:
    a["tokens"] = tokenize(a["text"])                 # with punctuation (Part 2)
    a["words"] = [t for t in a["tokens"] if is_word(t)]  # word tokens (Parts 1 & 3)

if paths["raw.txt"]:
    _, raw_ids, raw_docs = split_documents(read_text(paths["raw.txt"]), len(META), DOC_SPLIT)
else:
    raw_ids, raw_docs = [a["id"] for a in ARTICLES], [a["text"] for a in ARTICLES]
RAW_DOCS_TOK = [d.split() for d in raw_docs if d.strip()]   # raw = whitespace split only, no normalisation

# ---- raw word -> cleaned (A1-stemmed) form map, learned by aligning raw.txt with cleaned.txt per article ----
import difflib
def _raw_tokens(s):
    s = re.sub(r"\d+([.,]\d+)?", "<NUM>", normalize(s))
    return [t for t in tokenize(s, norm=False) if not is_punct(t) or t in "۔،؟"]
_pairs = collections.defaultdict(collections.Counter)
_raw_by_id = dict(zip(raw_ids, raw_docs))
for a in ARTICLES:
    if a["id"] not in _raw_by_id: continue
    r_, c_ = _raw_tokens(_raw_by_id[a["id"]]), a["tokens"]
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, r_, c_, autojunk=False).get_opcodes():
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            for x, y in zip(r_[i1:i2], c_[j1:j2]): _pairs[x][y] += 1
A1_MAP = {w: c.most_common(1)[0][0] for w, c in _pairs.items()}
def cf(w):
    """Map a dictionary (raw-script) word to the form it takes in cleaned.txt (A1 normalisation + stemming)."""
    w = normalize(w); return A1_MAP.get(w, w)
_changed = {w: c for w, c in A1_MAP.items() if w != c}
print(f"A1 form map: {len(A1_MAP):,} raw words aligned, {len(_changed):,} change form in cleaned.txt, e.g.",
      [(w, cf(w)) for w in ["پاکستان", "امریکہ", "کھلاڑیوں", "گیا", "گئی", "کراچی", "حکومت"]])

n_tok = sum(len(a["words"]) for a in ARTICLES)
print(f"Usable articles: {len(ARTICLES)} | word tokens (cleaned): {n_tok:,} | unique: {len(set(w for a in ARTICLES for w in a['words'])):,}")
print(f"raw.txt documents: {len(RAW_DOCS_TOK)} | raw tokens: {sum(map(len, RAW_DOCS_TOK)):,}")
print("Sample:", " ".join(ARTICLES[0]["tokens"][:25]))
RESULTS["corpus"] = {"articles": len(ARTICLES), "tokens": n_tok, "split": how}
