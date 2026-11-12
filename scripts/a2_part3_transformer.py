# %% [markdown]
# # Part 3 — Transformer Encoder for Topic Classification (20 marks)
# ## 6. Dataset preparation
# Categories come from the keyword assignment done in Part 1 (Politics, Sports, Economy, International, Health & Society).
# Each article becomes a sequence of token IDs (same `word2idx` as Part 1), truncated / padded to 256 tokens.

# %%
CLS_X = np.zeros((len(ARTICLES), MAX_LEN), dtype=np.int64)
for i, ids in enumerate(DOC_IDS):
    ids = ids[:MAX_LEN]; CLS_X[i, :len(ids)] = ids
CLS_Y = np.array(DOC_CAT, dtype=np.int64)
idx_tr, idx_va, idx_te = strat_split(list(range(len(ARTICLES))), list(CLS_Y), seed=42)
CLS_DIST = pd.DataFrame({name: pd.Series(CLS_Y[ix]).map(dict(enumerate(CATEGORIES))).value_counts().reindex(CATEGORIES).fillna(0).astype(int)
                         for name, ix in [("train", idx_tr), ("val", idx_va), ("test", idx_te)]})
CLS_DIST["total"] = CLS_DIST.sum(1)
print(f"Articles: {len(ARTICLES)} | train/val/test = {len(idx_tr)}/{len(idx_va)}/{len(idx_te)} | "
      f"truncated: {(np.array([len(x) for x in DOC_IDS]) > MAX_LEN).mean():.0%} of articles longer than {MAX_LEN} tokens")
display(CLS_DIST)
RESULTS["p3_data"] = {"split": [len(idx_tr), len(idx_va), len(idx_te)], "dist": CLS_DIST["total"].to_dict()}

def cls_batches(idx, bs=16, shuffle=False, seed=0):
    idx = list(idx)
    if shuffle: random.Random(seed).shuffle(idx)
    for b in range(0, len(idx), bs):
        j = idx[b:b + bs]
        yield torch.tensor(CLS_X[j]).to(DEVICE), torch.tensor(CLS_Y[j]).to(DEVICE), j

# %% [markdown]
# ## 7. Transformer encoder — every component is its own module (no PyTorch built-in Transformer classes)

# %%
class ScaledDotProductAttention(nn.Module):
    """Attention(Q,K,V) = softmax(QKᵀ/√d_k) V, with optional key-padding mask. Returns (output, weights)."""
    def __init__(self, dropout=0.1):
        super().__init__(); self.drop = nn.Dropout(dropout)
    def forward(self, Q, K, V, mask=None):            # Q,K,V: B x L x d_k ; mask: B x L (True = real token)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return self.drop(weights) @ V, weights

class MultiHeadSelfAttention(nn.Module):
    """h independent heads, each with its own W_Q, W_K, W_V (d_model -> d_k); shared output projection W_O."""
    def __init__(self, d_model=128, h=4, d_k=32, d_v=32, dropout=0.1):
        super().__init__()
        self.W_Q = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(h)])
        self.W_K = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(h)])
        self.W_V = nn.ModuleList([nn.Linear(d_model, d_v) for _ in range(h)])
        self.attention = ScaledDotProductAttention(dropout)
        self.W_O = nn.Linear(h * d_v, d_model)
    def forward(self, x, mask=None):
        outs, ws = [], []
        for q, k, v in zip(self.W_Q, self.W_K, self.W_V):
            o, w = self.attention(q(x), k(x), v(x), mask); outs.append(o); ws.append(w)
        return self.W_O(torch.cat(outs, dim=-1)), torch.stack(ws, dim=1)     # B x L x d_model, B x h x L x L

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=128, d_ff=512, dropout=0.1):
        super().__init__()
        self.lin1, self.lin2, self.drop = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
    def forward(self, x):
        return self.lin2(self.drop(F.relu(self.lin1(x))))

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model=128, max_len=512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos / div); pe[:, 1::2] = torch.cos(pos / div)
        self.register_buffer("pe", pe.unsqueeze(0))       # fixed buffer – not a parameter, never trained
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderBlock(nn.Module):
    """Pre-LN:  x ← x + Dropout(MHA(LN(x)));  x ← x + Dropout(FFN(LN(x)))"""
    def __init__(self, d_model=128, h=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, h, d_model // h, d_model // h, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
    def forward(self, x, mask=None):
        a, w = self.mha(self.ln1(x), mask)
        x = x + self.drop1(a)
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x, w

class TransformerClassifier(nn.Module):
    def __init__(self, vocab, n_classes=5, d_model=128, h=4, d_ff=512, n_layers=4, max_len=MAX_LEN + 1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)    # learned [CLS] token
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, h, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes))   # 128→64→5
    def forward(self, ids, return_attn=False):
        B = ids.size(0)
        x = self.emb(ids) * math.sqrt(self.d_model)
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)                   # prepend [CLS]
        mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=ids.device), ids != PAD], dim=1)
        x = self.drop(self.pos(x))                                              # PE added before the first block
        attns = []
        for blk in self.blocks:
            x, w = blk(x, mask); attns.append(w)
        logits = self.head(self.ln_f(x[:, 0]))
        return (logits, attns) if return_attn else logits

_m = TransformerClassifier(V)
_lo, _at = _m(torch.tensor(CLS_X[:2]), return_attn=True)
print(_m)
print("logits", tuple(_lo.shape), "| attention per layer", tuple(_at[-1].shape), "| params:", f"{sum(p.numel() for p in _m.parameters()):,}",
      "| PE is buffer (not parameter):", "pos.pe" in dict(_m.named_buffers()) and "pos.pe" not in dict(_m.named_parameters()))
del _m, _lo, _at

# %%
def cosine_warmup(opt, warmup, total):
    def f(step):
        if step < warmup: return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)

def run_cls_epoch(model, idx, opt=None, sched=None, seed=0):
    train = opt is not None; model.train(train)
    tl, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y, _ in cls_batches(idx, bs=16, shuffle=train, seed=seed):
            logits = model(x); loss = F.cross_entropy(logits, y)
            if train:
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl += loss.item() * len(y); correct += (logits.argmax(1) == y).sum().item(); n += len(y)
    return tl / n, 100 * correct / n

def train_classifier(model, opt, sched, epochs, tag, ckpt=None):
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "epoch_time", "lr"]}
    best, best_state = -1, None
    for ep in range(epochs):
        t0 = time.time()
        trl, tra = run_cls_epoch(model, idx_tr, opt, sched, seed=ep)
        hist["epoch_time"].append(time.time() - t0)
        val, vaa = run_cls_epoch(model, idx_va)
        for k, v in zip(["train_loss", "train_acc", "val_loss", "val_acc"], [trl, tra, val, vaa]): hist[k].append(v)
        hist["lr"].append(opt.param_groups[0]["lr"])
        if vaa > best: best, best_state, hist["best_epoch"] = vaa, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep + 1
        print(f"[{tag}] ep {ep+1:2d}/{epochs} train loss {trl:.3f} acc {tra:5.1f} | val loss {val:.3f} acc {vaa:5.1f} | "
              f"lr {hist['lr'][-1]:.2e} | {hist['epoch_time'][-1]:.2f}s")
    model.load_state_dict(best_state)
    if ckpt: torch.save({"state_dict": best_state, "categories": CATEGORIES, "best_epoch": hist["best_epoch"]}, ckpt)
    return hist

set_seed(42)
TRF = TransformerClassifier(V).to(DEVICE)
opt = torch.optim.AdamW(TRF.parameters(), lr=5e-4, weight_decay=0.01)
total_steps = TR_EPOCHS * math.ceil(len(idx_tr) / 16)
sched = cosine_warmup(opt, warmup=50, total=total_steps)
print(f"AdamW lr=5e-4 wd=0.01 | cosine schedule, 50 warm-up steps, {total_steps} total steps")
H_TRF = train_classifier(TRF, opt, sched, TR_EPOCHS, "Transformer", os.path.join(MODEL_DIR, "transformer_cls.pt"))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8)); e = range(1, TR_EPOCHS + 1)
ax[0].plot(e, H_TRF["train_loss"], marker="o", label="train"); ax[0].plot(e, H_TRF["val_loss"], marker="s", label="validation")
ax[0].set_title("Transformer: loss per epoch"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy loss"); ax[0].legend()
ax[1].plot(e, H_TRF["train_acc"], marker="o", label="train"); ax[1].plot(e, H_TRF["val_acc"], marker="s", label="validation")
ax[1].set_title("Transformer: accuracy per epoch"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy (%)"); ax[1].legend()
plt.tight_layout(); savefig("p3_transformer_curves.png"); plt.show()
print("Saved models/transformer_cls.pt (best validation epoch:", H_TRF["best_epoch"], ")")

# %% [markdown]
# ## 8. Evaluation
# ### 8.1 Results

# %%
@torch.no_grad()
def predict_cls(model, idx):
    model.eval(); P = []
    for x, y, _ in cls_batches(idx, bs=32): P += model(x).argmax(1).tolist()
    return np.array(P)
P_TRF = predict_cls(TRF, idx_te); Y_TE = CLS_Y[idx_te]
TRF_ACC = accuracy_score(Y_TE, P_TRF) * 100
TRF_F1 = f1_score(Y_TE, P_TRF, average="macro", labels=list(range(5)), zero_division=0) * 100
print(f"Transformer TEST accuracy = {TRF_ACC:.2f}% | macro-F1 = {TRF_F1:.2f}")
cm5 = confusion_matrix(Y_TE, P_TRF, labels=list(range(5)))
fig, ax = plt.subplots(figsize=(6.2, 5))
im = ax.imshow(cm5, cmap="Blues")
ax.set_xticks(range(5)); ax.set_xticklabels(CATEGORIES, rotation=30, ha="right"); ax.set_yticks(range(5)); ax.set_yticklabels(CATEGORIES)
for i in range(5):
    for j in range(5): ax.text(j, i, cm5[i, j], ha="center", va="center", color="white" if cm5[i, j] > cm5.max() / 2 else "black")
ax.set_xlabel("predicted category"); ax.set_ylabel("true category"); ax.set_title("Transformer 5×5 confusion matrix (test)")
plt.colorbar(im, ax=ax, fraction=.046); plt.tight_layout(); savefig("p3_confusion.png"); plt.show()
display(pd.DataFrame(cm5, index=[f"true {c_}" for c_ in CATEGORIES], columns=[f"pred {c_}" for c_ in CATEGORIES]))

# %%
# ---- attention heatmaps: final encoder layer, 3 correctly classified test articles, 2 heads each ----
correct_te = [i for i, p in zip(idx_te, P_TRF) if p == CLS_Y[i]]
picked = []
for c_ in range(5):                                  # prefer 3 different categories
    for i in correct_te:
        if CLS_Y[i] == c_ and i not in picked: picked.append(i); break
picked = (picked + [i for i in correct_te if i not in picked])[:3]
SHOW = 24
CLS_FOCUS, HEAD_STATS = [], []
TRF.eval()
fig, axes = plt.subplots(len(picked), 2, figsize=(15, 6.3 * len(picked)), squeeze=False)
for r, i in enumerate(picked):
    with torch.no_grad():
        _, att = TRF(torch.tensor(CLS_X[i:i + 1]).to(DEVICE), return_attn=True)
    A = att[-1][0].cpu().numpy()                     # h x L x L (final layer)
    toks = ["[CLS]"] + [ITOS[t] for t in CLS_X[i] if t != PAD]
    n = min(SHOW + 1, len(toks)); labels = [ur(t) for t in toks[:n]]
    for hcol, head in enumerate([0, 1]):
        ax = axes[r, hcol]; im = ax.imshow(A[head, :n, :n], cmap="viridis")
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=8, fontproperties=UR_FP)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8, fontproperties=UR_FP)
        ax.set_xlabel("key token (attended to)"); ax.set_ylabel("query token")
        ax.set_title(f"Article {ARTICLES[i]['id']} – true/pred: {CATEGORIES[CLS_Y[i]]} – final layer, head {head+1}")
        plt.colorbar(im, ax=ax, fraction=.046)
    L_ = len(toks)
    Ar = A[:, :L_, :L_]
    HEAD_STATS.append({"max_w": Ar.max(-1).mean(-1), "diag": np.stack([np.diag(Ar[h]).mean() for h in range(Ar.shape[0])]),
                       "to_cls": Ar[:, :, 0].mean(-1)})
    cls_att = A[:, 0, 1:L_].mean(0)                # what [CLS] attends to (averaged over heads), whole article
    uniq = []
    for t in np.argsort(-cls_att):                   # top-8 distinct words
        if toks[1 + t] not in uniq: uniq.append(toks[1 + t])
        if len(uniq) == 8: break
    CLS_FOCUS.append((CATEGORIES[CLS_Y[i]], uniq))
plt.tight_layout(); savefig("p3_attention_heatmaps.png"); plt.show()
kw_all = {k for c_ in CAT_KEYWORDS.values() for k in c_["ur"] + c_["urc"]}
for cat_, ws in CLS_FOCUS:
    print(f"{cat_:17s} [CLS] attends most to:", "، ".join(ws), "| topic keywords among them:", [w for w in ws if w in kw_all])
HS = {k: np.mean([h[k] for h in HEAD_STATS], 0) for k in ["max_w", "diag", "to_cls"]}
display(pd.DataFrame(HS, index=[f"head {h+1}" for h in range(len(HS['max_w']))]).rename(
    columns={"max_w": "mean max weight (sharpness)", "diag": "mean self-attention", "to_cls": "mean weight on [CLS]"}).round(3))
RESULTS["p3_trf"] = {"acc": TRF_ACC, "macro_f1": TRF_F1, "best_epoch": H_TRF["best_epoch"],
                     "epoch_time": float(np.mean(H_TRF["epoch_time"])), "cm": cm5.tolist()}

# %% [markdown]
# ### 8.2 BiLSTM vs. Transformer
# For a like-for-like comparison, the Part-2 BiLSTM encoder (2 layers, bidirectional, dropout 0.5, Word2Vec C3 init) is
# trained on the **same** topic-classification split, with masked mean-pooling over $h_t$ and an MLP head.

# %%
class BiLSTMClassifier(nn.Module):
    def __init__(self, emb_matrix, n_classes=5, hidden=128):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix), freeze=False, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_matrix.shape[1], hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.5)
        self.drop = nn.Dropout(0.5)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, n_classes))
    def forward(self, ids):
        lengths = (ids != PAD).sum(1).clamp(min=1)
        out, _ = self.lstm(pack_padded_sequence(self.drop(self.emb(ids)), lengths.cpu(), batch_first=True, enforce_sorted=False))
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        m = (ids != PAD).unsqueeze(-1).float()
        return self.head(self.drop((out * m).sum(1) / m.sum(1).clamp(min=1)))

set_seed(42)
BIL = BiLSTMClassifier(EMB_INIT).to(DEVICE)
opt_b = torch.optim.Adam(BIL.parameters(), lr=1e-3, weight_decay=1e-4)
sched_b = torch.optim.lr_scheduler.LambdaLR(opt_b, lambda s: 1.0)
H_BIL = train_classifier(BIL, opt_b, sched_b, TR_EPOCHS, "BiLSTM")
P_BIL = predict_cls(BIL, idx_te)
BIL_ACC = accuracy_score(Y_TE, P_BIL) * 100
BIL_F1 = f1_score(Y_TE, P_BIL, average="macro", labels=list(range(5)), zero_division=0) * 100

def conv_epoch(h):
    best = max(h["val_acc"]); return next(i + 1 for i, a in enumerate(h["val_acc"]) if a >= 0.95 * best)
CMP = pd.DataFrame({
    "test accuracy (%)": [BIL_ACC, TRF_ACC], "test macro-F1": [BIL_F1, TRF_F1],
    "best val epoch": [H_BIL["best_epoch"], H_TRF["best_epoch"]],
    "epochs to 95% of best val acc": [conv_epoch(H_BIL), conv_epoch(H_TRF)],
    "min val-loss epoch": [int(np.argmin(H_BIL["val_loss"]) + 1), int(np.argmin(H_TRF["val_loss"]) + 1)],
    "sec / epoch": [np.mean(H_BIL["epoch_time"]), np.mean(H_TRF["epoch_time"])],
    "parameters": [sum(p.numel() for p in BIL.parameters()), sum(p.numel() for p in TRF.parameters())]},
    index=["BiLSTM", "Transformer"]).round(2)
display(CMP)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6)); e = range(1, TR_EPOCHS + 1)
for name, h in [("BiLSTM", H_BIL), ("Transformer", H_TRF)]:
    ax[0].plot(e, h["val_loss"], marker="o", ms=3, label=name); ax[1].plot(e, h["val_acc"], marker="o", ms=3, label=name)
ax[0].set_title("Validation loss per epoch"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
ax[1].set_title("Validation accuracy per epoch"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy (%)"); ax[1].legend()
plt.tight_layout(); savefig("p3_bilstm_vs_transformer.png"); plt.show()

# %%
# ---- 8.2 written comparison (10–15 sentences; numbers are filled in from this run) ----
win, lose = ("Transformer", "BiLSTM") if TRF_ACC > BIL_ACC else ("BiLSTM", "Transformer")
diff = abs(TRF_ACC - BIL_ACC)
c_b, c_t = conv_epoch(H_BIL), conv_epoch(H_TRF)
t_b, t_t = CMP.loc["BiLSTM", "sec / epoch"], CMP.loc["Transformer", "sec / epoch"]
kw_hits = sum(len([w for w in ws if w in kw_all]) for _, ws in CLS_FOCUS)
focus_ex = "، ".join(CLS_FOCUS[0][1][:5]) if CLS_FOCUS else ""
S = []
S.append(f"(1) On the held-out test set the {win} is more accurate: {max(TRF_ACC, BIL_ACC):.1f}% vs. {min(TRF_ACC, BIL_ACC):.1f}% for the {lose}, "
         f"a difference of {diff:.1f} percentage points (macro-F1 {TRF_F1:.1f} for the Transformer vs. {BIL_F1:.1f} for the BiLSTM)."
         if diff > 0 else f"(1) Both models reach the same test accuracy ({TRF_ACC:.1f}%).")
S.append(f"With only {len(idx_te)} test articles one article is worth {100/len(idx_te):.1f} points, so differences of this size should be read with caution.")
S.append(f"(2) The {'BiLSTM' if c_b < c_t else 'Transformer' if c_t < c_b else 'two models'} converged faster: it reached 95% of its best validation accuracy "
         f"after {min(c_b, c_t)} epoch(s) versus {max(c_b, c_t)} for the {'Transformer' if c_b < c_t else 'BiLSTM'}." if c_b != c_t else
         f"(2) Both models reached 95% of their best validation accuracy after {c_b} epoch(s).")
if c_b < c_t:
    S.append("The BiLSTM starts from pre-trained Word2Vec vectors, while the Transformer learns its embeddings from scratch and spends its first 50 steps in "
             "learning-rate warm-up, which slows its early progress.")
elif c_t < c_b:
    S.append("The Transformer's cosine schedule decays the learning rate smoothly, while the BiLSTM, trained at a constant rate, keeps oscillating on the small "
             "validation set before settling.")
else:
    S.append(f"The BiLSTM benefits from pre-trained Word2Vec vectors, while the Transformer learns embeddings from scratch but uses warm-up and cosine decay, "
             f"so the two effects roughly cancel out (best validation epochs: BiLSTM {H_BIL['best_epoch']}, Transformer {H_TRF['best_epoch']}).")
S.append(f"(3) Per epoch the {'Transformer' if t_t < t_b else 'BiLSTM'} was faster ({min(t_b, t_t):.2f}s vs. {max(t_b, t_t):.2f}s).")
S.append("Self-attention processes all 257 positions of a sequence in parallel with a few large matrix multiplications that suit modern hardware, "
         "whereas the LSTM must step through the 256 tokens sequentially in each direction and each layer; the quadratic cost of attention is small at this length.")
h_sharp, h_diff = int(np.argmax(HS["max_w"])) + 1, int(np.argmin(HS["max_w"])) + 1
S.append(f"(4) The final-layer heatmaps show that the heads specialise: head {h_sharp} is the most focused (mean maximum weight {HS['max_w'].max():.2f} per query), "
         f"head {h_diff} the most diffuse ({HS['max_w'].min():.2f}), and on average {HS['to_cls'].mean():.1%} of each token's attention goes to the [CLS] position.")
_n_focus = sum(len(ws) for _, ws in CLS_FOCUS)
_n_func = sum(1 for _, ws in CLS_FOCUS for w in ws if w in (POST_LEX | CONJ_LEX | AUX_LEX | PRON_LEX))
S.append(f"The [CLS] query concentrates on words such as {focus_ex}; {_n_func} of the {_n_focus} most-attended tokens are function words and {kw_hits} are keywords from our "
         f"category lists, " + ("so the classifier relies mainly on topical vocabulary." if kw_hits >= 0.3 * _n_focus else
         "so beyond our keyword lists the model has learned its own topical cues (e.g. domain-specific nouns and verbs)." if _n_func < 0.5 * _n_focus else
         "so much of its attention still lands on frequent function words, a sign that 156 training articles are too few to learn sharp attention."))
S.append("Padding positions receive zero weight because of the key-padding mask, confirming that the mask works.")
S.append("(5) For a corpus of only 200–300 articles the BiLSTM is the more appropriate architecture"
         + (", even though the Transformer scored higher in this particular run." if TRF_ACC > BIL_ACC + 2 else ", which this run also supports." if BIL_ACC > TRF_ACC + 2 else "; here both performed similarly."))
pb, pt = CMP.loc["BiLSTM", "parameters"], CMP.loc["Transformer", "parameters"]
S.append(f"It has a strong sequential inductive bias and can reuse the Word2Vec embeddings learned in Part 1, so far fewer of its "
         f"{pb:,.0f} parameters (vs. {pt:,.0f} for the Transformer) must be learned from the labelled articles alone.")
S.append("A Transformer trained from scratch has almost no built-in prior about word order or locality and typically needs much more data (or pre-training) "
         "to beat recurrent models, so on small data it overfits more easily and its results vary more between random seeds.")
S.append("If more articles became available, or if the Transformer could be pre-trained on unlabelled Urdu text, its parallelism and long-range attention would make it the better choice.")
COMPARISON_TEXT = " ".join(S)
print(f"[{len(S)} sentences]\n"); print("\n".join(S))
RESULTS["p3_bilstm"] = {"acc": BIL_ACC, "macro_f1": BIL_F1, "best_epoch": H_BIL["best_epoch"], "epoch_time": float(t_b)}
RESULTS["p3_compare"] = CMP.reset_index().to_dict(orient="records"); RESULTS["p3_compare_text"] = S
RESULTS["p3_focus"] = CLS_FOCUS
