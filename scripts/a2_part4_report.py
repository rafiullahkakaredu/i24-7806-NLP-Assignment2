# %% [markdown]
# # Report & submission package
# The cell below (1) saves every number of this run to `results.json`, (2) builds **`report.pdf`** (Times New Roman 12 pt,
# 1.5 line spacing, 2–3 pages, sections Overview / Part 1 / Part 2 / Part 3 / Conclusion) directly from those numbers,
# and (3) zips the submission folder. Standalone use: `python scripts/a2_part4_report.py results.json report.pdf`.

# %%
import os, sys, json, subprocess, shutil, glob
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm as _cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def _times_fonts():
    """Use real Times New Roman if available (tries to install it on Colab), else the PDF core Times font."""
    cands = {"r": ["/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
                   "C:/Windows/Fonts/times.ttf", "/Library/Fonts/Times New Roman.ttf", "/System/Library/Fonts/Supplemental/Times New Roman.ttf"],
             "b": ["/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf", "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
                   "C:/Windows/Fonts/timesbd.ttf", "/Library/Fonts/Times New Roman Bold.ttf", "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"]}
    def pick():
        r_ = next((p for p in cands["r"] if os.path.exists(p)), None); b_ = next((p for p in cands["b"] if os.path.exists(p)), None)
        return r_, b_
    r_, b_ = pick()
    if not r_ and "google.colab" in sys.modules:
        subprocess.run("echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections && "
                       "timeout 180 apt-get -qq install -y ttf-mscorefonts-installer > /dev/null 2>&1", shell=True)
        r_, b_ = pick()
    if r_ and b_:
        pdfmetrics.registerFont(TTFont("TNR", r_)); pdfmetrics.registerFont(TTFont("TNR-Bold", b_))
        return "TNR", "TNR-Bold", "Times New Roman (TrueType)"
    return "Times-Roman", "Times-Bold", "Times (PDF core font, metric-equivalent to Times New Roman)"

def build_report(R, out_pdf, fig_dir=None, with_figure=True, small=9.5):
    FR, FB, font_desc = _times_fonts()
    body = ParagraphStyle("b", fontName=FR, fontSize=12, leading=18, alignment=4, spaceAfter=3)      # 12 pt, 1.5 spacing
    h1 = ParagraphStyle("h1", fontName=FB, fontSize=13, leading=18, spaceBefore=5, spaceAfter=2)
    title = ParagraphStyle("t", fontName=FB, fontSize=15, leading=19, alignment=1)
    sub = ParagraphStyle("s", fontName=FR, fontSize=12, leading=16, alignment=1)
    cap = ParagraphStyle("c", fontName=FR, fontSize=10, leading=12, alignment=1)
    def tbl(rows, widths=None):
        t = Table(rows, colWidths=widths, hAlign="CENTER")
        t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), FR, small), ("FONT", (0, 0), (-1, 0), FB, small),
                               ("LINEABOVE", (0, 0), (-1, 0), .8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), .5, colors.black),
                               ("LINEBELOW", (0, -1), (-1, -1), .8, colors.black), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                               ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
        return t
    f = lambda x, d=2: "–" if x is None else f"{x:.{d}f}"
    P = []; A = P.append
    A(Paragraph("DS 5007 Natural Language Processing — Assignment 2 Report", title))
    A(Paragraph(f"{R['student']} &nbsp;|&nbsp; {R['roll']} &nbsp;|&nbsp; GitHub: {R['github']}", sub)); A(Spacer(1, 4))

    c = R["corpus"]; cats = R["categories"]
    A(Paragraph("1. Overview", h1))
    A(Paragraph(
        f"This assignment extends the Assignment-1 BBC Urdu pipeline with neural models written from scratch in PyTorch (no pre-trained "
        f"models, Gensim, HuggingFace or built-in Transformer modules). The corpus contains {c['articles']} articles and {c['tokens']:,} word "
        f"tokens from <i>cleaned.txt</i>. Articles were labelled with five topics by keyword matching (" +
        ", ".join(f"{k} {v}" for k, v in cats.items()) + "). We build count-based and predictive word embeddings (Part 1), "
        "a BiLSTM-CRF for POS tagging and NER on 500 rule-annotated sentences (Part 2) and a Pre-LN Transformer encoder for topic "
        "classification (Part 3).", body))

    A(Paragraph("2. Part 1 Results — Word Embeddings", h1))
    an = R["p1_analogy"]; w = R["p1_w2v"]
    A(Paragraph(
        f"A term–document matrix over the 10,000 most frequent tokens (+&lt;UNK&gt;) was weighted with TF·log(N/(1+df)) "
        f"(shape {R['p1_tfidf']['shape'][0]}×{R['p1_tfidf']['shape'][1]}, vocabulary coverage {R['p1_tfidf']['coverage']:.1%}); "
        f"the top-10 discriminative words per topic are listed in the notebook. A ±5 window co-occurrence matrix gave a PPMI matrix with "
        f"{R['p1_ppmi']['nnz']:,} non-zero cells, visualised with t-SNE. Skip-gram with negative sampling (separate V and U matrices, "
        f"K=10 noise words from f(w)<super>3/4</super>, d=100, Adam 10<super>-3</super>, batch 1024) reduced the loss from "
        f"{w['epoch_loss'][0]:.3f} to {w['epoch_loss'][-1]:.3f} over {len(w['epoch_loss'])} epochs. {an['correct10']}/10 reported analogies "
        f"were solved in the top-3 ({an['pool_correct']}/{an['pool_n']} of the full pool).", body))
    rows = [["Condition", "dim", "vocab", "MRR (20 pairs)", "hits@10"]]
    for r in R["p1_conditions"]:
        rows.append([r["condition"], r["dim"], f"{r['vocab']:,}", f"{r['MRR']:.3f}", r["hits@10"]])
    A(tbl(rows)); A(Paragraph("Table 1. Four-condition embedding comparison (MRR over 20 manually labelled word pairs).", cap))
    A(Paragraph(R["p1_cond_discussion"], body))

    A(Paragraph("3. Part 2 Results — POS Tagging and NER", h1))
    d = R["p2_data"]; pos = R["p2_pos"]; ner = R["p2_ner"]; lx = R["p2_lexicon"]; gz = R["p2_gazetteer"]
    A(Paragraph(
        f"500 sentences ({', '.join(f'{k} {v}' for k, v in d['topics'].items())}) were annotated by a rule-based tagger (Assignment-1 "
        f"stemmer/lemmatiser + lexicons of {lx['NOUN']} nouns, {lx['VERB']} verb forms, {lx['ADJ']} adjectives, {lx['ADV']} adverbs; "
        f"12 tags incl. AUX) and a gazetteer ({gz['PER (Pakistani)']} Pakistani persons, {gz['LOC']} locations, {gz['ORG']} organisations) in BIO "
        f"format, then split {d['split'][0]}/{d['split'][1]}/{d['split'][2]} stratified by topic. The 2-layer BiLSTM (forward and backward states concatenated, dropout 0.5, "
        f"packed sequences, padding masked) was trained with Adam and early stopping (patience 5). POS test accuracy is "
        f"<b>{pos['acc']:.2f}%</b> with macro-F1 <b>{pos['macro_f1']:.2f}</b> ({pos['mode']} embeddings); the most confused pairs were "
        f"{', '.join(pos['confused'])}.", body))
    md = R["p2_modes"]
    rows = [["Embeddings", "POS val macro-F1", "POS test acc.", "POS test macro-F1", "NER val F1"],
            ["frozen", f(md["POS val macro-F1"]["frozen"]), f(pos["frozen"]["test accuracy"]), f(pos["frozen"]["test macro-F1"]), f(md["NER val F1 (conlleval)"]["frozen"])],
            ["fine-tuned", f(md["POS val macro-F1"]["fine-tuned"]), f(pos["finetuned"]["test accuracy"]), f(pos["finetuned"]["test macro-F1"]), f(md["NER val F1 (conlleval)"]["fine-tuned"])]]
    A(tbl(rows)); A(Paragraph("Table 2. Frozen vs. fine-tuned Word2Vec (C3) initialisation.", cap))
    rows = [["Entity", "P (CRF)", "R (CRF)", "F1 (CRF)", "F1 (softmax)", "support"]]
    for t in ["PER", "LOC", "ORG", "MISC", "overall"]:
        a_, b_ = ner["crf"].get(t, {}), ner["softmax"].get(t, {})
        rows.append([t, f(a_.get("precision")), f(a_.get("recall")), f(a_.get("f1")), f(b_.get("f1")), a_.get("support", 0)])
    A(tbl(rows)); A(Paragraph("Table 3. NER test results (conlleval, entity level): BiLSTM-CRF with Viterbi vs. softmax decoding.", cap))
    rows = [["Run", "POS test macro-F1", "NER test F1"]]
    for k, v in R["p2_ablation"].items():
        rows.append([k.replace(" (BiLSTM, dropout, W2V init, CRF for NER)", ""), f(v[1]), f(v[3])])
    A(tbl(rows)); A(Paragraph("Table 4. Ablations A1–A4 (one change at a time, same split).", cap))
    A(Paragraph(" ".join(t.split(" – ")[0] + ": " + t.split(" – ")[-1] for t in R["p2_ablation_text"]), body))
    et = ner.get("err_types", {})
    A(Paragraph(f"Error analysis on the test set ({ner['n_fp']} false positives, {ner['n_fn']} false negatives; 5 of each are explained in the "
                "notebook) groups the errors as: " + (", ".join(f"{k} ({v})" for k, v in et.items()) or "none") +
                "." + (" Context-generalisation false positives are often plausible entities absent from the silver gazetteer, i.e. the model "
                       "generalises beyond its labels." if any("context" in k for k in et) else ""), body))

    A(Paragraph("4. Part 3 Results — Transformer Encoder", h1))
    t = R["p3_trf"]; b = R["p3_bilstm"]
    A(Paragraph(
        f"Articles were truncated/padded to 256 tokens and split {R['p3_data']['split'][0]}/{R['p3_data']['split'][1]}/{R['p3_data']['split'][2]} "
        f"(stratified). The encoder uses separate scaled dot-product attention, 4-head self-attention (d_model=128, d_k=d_v=32, per-head "
        f"projections), FFN (512, ReLU), a fixed sinusoidal positional buffer, 4 Pre-LN blocks and a learned [CLS] token with a 128→64→5 MLP, "
        f"trained with AdamW (5·10<super>-4</super>, wd 0.01) and cosine schedule with 50 warm-up steps for 20 epochs. "
        f"Test accuracy is <b>{t['acc']:.2f}%</b> and macro-F1 <b>{t['macro_f1']:.2f}</b>; a BiLSTM classifier on the same split reaches "
        f"{b['acc']:.2f}% / {b['macro_f1']:.2f}.", body))
    rows = [["Model", "test acc. (%)", "macro-F1", "best val epoch", "epochs to 95%", "s / epoch", "params"]]
    for r in R["p3_compare"]:
        rows.append([r["index"], f(r["test accuracy (%)"]), f(r["test macro-F1"]), int(r["best val epoch"]),
                     int(r["epochs to 95% of best val acc"]), f(r["sec / epoch"]), f"{int(r['parameters']):,}"])
    A(tbl(rows)); A(Paragraph("Table 5. BiLSTM vs. Transformer on 5-class topic classification.", cap))
    txt = R["p3_compare_text"]
    A(Paragraph(" ".join(s for s in txt if s.startswith("(")), body))
    if with_figure and fig_dir and os.path.exists(os.path.join(fig_dir, "p3_transformer_curves.png")):
        A(KeepTogether([Image(os.path.join(fig_dir, "p3_transformer_curves.png"), width=15.5 * _cm, height=4.9 * _cm),
                        Paragraph("Figure 1. Transformer training/validation loss and accuracy per epoch.", cap)]))

    A(Paragraph("5. Conclusion", h1))
    best_c = max(R["p1_conditions"], key=lambda r: r["MRR"])["condition"]
    A(Paragraph(
        f"All three parts were implemented from scratch and run end-to-end. Skip-gram embeddings trained on cleaned text ({best_c} best by MRR) "
        f"capture topical and morphological regularities and transfer usefully to sequence labelling. The BiLSTM-CRF reaches "
        f"{ner['crf']['overall']['f1']:.1f} entity F1 and {pos['acc']:.1f}% POS accuracy on silver labels, with bidirectionality, dropout and "
        f"pre-trained initialisation evaluated by ablation. On the small topic corpus the Transformer obtains {t['acc']:.1f}% accuracy; with only "
        f"a few hundred articles a recurrent model with pre-trained embeddings remains the safer choice. Main limitations are the silver "
        f"(rule-generated) labels and the small corpus; manual correction of annotations and more articles would be the next steps.", body))

    def _build(story):
        doc = SimpleDocTemplate(out_pdf, pagesize=A4, leftMargin=2.2 * _cm, rightMargin=2.2 * _cm, topMargin=1.8 * _cm, bottomMargin=1.8 * _cm,
                                title="Assignment 2 Report", author=R["student"])
        doc.build(story)
    _build(P)
    from pypdf import PdfReader
    n = len(PdfReader(out_pdf).pages)
    if n > 3 and with_figure:
        return build_report(R, out_pdf, fig_dir, with_figure=False, small=small)
    if n > 3 and small > 8.5:
        return build_report(R, out_pdf, fig_dir, with_figure=False, small=small - 0.5)
    print(f"report.pdf written: {n} page(s), font = {font_desc}")
    return n

def _jsonable(o):
    if isinstance(o, dict): return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonable(v) for v in o]
    if hasattr(o, "item"): return o.item()
    return o

if "RESULTS" in globals():                       # running inside the notebook / run_all.py
    try:
        import pypdf  # noqa
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pypdf"])
    json.dump(_jsonable(RESULTS), open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    build_report(_jsonable(RESULTS), os.path.join(OUT_DIR, "report.pdf"), FIG_DIR)
    zip_path = shutil.make_archive(os.path.join(os.path.dirname(os.path.abspath(OUT_DIR)), SUB_NAME), "zip",
                                   os.path.dirname(os.path.abspath(OUT_DIR)), os.path.basename(OUT_DIR))
    print("\nSubmission folder:", OUT_DIR)
    for root, _, fs in sorted(os.walk(OUT_DIR)):
        for f_ in sorted(fs):
            p_ = os.path.join(root, f_); print(f"  {os.path.relpath(p_, OUT_DIR):45s} {os.path.getsize(p_)/1e6:8.2f} MB")
    print("Zip:", zip_path, f"({os.path.getsize(zip_path)/1e6:.1f} MB)")
    print("\nNEXT: File → Download → .ipynb, rename it to", f"{SUB_NAME}.ipynb", "and add it to the zip / repo root.")
    if IN_COLAB and not FAST:
        try:
            from google.colab import files as _files; _files.download(zip_path)
        except Exception as e:
            print("Download it from the Files panel:", e)
elif __name__ == "__main__" and len(sys.argv) >= 3:
    build_report(json.load(open(sys.argv[1], encoding="utf-8")), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
