# %% [markdown]
# # Part 2 — Sequence Labeling: POS Tagging & NER (25 marks)
# ## 3. Dataset preparation
# **POS tag set (12 tags).** The assignment lists 11 tags (`NOUN VERB ADJ ADV PRON DET CONJ POST NUM PUNC UNK`) but asks
# for 12; the 12th tag used here is **`AUX`** (auxiliary / copula: ہے، ہیں، تھا، گا …), which is extremely frequent in Urdu
# and behaves very differently from main verbs.
#
# **Rule-based tagger.** (1) punctuation/number/Latin rules → (2) closed-class lexicons (POST, CONJ, AUX, PRON, DET, NUM)
# → (3) open-class lexicons with ≥ 200 entries each (NOUN, VERB, ADJ, ADV) → (4) the same lookup on the **stem / lemma**
# produced by the Assignment-1 stemmer/lemmatiser → (5) morphological suffix rules (‑نا/‑تا verb forms, nisba ‑ی
# adjectives, tanween ‑اً adverbs) → (6) context rules (demonstrative + noun ⇒ DET, unknown before a light verb ⇒ NOUN) →
# default NOUN.

# %%
POS_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "CONJ", "POST", "NUM", "PUNC", "AUX", "UNK"]
def _L(s): return {normalize(w) for w in s.split()}

POST_LEX = _L("کا کی کے میں نے کو سے پر تک لیے لئے ساتھ بعد بغیر سمیت طرف متعلق بارے خلاف دوران ذریعے ذریعہ مطابق باوجود علاوہ "
              "قریب اندر باہر نیچے اوپر پاس سامنے پیچھے بجائے تحت درمیان بیچ جانب والا والی والے والوں برابر ہمراہ عوض سوا بابت "
              "بمطابق بدولت سبب وجہسے تلے بیرون اردگرد گرد مقابلے بدلے")
CONJ_LEX = _L("اور یا لیکن مگر بلکہ کہ اگر تو جبکہ جب تب چونکہ کیونکہ لہذا لہٰذا تاکہ حالانکہ البتہ نیز ورنہ چاہے یعنی اگرچہ "
              "چنانچہ کہ جیسا جیساکہ بشرطیکہ وگرنہ اوراگر خواہ")
AUX_LEX = _L("ہے ہیں تھا تھی تھے تھیں ہوں گا گی گے رہا رہی رہے رہیں چکا چکی چکے چکیں سکتا سکتی سکتے سکا سکی سکے ہوگا ہوگی ہوگے "
             "ہونگے ہوں گے")
PRON_LEX = _L("میں ہم تم آپ وہ یہ اس ان اسے انہیں انھیں انہوں انھوں اسی انہی انھی مجھے ہمیں تمہیں مجھ ہماری ہمارا ہمارے میرا "
              "میری میرے تمہارا تمہاری تمہارے اپنا اپنی اپنے جو جس جن جنہوں جنہیں جسے کون کس کسی کوئی کچھ سب خود جنھوں "
              "جنھیں کسے انکا ان کا اسکا انھیں آپکو یہی وہی")
DET_LEX = _L("ہر کئی چند تمام سبھی ایسے ایسی ایسا ویسا ویسے دیگر بعض متعدد مختلف ہرایک فلاں دونوں تینوں")
DEMONSTRATIVES = _L("یہ وہ اس ان اسی انہی یہی وہی")
NUM_LEX = _L("ایک دو تین چار پانچ چھ سات آٹھ نو دس گیارہ بارہ تیرہ چودہ پندرہ سولہ سترہ اٹھارہ انیس بیس پچیس تیس چالیس پچاس "
             "ساٹھ ستر اسی نوے سو ہزار لاکھ کروڑ ارب کھرب پہلا پہلی پہلے دوسرا دوسری دوسرے تیسرا تیسری تیسرے چوتھا چوتھی "
             "پانچواں آدھا آدھی نصف درجن درجنوں سینکڑوں ہزاروں لاکھوں کروڑوں اربوں")
PRIORITY_NUM_EXCEPT = _L("پہلے")      # پہلے is usually 'before/earlier' (ADV), not 'first'
NUM_LEX -= PRIORITY_NUM_EXCEPT
PRON_LEX -= _L("میں")                 # in news text میں is overwhelmingly the postposition 'in'

# ---------- open classes (>= 200 entries each) ----------
VERB_ROOTS = ("کر ہو جا آ دے لے دیکھ سن بول لکھ پڑھ بیٹھ اٹھ چل رکھ مل بن بنا سمجھ سوچ چاہ لگ کھا پی رو ہنس بھیج بچ بچا پہنچ "
              "نکل نکال ڈال کھول توڑ جوڑ مار بتا دکھا سکھا سیکھ جیت ہار کھیل گر گرا بڑھ بڑھا گھٹ چھوڑ پکڑ روک رک کاٹ بیچ خرید "
              "اٹھا لا پا دوڑ بھاگ ٹھہر مان پہچان لوٹ بدل کھو ڈھونڈ چن سنا پوچھ بلا بھر کما گزر گزار جل جلا تھام نبھا چلا اتر "
              "اتار چڑھ لڑ ملا بانٹ ٹوٹ پھیل پھیلا سنبھال ڈر ڈرا کھینچ دبا ہٹ ہٹا کٹ جاگ جگا سمجھا منا گنوا اپنا سنوار بکھر "
              "دہرا ٹھکرا نمٹ نمٹا الجھ سلجھا دھمکا اکسا بٹھا بلوا").split()
VERB_IRREGULAR = _L("کیا کیے کئے کیں کرو کریں کرے کرکے ہوا ہوئی ہوئے ہوئیں ہوکر گیا گئی گئے گئیں جائے جائیں جاؤ دیا دیے دئیے دیں "
                    "دیجیے لیا لیے لیں لیجیے کہا کہی کہے کہنا کہنے کہتے کہتا کہتی کہیں آیا آئی آئے آئیں آؤ")
def verb_forms(r):
    forms = {r + s for s in ["نا", "نے", "نی", "تا", "تی", "تے", "تیں", "یں", "کر", "و"]}
    if r[-1] in "اوآ":
        forms |= {r + s for s in ["یا", "ئی", "ئے", "ئیں", "ئیے", "ئے گا"]}
    else:
        forms |= {r + s for s in ["ا", "ی", "ے", "یں"]}
    return forms
VERB_LEX = set(VERB_IRREGULAR)
for r_ in VERB_ROOTS:
    VERB_LEX |= {normalize(f) for f in verb_forms(r_) if " " not in f}
VERB_LEX |= {normalize(r_) for r_ in VERB_ROOTS if len(r_) > 1}
VERB_ROOT_SET = {normalize(r_) for r_ in VERB_ROOTS}

_ADJ_A = ("بڑا چھوٹا اچھا برا نیا پرانا لمبا اونچا گہرا ٹھنڈا پورا گندا سچا جھوٹا بوڑھا کھلا بھرا پچھلا اگلا ہلکا سستا مہنگا "
          "کالا نیلا پیلا اکیلا سیدھا ٹیڑھا موٹا پتلا گیلا سوکھا میٹھا کڑوا بھوکا پیاسا تیسرا").split()
ADJ_LEX = set()
for a_ in _ADJ_A:
    ADJ_LEX |= {a_, a_[:-1] + "ی", a_[:-1] + "ے"}
ADJ_LEX |= {"نئی", "نئے"}
ADJ_LEX |= _L("گرم سرد مشکل آسان خوبصورت اہم ضروری سیاسی معاشی اقتصادی سماجی قومی عالمی مقامی صوبائی وفاقی سرکاری نجی "
              "فوجی عدالتی قانونی غیرقانونی آئینی تعلیمی طبی مالی تجارتی زرعی صنعتی مذہبی ثقافتی تاریخی جمہوری عوامی شدید "
              "سخت نرم کمزور مضبوط طاقتور بہتر بدتر بہترین پورے مکمل خاص عام واضح صاف نیک ایماندار کامیاب ناکام مشہور معروف "
              "سینئر جونیئر سابق موجودہ آئندہ گزشتہ آخری ابتدائی اصل حقیقی ممکن ناممکن محفوظ خطرناک زخمی ہلاک گرفتار ذمہ دار "
              "شامل ناراض خوش پریشان مایوس مصروف آزاد خالی تیز سست لازمی اضافی متاثر مخصوص مناسب غلط صحیح درست امیر غریب "
              "جوان نوجوان بیمار تازہ مسلح اعلیٰ ادنیٰ بنیادی مرکزی شمالی جنوبی مشرقی مغربی پاکستانی بھارتی امریکی چینی "
              "افغان ایرانی روسی برطانوی سعودی اسرائیلی فلسطینی اسلامی ڈیجیٹل انتخابی پارلیمانی حکومتی عسکری سفارتی دفاعی "
              "علاقائی ماحولیاتی موسمیاتی سائنسی تکنیکی ادبی نفسیاتی جسمانی ذہنی دلچسپ افسوسناک تشویشناک شاندار زبردست "
              "بھاری قیمتی اکثریتی اقلیتی مستقل عارضی طویل مختصر وسیع تنگ اندرونی بیرونی ملکی غیرملکی پیچیدہ سادہ روشن "
              "تاریک سفید لال سرخ سبز زرد تنہا خفیہ حالیہ ممکنہ یقینی سالانہ ماہانہ روزانہ ہفتہ وار متنازع متنازعہ مسلمان "
              "قریبی دوطرفہ کثیر قدیم جدید مقبول غیرمعمولی معمولی منفی مثبت براہ ابتر عظیم اضافہ شدہ مشترکہ متحدہ "
              "انسانی بین الاقوامی آن لائن صحتمند بے گناہ مجرم ہنگامی فوری خصوصی عبوری نگران")
ADV_LEX = _L("آج کل پرسوں اب ابھی تب پھر دوبارہ ہمیشہ کبھی اکثر عموماً جلد جلدی فوراً پہلے بعدازاں آجکل روزانہ تاحال ابتک "
             "آخرکار بالآخر آخر فیالحال جلدازجلد سویرے اچانک یکایک لگاتار مسلسل بارہا باربار یہاں وہاں جہاں کہاں کہیں ادھر "
             "اُدھر اِدھر جدھر کدھر نزدیک دور آگے یہیں وہیں آہستہ تیزی آسانی خاموشی فوری تقریباً یقیناً خصوصاً اصولاً قانوناً "
             "عملاً ظاہراً بظاہر باضابطہ باقاعدہ دراصل درحقیقت واقعی حقیقتاً غالباً شاید ضرور بالکل قطعاً ہرگز نہایت انتہائی "
             "بہت کافی زیادہ کم تھوڑا ذرا خاصا صرف محض فقط بس بھی ہی نہیں نہ مت جی ہاں جیسے کیسے کیوں کب کتنا کتنی کتنے "
             "اتنا اتنی اتنے جتنا جتنی جتنے یوں یونہی اکٹھے علیحدہ الگ براہراست مزید نسبتاً سراسر اتفاقاً ارادتاً اصلاً "
             "اجمالاً احتیاطاً اختصاراً ابتداءً تدریجاً ترتیباً تفصیلاً حتماً رسماً زبردستی سرعام شرعاً صراحتاً ضمناً "
             "طبعاً عمداً فرضاً فطرتاً قدرے قریباً کلیتاً لازماً مثلاً مستقلاً مطلقاً معمولاً واقعتاً وقتاً یکسر اخلاقاً "
             "اکثروبیشتر جابجا بیشتر زیادہتر تاہم بہرحال بہرصورت بلاشبہ بیشک بےشک لامحالہ ازخود خودبخود باہم بخوبی "
             "بآسانی بمشکل بدستور بروقت برملا بلاتاخیر بلامقابلہ بیک دنبدن روزبروز رفتہرفتہ دھیرے چپکے چپچاپ عجلت "
             "سرسری اسطرح یکدم فیالفور ہنوز جبھی تبھی کبھیکبھار گاہے اکثراوقات فیالوقت سردست ازسرنو مکرر پیشتر "
             "قبلازیں تادیر دیرتک تادم ہرگاہ بعینہٖ کماحقہ علیالاعلان سراسیمگی بےحد ازحد بیحد انتہائ کھلےعام چپکےسے "
             "جلدہی ابھیتک اببھی تبسے جبسے ہمہوقت بارے ویسے ایسے ہوبہو واپس ساتھساتھ اکثریت ہرروز ہرسال سالہاسال "
             "صبحسویرے دیرسے شروعسے آخرتک ہمیشہسے اوپرسے نیچےسے اندرسے باہرسے پیچھےسے آگےسے غرضیکہ القصہ بالترتیب "
             "بالخصوص بالعموم بالواسطہ بلاواسطہ بالمشافہ بالفعل بالجبر بالکلیہ بالضرور بالاتفاق بالاخر فیالواقع "
             "فیالحقیقت فیزمانہ علیحدگی ازراہ برسرعام بذریعہ بخیروعافیت")
NOUN_LEX = _L("حکومت حکومتوں وزیر وزرا وزیراعظم ملک ملکوں ممالک عوام لوگ لوگوں شہر شہروں گاؤں ہسپتال اسپتال عدالت عدالتوں فوج "
              "پولیس صحت تعلیم معیشت آبادی سال سالوں برس دن دنوں ماہ مہینے ہفتے ہفتہ رات صبح شام وقت بات باتوں کام پانی بجلی "
              "گیس تیل قیمت قیمتوں روپے روپیہ ڈالر بینک قرض بجٹ ٹیکس تجارت کاروبار مارکیٹ صنعت زراعت کسان کسانوں مزدور "
              "طلبہ طالب استاد اساتذہ سکول اسکول یونیورسٹی کالج ڈاکٹر ڈاکٹروں مریض مریضوں بیماری وائرس ویکسین دوا ادویات "
              "علاج سیلاب بارش زلزلہ موسم گرمی سردی جنگ حملہ حملے حملوں دھماکہ دھماکے ہلاکت ہلاکتوں واقعہ واقعے خبر خبروں "
              "اخبار رپورٹ بیان اعلان فیصلہ فیصلے قانون قوانین آئین انتخابات انتخاب ووٹ ووٹوں جماعت جماعتوں پارٹی رہنما "
              "رہنماؤں لیڈر صدر اسمبلی پارلیمان سینیٹ کمیٹی کمیشن ادارہ ادارے اداروں محکمہ دفتر عہدہ عہدے افسر افسران "
              "اہلکار اہلکاروں ملازم ملازمین نوکری تنخواہ معاہدہ معاہدے مذاکرات ملاقات اجلاس دورہ دورے سفر سڑک سڑکوں "
              "گاڑی گاڑیوں ٹرین جہاز گھر گھروں خاندان بچے بچوں بچہ عورت عورتوں خواتین مرد مردوں لڑکی لڑکیوں لڑکا لڑکے "
              "بیٹا بیٹی بیٹے والد والدہ ماں باپ بھائی بہن شوہر بیوی دوست زندگی موت جان دل ذہن سر ہاتھ آنکھ زبان کتاب "
              "ٹیم ٹیموں کھلاڑی کھلاڑیوں میچ کرکٹ کپتان سکور رنز وکٹ وکٹیں ٹورنامنٹ سیریز فتح شکست کامیابی ناکامی مسئلہ "
              "مسائل حل سوال سوالات جواب وجہ نتیجہ نتائج مقصد منصوبہ منصوبے پروگرام نظام ترقی امن سلامتی سکیورٹی دہشتگردی "
              "دہشتگرد احتجاج مظاہرہ مظاہرے مظاہرین تحقیقات الزام الزامات مقدمہ مقدمے جج ججوں وکیل وکلا سزا جیل ملزم ملزمان "
              "گرفتاری حراست ضمانت درخواست سماعت حکم میڈیا چینل ویڈیو تصویر انٹرنیٹ فون ٹیکنالوجی سائنس تحقیق ماہر ماہرین "
              "دنیا علاقہ علاقے علاقوں صوبہ صوبے صوبوں سرحد ضلع سمندر دریا پہاڑ زمین ماحول آلودگی جنگل درخت خوراک غذا "
              "گندم چاول آٹا غربت روزگار مہنگائی اضافہ کمی شرح فیصد تعداد رقم فنڈز امداد پیسے دولت سرمایہ برآمدات "
              "درآمدات حق حقوق آزادی انصاف مذہب مسجد مدرسہ کارروائی آپریشن شہری شہریوں پناہ گزین ریاست قوم معلومات "
              "تقریب حصہ جگہ طرح بنیاد رائے نمائندہ نمائندے پالیسی صورتحال اطلاعات ذرائع ترجمان سربراہ عدلیہ اپوزیشن "
              "ہفتوں مہینوں گھنٹے گھنٹوں منٹ تاریخ کہانی ثقافت فلم گانا موسیقی کھیل کھیلوں اولمپکس فٹبال ہاکی کوچ "
              "سٹیڈیم میدان گول کھلاڑیوں وفد وفود سفیر سفارتخانے حمایت مخالفت تنقید تشویش خطرہ خطرات نقصان فائدہ")
for name_, lex_ in [("NOUN", NOUN_LEX), ("VERB", VERB_LEX), ("ADJ", ADJ_LEX), ("ADV", ADV_LEX)]:
    lex_ -= POST_LEX | CONJ_LEX | AUX_LEX
LEX_SIZES_RAW = {"NOUN": len(NOUN_LEX), "VERB": len(VERB_LEX), "ADJ": len(ADJ_LEX), "ADV": len(ADV_LEX)}
# cleaned.txt is stemmed by the A1 pipeline, so several dictionary words collapse onto one cleaned form
# (e.g. حکومتی/ADJ and حکومت/NOUN -> حکومت). Each cleaned form gets the class whose raw words produce it most often
# in this corpus (counts from the raw<->cleaned alignment). Hand-written entries keep priority.
_ORIG = [("POST", POST_LEX), ("CONJ", CONJ_LEX), ("AUX", AUX_LEX), ("NUM", NUM_LEX), ("PRON", PRON_LEX), ("DET", DET_LEX),
         ("VERB", VERB_LEX), ("ADV", ADV_LEX), ("ADJ", ADJ_LEX), ("NOUN", NOUN_LEX)]
_ALL_ORIG = set().union(*[l for _, l in _ORIG])
_votes = collections.defaultdict(collections.Counter)
for tag_, lex_ in _ORIG:
    for w in lex_:
        c_ = cf(w)
        if c_ != w and c_ not in _ALL_ORIG:
            _votes[c_][tag_] += _pairs.get(w, {}).get(c_, 0) + 0.5      # corpus frequency (+0.5 so unseen entries still vote)
CLEAN_LEX = {c_: v.most_common(1)[0][0] for c_, v in _votes.items()}
VERB_ROOT_SET |= {cf(w) for w in list(VERB_ROOT_SET)}
DEMONSTRATIVES |= {cf(w) for w in list(DEMONSTRATIVES)}
print(f"Cleaned-form entries added: {len(CLEAN_LEX):,}; examples:", list(CLEAN_LEX.items())[:8])
print("Hand-written open-class lexicon entries (before adding cleaned forms):", LEX_SIZES_RAW)
LEX_SIZES = {"NOUN": len(NOUN_LEX), "VERB": len(VERB_LEX), "ADJ": len(ADJ_LEX), "ADV": len(ADV_LEX), "PRON": len(PRON_LEX),
             "DET": len(DET_LEX), "CONJ": len(CONJ_LEX), "POST": len(POST_LEX), "NUM": len(NUM_LEX), "AUX": len(AUX_LEX),
             "cleaned-form entries": len(CLEAN_LEX)}
print("Lexicon sizes:", LEX_SIZES)
assert min(LEX_SIZES_RAW.values()) >= 200, "each major open class needs >= 200 entries"
RESULTS["p2_lexicon"] = LEX_SIZES_RAW

# %%
GAZ_LOC_WORDS = set()          # filled by the gazetteer cell below
LIGHT_VERBS = {normalize(w) for w in "کر کیا کی کیے کرنے کرنا کرتے کرتی کرتا ہو ہوا ہوئی ہوئے ہونے ہوتا ہوتی ہوتے دیا دی دیے لیا".split()}
OPEN_ORDER = [("VERB", VERB_LEX), ("ADV", ADV_LEX), ("ADJ", ADJ_LEX), ("NOUN", NOUN_LEX)]
CLOSED_ORDER = [("POST", POST_LEX), ("CONJ", CONJ_LEX), ("AUX", AUX_LEX), ("NUM", NUM_LEX), ("PRON", PRON_LEX), ("DET", DET_LEX)]

def _lookup(w):
    for tag, lex in CLOSED_ORDER + OPEN_ORDER:
        if w in lex: return tag
    return CLEAN_LEX.get(w)

def pos_word(w):
    if is_punct(w): return "PUNC"
    if is_num(w) or re.fullmatch(r"\d+\w*", w): return "NUM"
    if is_latin(w) or not any(c.isalpha() for c in w) or len(w) == 1: return "UNK"
    t = _lookup(w)
    if t: return t
    for cand in (lemmatize(w), stem(w)):                    # Assignment-1 stemmer / lemmatiser
        for tag, lex in OPEN_ORDER:
            if cand in lex: return tag
        if cand in VERB_ROOT_SET and w[len(cand):] in {"نا", "نے", "نی", "تا", "تی", "تے", "یں", "ا", "ی", "ے", "یا", "ئی", "ئے"}:
            return "VERB"
    if w.endswith("اً") or w.endswith("اً"): return "ADV"
    if w.endswith("ی") and (w[:-1] in NOUN_LEX or w[:-1] in GAZ_LOC_WORDS): return "ADJ"      # nisba adjective
    if w.endswith("انہ") or w.endswith("ناک"): return "ADJ"
    return None

def pos_tag(tokens):
    tags = [pos_word(w) for w in tokens]
    for i, w in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if w in DEMONSTRATIVES:                        # demonstrative followed by a nominal -> determiner
            n_tag = tags[i + 1] if i + 1 < len(tokens) else None
            tags[i] = "DET" if n_tag in ("NOUN", "ADJ", "NUM", None) and nxt and not is_punct(nxt) and nxt not in POST_LEX else "PRON"
        if tags[i] is None:
            if nxt in LIGHT_VERBS or nxt in POST_LEX or (i > 0 and tags[i - 1] in ("ADJ", "DET", "NUM")):
                tags[i] = "NOUN"
            elif nxt in AUX_LEX and re.search(r"(تا|تی|تے|یا|ئی|ئے)$", w):
                tags[i] = "VERB"
            else:
                tags[i] = "NOUN"
    return tags

_demo = [cf(t) for t in tokenize("حکومت نے نئی پالیسی کا اعلان کیا اور یہ فیصلہ عوام کے لیے بہت اہم ہے۔")]
print("Rule-based POS demo:", list(zip(_demo, pos_tag(_demo))))
_cnt = collections.Counter(pos_word(w) or "None→rules" for a in ARTICLES[:50] for w in a["tokens"])
print("Lexicon/rule coverage on 50 articles:", dict(_cnt.most_common()))

# %% [markdown]
# ### NER: seed gazetteer + BIO annotation
# Gazetteer: ≥ 50 Pakistani persons (plus some foreign leaders), ≥ 50 locations, ≥ 30 organisations, and a MISC list
# (nationalities, languages, events, festivals). Annotation = greedy **longest-match** over the token sequence
# (multi-word entities get `B-`/`I-` tags).

# %%
GAZ_PER_PK = """عمران خان|نواز شریف|شہباز شریف|مریم نواز|بلاول بھٹو زرداری|آصف علی زرداری|بے نظیر بھٹو|ذوالفقار علی بھٹو|پرویز مشرف|
عاصم منیر|قمر جاوید باجوہ|راحیل شریف|ایوب خان|ضیاء الحق|محمد علی جناح|علامہ اقبال|لیاقت علی خان|فاطمہ جناح|اسحاق ڈار|
شاہ محمود قریشی|پرویز الٰہی|چوہدری شجاعت حسین|فضل الرحمان|مولانا فضل الرحمان|سراج الحق|حافظ نعیم الرحمان|شیخ رشید|فواد چوہدری|
اسد عمر|علی امین گنڈاپور|مراد علی شاہ|مریم اورنگزیب|خواجہ آصف|احسن اقبال|رانا ثنا اللہ|عطا تارڑ|محسن نقوی|انوار الحق کاکڑ|
عارف علوی|یوسف رضا گیلانی|راجہ پرویز اشرف|شاہد خاقان عباسی|جہانگیر ترین|عمر ایوب|قاضی فائز عیسیٰ|ثاقب نثار|افتخار محمد چوہدری|
عمر عطا بندیال|یحییٰ آفریدی|منصور علی شاہ|بابر اعظم|محمد رضوان|شاہین شاہ آفریدی|شاہین آفریدی|نسیم شاہ|شاداب خان|فخر زمان|
وسیم اکرم|وقار یونس|شعیب اختر|شاہد آفریدی|انضمام الحق|یونس خان|مصباح الحق|جاوید میانداد|سرفراز احمد|ارشد ندیم|ملالہ یوسفزئی|
عبدالستار ایدھی|عبدالقدیر خان|نصرت فتح علی خان|عاطف اسلم|علی ظفر|ماہرہ خان|حامد میر|طلعت حسین|ارشد شریف|بشریٰ بی بی|
علیمہ خان|حمزہ شہباز|مریم|بلاول|عمران|زرداری|گنڈاپور|اعظم نذیر تارڑ|شیری رحمان|پرویز خٹک|عثمان بزدار|چوہدری پرویز الٰہی"""
GAZ_PER_FOREIGN = "نریندر مودی|مودی|جو بائیڈن|بائیڈن|ڈونلڈ ٹرمپ|ٹرمپ|ولادیمیر پوتن|پوتن|شی جن پنگ|نیتن یاہو|زیلنسکی|اشرف غنی|حامد کرزئی|رجب طیب اردوغان|اردوغان|محمد بن سلمان|اسامہ بن لادن|ایلون مسک|کملا ہیرس|راہول گاندھی"
GAZ_LOC = """پاکستان|اسلام آباد|کراچی|لاہور|پشاور|کوئٹہ|راولپنڈی|ملتان|فیصل آباد|حیدرآباد|سیالکوٹ|گوجرانوالہ|سرگودھا|بہاولپور|سکھر|
لاڑکانہ|گوادر|ایبٹ آباد|مردان|سوات|مظفرآباد|گلگت|سکردو|چترال|ڈیرہ اسماعیل خان|بنوں|کوہاٹ|وزیرستان|شمالی وزیرستان|جنوبی وزیرستان|
باجوڑ|خیبر|پنجاب|سندھ|بلوچستان|خیبر پختونخوا|پختونخوا|گلگت بلتستان|کشمیر|آزاد کشمیر|تھر|چولستان|مری|نوشہرہ|جہلم|گجرات|شیخوپورہ|
قصور|اوکاڑہ|ساہیوال|رحیم یار خان|ڈیرہ غازی خان|خضدار|تربت|ژوب|چمن|طورخم|کرم|پاراچنار|ہنزہ|تھرپارکر|میرپور|بھارت|انڈیا|افغانستان|
ایران|چین|امریکہ|روس|برطانیہ|سعودی عرب|متحدہ عرب امارات|قطر|ترکی|ترکیہ|اسرائیل|غزہ|فلسطین|یوکرین|بنگلہ دیش|سری لنکا|نیپال|
جاپان|جرمنی|فرانس|کابل|تہران|دہلی|نئی دہلی|ممبئی|بیجنگ|ماسکو|لندن|واشنگٹن|نیویارک|دبئی|ریاض|مکہ|مدینہ|یورپ|ایشیا|افریقہ|
خلیج|بحیرہ عرب|دریائے سندھ|ہمالیہ|مقبوضہ کشمیر|لبنان|شام|عراق|مصر|کینیڈا|آسٹریلیا|نیوزی لینڈ|جنوبی افریقہ|ویسٹ انڈیز|زمبابوے"""
GAZ_ORG = """پاکستان تحریک انصاف|تحریک انصاف|پی ٹی آئی|مسلم لیگ ن|ن لیگ|پاکستان مسلم لیگ|مسلم لیگ|پیپلز پارٹی|پاکستان پیپلز پارٹی|پی پی پی|
جمعیت علمائے اسلام|جے یو آئی|جماعت اسلامی|ایم کیو ایم|متحدہ قومی موومنٹ|عوامی نیشنل پارٹی|اے این پی|تحریک لبیک|ٹی ایل پی|
تحریک طالبان پاکستان|ٹی ٹی پی|طالبان|سپریم کورٹ|ہائی کورٹ|اسلام آباد ہائی کورٹ|لاہور ہائی کورٹ|سندھ ہائی کورٹ|الیکشن کمیشن|نیب|
قومی احتساب بیورو|ایف آئی اے|آئی ایس پی آر|آئی ایس آئی|پاک فوج|پاکستانی فوج|پاک فضائیہ|پاک بحریہ|سٹیٹ بینک|اسٹیٹ بینک|پی سی بی|
پاکستان کرکٹ بورڈ|آئی سی سی|آئی ایم ایف|عالمی بینک|ورلڈ بینک|اقوام متحدہ|سلامتی کونسل|نیٹو|یورپی یونین|او آئی سی|سارک|
شنگھائی تعاون تنظیم|ایشیائی ترقیاتی بینک|عالمی ادارہ صحت|ڈبلیو ایچ او|یونیسیف|پیمرا|واپڈا|پی آئی اے|نادرا|ہائر ایجوکیشن کمیشن|
بی بی سی|قومی اسمبلی|پنجاب اسمبلی|سندھ اسمبلی|حماس|حزب اللہ|داعش|القاعدہ|ایدھی فاؤنڈیشن|پنجاب پولیس|سندھ پولیس|رینجرز|
کراچی کنگز|لاہور قلندرز|پی ایس ایل|پی ڈی ایم|ایف بی آر|اوگرا|نیپرا|پی ڈی ایم اے|این ڈی ایم اے"""
GAZ_MISC = """پاکستانی|بھارتی|امریکی|افغان|چینی|ایرانی|روسی|برطانوی|سعودی|اسرائیلی|فلسطینی|کشمیری|پنجابی|سندھی|بلوچ|پشتون|اردو|انگریزی|
پشتو|عربی|فارسی|ورلڈ کپ|ایشیا کپ|چیمپئنز ٹرافی|ٹی ٹوئنٹی ورلڈ کپ|اولمپکس|عید الفطر|عید الاضحیٰ|عید|رمضان|محرم|یوم آزادی|کورونا|
کووڈ|سی پیک|قرآن|اسلام|ہندو|مسلمان|عیسائی|سکھ"""

def _gz(s): return [tuple(tokenize(x.strip())) for x in s.replace("\n", "").split("|") if x.strip()]
def _gzc(lst): return lst + [tuple(cf(t) for t in p) for p in lst]          # + cleaned (stemmed) form of each phrase
GAZETTEER = {}
for typ, lst in [("MISC", _gzc(_gz(GAZ_MISC))), ("LOC", _gzc(_gz(GAZ_LOC))), ("PER", _gzc(_gz(GAZ_PER_PK) + _gz(GAZ_PER_FOREIGN))), ("ORG", _gzc(_gz(GAZ_ORG)))]:
    for ph in lst:
        GAZETTEER[ph] = typ                                  # later types (ORG > PER > LOC > MISC) override on exact duplicates
GAZ_MAXLEN = max(len(p) for p in GAZETTEER)
GAZ_LOC_WORDS = {p[0] for p, t in GAZETTEER.items() if t == "LOC" and len(p) == 1}
GAZ_COUNTS = {"PER (Pakistani)": len(_gz(GAZ_PER_PK)), "PER (foreign)": len(_gz(GAZ_PER_FOREIGN)), "LOC": len(_gz(GAZ_LOC)),
              "ORG": len(_gz(GAZ_ORG)), "MISC": len(_gz(GAZ_MISC))}
print("Gazetteer sizes:", GAZ_COUNTS)
assert GAZ_COUNTS["PER (Pakistani)"] >= 50 and GAZ_COUNTS["LOC"] >= 50 and GAZ_COUNTS["ORG"] >= 30
RESULTS["p2_gazetteer"] = GAZ_COUNTS
NER_TAGS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]

def ner_tag(tokens):
    tags = ["O"] * len(tokens); i = 0
    while i < len(tokens):
        for L in range(min(GAZ_MAXLEN, len(tokens) - i), 0, -1):
            typ = GAZETTEER.get(tuple(tokens[i:i + L]))
            if typ:
                tags[i] = "B-" + typ
                for j in range(i + 1, i + L): tags[j] = "I-" + typ
                i += L; break
        else:
            i += 1
    return tags

_demo = [cf(t) for t in tokenize("وزیراعظم شہباز شریف نے اسلام آباد میں آئی ایم ایف کے وفد سے ملاقات کی۔")]
print(list(zip(_demo, pos_tag(_demo), ner_tag(_demo))))

# %%
# ---- 3.1 select 500 sentences (>= 100 from each of 3 topic categories) ----
SENT_SPLIT = re.compile(r"(?<=[۔؟!?])\s+|\n+")
def sentences_of(article):
    sents = []
    for s in SENT_SPLIT.split(article["text"]):
        toks = tokenize(s)
        if 5 <= len(toks) <= 45: sents.append(toks)
    if len(sents) < 3:                                      # punctuation removed in cleaning -> fixed-size chunks
        toks = article["tokens"]
        sents = [toks[i:i + 20] for i in range(0, len(toks) - 5, 20)]
    return sents

rng = random.Random(42)
pool = collections.defaultdict(list)
for a in ARTICLES:
    for s in sentences_of(a):
        pool[a["category"]].append({"tokens": s, "topic": a["category"], "article": a["id"]})
avail = {c_: len(v) for c_, v in pool.items()}
print("Candidate sentences per category:", avail)
top3 = sorted(avail, key=avail.get, reverse=True)[:3]
quota = {top3[0]: 167, top3[1]: 167, top3[2]: 166}
selected = []
for c_ in top3:
    cand = pool[c_][:]; rng.shuffle(cand)
    has = [any(t != "O" for t in ner_tag(s["tokens"])) for s in cand]
    with_ent = [s for s, h in zip(cand, has) if h]
    without = [s for s, h in zip(cand, has) if not h]
    k_ent = min(len(with_ent), int(quota[c_] * 0.7))           # oversample entity-bearing sentences for NER
    pick = with_ent[:k_ent] + without[:quota[c_] - k_ent]
    if len(pick) < quota[c_]: pick += [s for s in with_ent[k_ent:]][:quota[c_] - len(pick)]
    selected += pick
if len(selected) < 500:                                         # top-up from other categories if a category is small
    chosen = {id(s) for s in selected}
    rest = [s for c_ in pool for s in pool[c_] if id(s) not in chosen]; rng.shuffle(rest)
    selected += rest[:500 - len(selected)]
selected = selected[:500]
for s in selected:
    s["pos"] = pos_tag(s["tokens"]); s["ner"] = ner_tag(s["tokens"])
print(f"Selected {len(selected)} sentences; per topic:", collections.Counter(s["topic"] for s in selected))

# ---- 70 / 15 / 15 split stratified by topic ----
def strat_split(items, labels, seed=42):
    lab = np.array(labels); idx = np.arange(len(items))
    strat = lab if min(collections.Counter(lab).values()) >= 3 else None
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=seed, stratify=strat)
    strat2 = lab[tmp] if strat is not None and min(collections.Counter(lab[tmp]).values()) >= 2 else None
    va, te = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=strat2)
    return [items[i] for i in tr], [items[i] for i in va], [items[i] for i in te]
S_TRAIN, S_VAL, S_TEST = strat_split(selected, [s["topic"] for s in selected])
print(f"train/val/test = {len(S_TRAIN)}/{len(S_VAL)}/{len(S_TEST)}")

def write_conll(path, sents, key):
    with open(path, "w", encoding="utf-8") as f:
        for s in sents:
            for w, t in zip(s["tokens"], s[key]): f.write(f"{w}\t{t}\n")
            f.write("\n")
for split, ss in [("train", S_TRAIN), ("val", S_VAL), ("test", S_TEST)]:
    write_conll(os.path.join(DATA_OUT, f"pos_{split}.conll"), ss, "pos")
    write_conll(os.path.join(DATA_OUT, f"ner_{split}.conll"), ss, "ner")
print("Saved data/pos_{train,val,test}.conll and data/ner_{train,val,test}.conll")

# %%
# ---- class-label distributions ----
def dist(sents, key, labels):
    c_ = collections.Counter(t for s in sents for t in s[key]); return [c_.get(l, 0) for l in labels]
POS_DIST = pd.DataFrame({sp_: dist(ss, "pos", POS_TAGS) for sp_, ss in [("train", S_TRAIN), ("val", S_VAL), ("test", S_TEST)]}, index=POS_TAGS)
NER_DIST = pd.DataFrame({sp_: dist(ss, "ner", NER_TAGS) for sp_, ss in [("train", S_TRAIN), ("val", S_VAL), ("test", S_TEST)]}, index=NER_TAGS)
TOPIC_DIST = pd.DataFrame({sp_: collections.Counter(s["topic"] for s in ss) for sp_, ss in [("train", S_TRAIN), ("val", S_VAL), ("test", S_TEST)]}).fillna(0).astype(int)
print("Topic distribution per split:"); display(TOPIC_DIST)
print("POS tag distribution:"); display(POS_DIST.T)
print("NER tag distribution:"); display(NER_DIST.T)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
POS_DIST.plot.bar(ax=ax[0]); ax[0].set_title("POS tag distribution per split"); ax[0].set_xlabel("POS tag"); ax[0].set_ylabel("tokens")
NER_DIST.drop("O").plot.bar(ax=ax[1]); ax[1].set_title("NER tag distribution per split (O omitted)"); ax[1].set_xlabel("BIO tag"); ax[1].set_ylabel("tokens")
plt.tight_layout(); savefig("p2_label_dist.png"); plt.show()
RESULTS["p2_data"] = {"n": len(selected), "split": [len(S_TRAIN), len(S_VAL), len(S_TEST)],
                      "topics": dict(collections.Counter(s["topic"] for s in selected)),
                      "pos_total": POS_DIST.sum(1).to_dict(), "ner_total": NER_DIST.sum(1).to_dict()}

# %% [markdown]
# ## 4. BiLSTM sequence labeler (+ CRF for NER)
# * embeddings initialised from Word2Vec **C3** (`embeddings_w2v.npy`), frozen or fine-tuned
# * 2-layer bidirectional LSTM; per-token representation $h_t=[\overrightarrow{h_t}\,\Vert\,\overleftarrow{h_t}]$; dropout 0.5 between layers
# * NER: CRF with learnable transition matrix (+ start/end scores), trained with the forward algorithm, decoded with **Viterbi**
# * POS: linear layer + cross-entropy
# * variable lengths: `pack_padded_sequence`; padding is masked out of every loss (`ignore_index` / CRF mask)
# * Adam (η=1e-3, weight decay 1e-4), early stopping on validation F1 with patience 5 (counted after a 15-epoch warm-up; best epoch restored)

# %%
POS2I = {t: i for i, t in enumerate(POS_TAGS)}; NER2I = {t: i for i, t in enumerate(NER_TAGS)}
# Word2Vec vectors are tiny (|x|~0.1) compared with PyTorch's N(0,1) default; rescale to unit std so that the pretrained and
# random initialisations (ablation A3) start from the same input scale – only the *directions* learned by Word2Vec differ.
EMB_INIT = (EMB_C3 / EMB_C3[2:].std()).astype(np.float32); EMB_INIT[PAD] = 0
print(f"Word2Vec C3 matrix {EMB_C3.shape}; std before={EMB_C3[2:].std():.4f} after rescaling={EMB_INIT[2:].std():.2f}")

def encode(sents, key, t2i):
    X = [[W2I.get(w, UNK) for w in s["tokens"]] for s in sents]
    Y = [[t2i[t] for t in s[key]] for s in sents]
    return X, Y

def batches(X, Y, bs=32, shuffle=False, seed=0):
    order = list(range(len(X)))
    if shuffle: random.Random(seed).shuffle(order)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]; L = max(len(X[i]) for i in idx)
        x = torch.zeros(len(idx), L, dtype=torch.long); y = torch.full((len(idx), L), -100, dtype=torch.long)
        for j, i in enumerate(idx):
            x[j, :len(X[i])] = torch.tensor(X[i]); y[j, :len(Y[i])] = torch.tensor(Y[i])
        lengths = torch.tensor([len(X[i]) for i in idx])
        yield x.to(DEVICE), y.to(DEVICE), lengths, idx

class CRF(nn.Module):
    """Linear-chain CRF. transitions[i, j] = score of moving from tag i to tag j."""
    def __init__(self, n_tags):
        super().__init__()
        self.n = n_tags
        self.transitions = nn.Parameter(torch.empty(n_tags, n_tags).uniform_(-0.1, 0.1))
        self.start = nn.Parameter(torch.zeros(n_tags)); self.end = nn.Parameter(torch.zeros(n_tags))
    def _gold_score(self, em, tags, mask):
        B, L, _ = em.shape; ar = torch.arange(B, device=em.device)
        score = self.start[tags[:, 0]] + em[ar, 0, tags[:, 0]]
        for t in range(1, L):
            s_t = self.transitions[tags[:, t - 1], tags[:, t]] + em[ar, t, tags[:, t]]
            score = score + s_t * mask[:, t]
        last = tags[ar, mask.sum(1).long() - 1]
        return score + self.end[last]
    def _log_partition(self, em, mask):                    # forward algorithm
        alpha = self.start + em[:, 0]
        for t in range(1, em.size(1)):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + em[:, t].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1).bool(), nxt, alpha)
        return torch.logsumexp(alpha + self.end, dim=1)
    def nll(self, em, tags, mask):
        tags = tags.clamp(min=0); mask = mask.float()
        return (self._log_partition(em, mask) - self._gold_score(em, tags, mask)).mean()
    @torch.no_grad()
    def viterbi(self, em, mask):
        B, L, T = em.shape
        score = self.start + em[:, 0]; history = []
        for t in range(1, L):
            s = score.unsqueeze(2) + self.transitions.unsqueeze(0)          # B x T(prev) x T(cur)
            best, idx = s.max(1)
            best = best + em[:, t]
            history.append(idx)
            score = torch.where(mask[:, t].unsqueeze(1).bool(), best, score)
        score = score + self.end
        lengths = mask.sum(1).long().tolist(); paths = []
        for b in range(B):
            last = int(score[b].argmax()); path = [last]
            for t in range(lengths[b] - 1, 0, -1):
                last = int(history[t - 1][b, last]); path.append(last)
            paths.append(path[::-1])
        return paths

class BiLSTMTagger(nn.Module):
    def __init__(self, emb_matrix, n_tags, hidden=128, layers=2, bidirectional=True, dropout=0.5,
                 use_crf=False, freeze=False, random_init=False):
        super().__init__()
        Vn, d = emb_matrix.shape
        self.emb = nn.Embedding(Vn, d, padding_idx=PAD)
        if not random_init:
            self.emb.weight.data.copy_(torch.tensor(emb_matrix))
        self.emb.weight.requires_grad = not freeze
        self.bidirectional, self.hidden = bidirectional, hidden
        self.lstm = nn.LSTM(d, hidden, num_layers=layers, bidirectional=bidirectional, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)        # dropout between the 2 LSTM layers
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * (2 if bidirectional else 1), n_tags)
        self.crf = CRF(n_tags) if use_crf else None
    def emissions(self, x, lengths):
        e = self.drop(self.emb(x))
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        if self.bidirectional:
            fwd, bwd = out[..., :self.hidden], out[..., self.hidden:]
            out = torch.cat([fwd, bwd], dim=-1)                              # h_t = [→h_t ‖ ←h_t]
        return self.fc(self.drop(out))
    def loss(self, x, y, lengths):
        em = self.emissions(x, lengths); mask = y != -100
        if self.crf is not None:
            return self.crf.nll(em, y, mask)
        return F.cross_entropy(em.reshape(-1, em.size(-1)), y.reshape(-1), ignore_index=-100)   # padding ignored
    @torch.no_grad()
    def predict(self, x, lengths):
        em = self.emissions(x, lengths)
        mask = torch.arange(x.size(1), device=x.device)[None] < lengths.to(x.device)[:, None]
        if self.crf is not None: return self.crf.viterbi(em, mask)
        am = em.argmax(-1).tolist(); return [am[b][:int(lengths[b])] for b in range(len(am))]

print(BiLSTMTagger(EMB_INIT, len(NER_TAGS), use_crf=True))

# %%
# ---- conlleval (Python port of the official conlleval.pl chunk logic) ----
def _split_tag(t):
    if t == "O": return "O", ""
    p, _, typ = t.partition("-"); return p, typ
def _end_of_chunk(pt, ct, pp, cp):
    if pt in ("E", "S"): return True
    if pt in ("B", "I") and ct in ("B", "S", "O"): return True
    return pt not in ("O", ".") and pp != cp
def _start_of_chunk(pt, ct, pp, cp):
    if ct in ("B", "S"): return True
    if (pt, ct) in {("E", "E"), ("E", "I"), ("S", "E"), ("S", "I"), ("O", "E"), ("O", "I")}: return True
    return ct not in ("O", ".") and pp != cp

def chunks(tags):
    """Return set of (start, end, type) chunks following conlleval semantics."""
    out, start, ptag, ptype = [], None, "O", ""
    for i, t in enumerate(tags + ["O"]):
        ct, cp = _split_tag(t)
        if start is not None and _end_of_chunk(ptag, ct, ptype, cp):
            out.append((start, i - 1, ptype)); start = None
        if _start_of_chunk(ptag, ct, ptype, cp):
            start = i
        ptag, ptype = ct, cp
    return out

def conlleval(gold_seqs, pred_seqs, types=("PER", "LOC", "ORG", "MISC")):
    stats = {t: [0, 0, 0] for t in types}   # correct, predicted, gold
    tok_ok = tok_n = 0
    for g, p in zip(gold_seqs, pred_seqs):
        gc, pc = set(chunks(list(g))), set(chunks(list(p)))
        for (_, _, t) in gc: stats.setdefault(t, [0, 0, 0])[2] += 1
        for (_, _, t) in pc: stats.setdefault(t, [0, 0, 0])[1] += 1
        for (_, _, t) in gc & pc: stats[t][0] += 1
        tok_ok += sum(a == b for a, b in zip(g, p)); tok_n += len(g)
    def prf(c_, p_, g_):
        P = 100 * c_ / p_ if p_ else 0.0; R = 100 * c_ / g_ if g_ else 0.0
        return P, R, (2 * P * R / (P + R) if P + R else 0.0)
    res = {t: dict(zip(["precision", "recall", "f1"], prf(*stats[t])), support=stats[t][2]) for t in stats}
    tot = [sum(stats[t][i] for t in stats) for i in range(3)]
    res["overall"] = dict(zip(["precision", "recall", "f1"], prf(*tot)), support=tot[2], accuracy=100 * tok_ok / max(tok_n, 1))
    return res

def conlleval_report(res):
    o = res["overall"]
    lines = [f"accuracy: {o['accuracy']:6.2f}%; precision: {o['precision']:6.2f}%; recall: {o['recall']:6.2f}%; FB1: {o['f1']:6.2f}"]
    for t in [k for k in res if k != "overall"]:
        r_ = res[t]; lines.append(f"{t:>17}: precision: {r_['precision']:6.2f}%; recall: {r_['recall']:6.2f}%; FB1: {r_['f1']:6.2f}  {r_['support']}")
    return "\n".join(lines)

print(conlleval_report(conlleval([["B-PER", "I-PER", "O", "B-LOC"]], [["B-PER", "I-PER", "O", "B-ORG"]])))

# %%
# ---- training loop with early stopping on validation F1 ----
def evaluate(model, X, Y, task):
    model.eval(); preds, golds, tot_loss, nb = [], [], 0.0, 0
    for x, y, lengths, idx in batches(X, Y, bs=64):
        tot_loss += model.loss(x, y, lengths).item(); nb += 1
        p = model.predict(x, lengths)
        for j, i in enumerate(idx):
            preds.append(p[j]); golds.append(Y[i])
    if task == "pos":
        gf = [t for s in golds for t in s]; pf = [t for s in preds for t in s]
        f1 = f1_score(gf, pf, average="macro", labels=sorted(set(gf)), zero_division=0) * 100
    else:
        f1 = conlleval([[NER_TAGS[t] for t in s] for s in golds], [[NER_TAGS[t] for t in s] for s in preds])["overall"]["f1"]
    return f1, tot_loss / max(nb, 1), preds

def train_tagger(task, freeze=False, bidirectional=True, dropout=0.5, use_crf=None, random_init=False,
                 max_epochs=SEQ_MAX_EPOCHS, patience=5, min_epochs=15, tag=""):
    set_seed(42)
    key, t2i, tags = ("pos", POS2I, POS_TAGS) if task == "pos" else ("ner", NER2I, NER_TAGS)
    use_crf = (task == "ner") if use_crf is None else use_crf
    Xtr, Ytr = encode(S_TRAIN, key, t2i); Xva, Yva = encode(S_VAL, key, t2i)
    model = BiLSTMTagger(EMB_INIT, len(tags), bidirectional=bidirectional, dropout=dropout, use_crf=use_crf,
                         freeze=freeze, random_init=random_init).to(DEVICE)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    hist = {"train_loss": [], "val_loss": [], "val_f1": [], "epoch_time": []}
    best_f1, best_state, wait = -1, None, 0
    for ep in range(max_epochs):
        model.train(); t0 = time.time(); tl, nb = 0.0, 0
        for x, y, lengths, _ in batches(Xtr, Ytr, bs=32, shuffle=True, seed=ep):
            loss = model.loss(x, y, lengths)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tl += loss.item(); nb += 1
        hist["epoch_time"].append(time.time() - t0)
        vf1, vloss, _ = evaluate(model, Xva, Yva, task)
        hist["train_loss"].append(tl / nb); hist["val_loss"].append(vloss); hist["val_f1"].append(vf1)
        if vf1 > best_f1:
            best_f1, wait = vf1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; hist["best_epoch"] = ep + 1
        elif ep + 1 > min_epochs:             # patience only counts after a short warm-up: NER models predict all-'O'
            wait += 1                          # (F1 = 0) for the first epochs, which would otherwise trigger a premature stop
            if wait >= patience: break
    model.load_state_dict(best_state)
    print(f"[{tag or task}] stopped after {len(hist['val_f1'])} epochs | best val F1 = {best_f1:.2f} (epoch {hist['best_epoch']})")
    return model, hist, best_f1

def plot_curves(hists, title, fname):
    fig, ax = plt.subplots(1, len(hists), figsize=(5.2 * len(hists), 3.4), squeeze=False)
    for a_, (name, h) in zip(ax[0], hists.items()):
        e = range(1, len(h["train_loss"]) + 1)
        a_.plot(e, h["train_loss"], marker="o", ms=3, label="train loss"); a_.plot(e, h["val_loss"], marker="s", ms=3, label="val loss")
        a_.axvline(h["best_epoch"], ls="--", c="grey", lw=1, label=f"best val F1 (ep {h['best_epoch']})")
        a_.set_title(f"{title}: {name}"); a_.set_xlabel("epoch"); a_.set_ylabel("loss"); a_.legend(fontsize=8)
    plt.tight_layout(); savefig(fname); plt.show()

print("Training utilities ready: Adam(lr=1e-3, wd=1e-4), batch 32, grad-clip 5, early stopping patience 5 (after 15 warm-up epochs),",
      f"max {SEQ_MAX_EPOCHS} epochs, device={DEVICE}")

# %%
# ---- frozen vs fine-tuned embeddings ----
pos_frozen, h_pos_fz, f_pos_fz = train_tagger("pos", freeze=True, tag="POS frozen")
pos_ft, h_pos_ft, f_pos_ft = train_tagger("pos", freeze=False, tag="POS fine-tuned")
ner_frozen, h_ner_fz, f_ner_fz = train_tagger("ner", freeze=True, tag="NER-CRF frozen")
ner_ft, h_ner_ft, f_ner_ft = train_tagger("ner", freeze=False, tag="NER-CRF fine-tuned")
plot_curves({"POS frozen": h_pos_fz, "POS fine-tuned": h_pos_ft}, "BiLSTM", "p2_curves_pos.png")
plot_curves({"NER-CRF frozen": h_ner_fz, "NER-CRF fine-tuned": h_ner_ft}, "BiLSTM-CRF", "p2_curves_ner.png")
MODE_DF = pd.DataFrame({"POS val macro-F1": [f_pos_fz, f_pos_ft], "NER val F1 (conlleval)": [f_ner_fz, f_ner_ft]},
                       index=["frozen", "fine-tuned"]).round(2)
display(MODE_DF)
POS_BEST, POS_MODE = (pos_ft, "fine-tuned") if f_pos_ft >= f_pos_fz else (pos_frozen, "frozen")
NER_BEST, NER_MODE = (ner_ft, "fine-tuned") if f_ner_ft >= f_ner_fz else (ner_frozen, "frozen")
torch.save({"state_dict": POS_BEST.state_dict(), "tags": POS_TAGS, "mode": POS_MODE}, os.path.join(MODEL_DIR, "bilstm_pos.pt"))
torch.save({"state_dict": NER_BEST.state_dict(), "tags": NER_TAGS, "mode": NER_MODE}, os.path.join(MODEL_DIR, "bilstm_ner.pt"))
print(f"Saved models/bilstm_pos.pt ({POS_MODE}) and models/bilstm_ner.pt ({NER_MODE})")

# %% [markdown]
# ## 5. Evaluation
# ### 5.1 POS tagging

# %%
Xte_p, Yte_p = encode(S_TEST, "pos", POS2I)
def pos_test_metrics(model):
    _, _, pr = evaluate(model, Xte_p, Yte_p, "pos")
    g = [POS_TAGS[t] for s in Yte_p for t in s]; p = [POS_TAGS[t] for s in pr for t in s]
    return accuracy_score(g, p) * 100, f1_score(g, p, average="macro", labels=sorted(set(g)), zero_division=0) * 100, g, p, pr
acc_ft, mf1_ft, g_ft, p_ft, pr_ft = pos_test_metrics(pos_ft)
acc_fz, mf1_fz, g_fz, p_fz, pr_fz = pos_test_metrics(pos_frozen)
acc_b, mf1_b, G_POS, P_POS, PR_POS = pos_test_metrics(POS_BEST)
print(f"POS test ({POS_MODE}): token accuracy = {acc_b:.2f}% | macro-F1 = {mf1_b:.2f}")

cm = confusion_matrix(G_POS, P_POS, labels=POS_TAGS)
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(np.log1p(cm), cmap="Blues")
ax.set_xticks(range(12)); ax.set_xticklabels(POS_TAGS, rotation=45); ax.set_yticks(range(12)); ax.set_yticklabels(POS_TAGS)
for i in range(12):
    for j in range(12):
        if cm[i, j]: ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if np.log1p(cm[i, j]) > np.log1p(cm.max()) * .6 else "black")
ax.set_xlabel("predicted tag"); ax.set_ylabel("gold tag"); ax.set_title(f"POS confusion matrix (test, {POS_MODE}; colour = log count)")
plt.colorbar(im, ax=ax, fraction=.046); plt.tight_layout(); savefig("p2_pos_confusion.png"); plt.show()

off = [((POS_TAGS[i], POS_TAGS[j]), cm[i, j] + cm[j, i]) for i in range(12) for j in range(i + 1, 12) if cm[i, j] + cm[j, i] > 0]
off = sorted(off, key=lambda x: -x[1])[:3]
off = [(ab, int(n)) for ab, n in off]
print("3 most confused tag pairs:", [(f"{a}↔{b}", n) for (a, b), n in off])
all_sents = [(S_TEST, PR_POS)]
_, _, pr_val = evaluate(POS_BEST, *encode(S_VAL, "pos", POS2I), "pos"); all_sents.append((S_VAL, pr_val))
CONF_EX = {}
for (a, b), n in off:
    ex = []
    for ss, prs in all_sents:
        for s, p in zip(ss, prs):
            hits = [i for i, (gt, pt) in enumerate(zip(s["pos"], p)) if {gt, POS_TAGS[pt]} == {a, b}]
            if hits:
                marked = " ".join(f"[{w}/{s['pos'][i]}→{POS_TAGS[p[i]]}]" if i in hits else w for i, w in enumerate(s["tokens"]))
                ex.append(marked)
            if len(ex) >= 2: break
        if len(ex) >= 2: break
    CONF_EX[f"{a}↔{b}"] = ex
    print(f"\n=== {a} ↔ {b} ({n} confusions) ===")
    for e_ in ex: print("  •", e_)
print("\nTypical causes: Urdu adjectives used nominally (ADJ↔NOUN), light-verb constructions where the noun/verb "
      "boundary is fuzzy (NOUN↔VERB), and words such as ہو/رہا that are AUX or VERB depending on context.")
POS_MODE_DF = pd.DataFrame({"val macro-F1": [f_pos_fz, f_pos_ft], "test accuracy": [acc_fz, acc_ft], "test macro-F1": [mf1_fz, mf1_ft]},
                           index=["frozen", "fine-tuned"]).round(2)
display(POS_MODE_DF)
RESULTS["p2_pos"] = {"mode": POS_MODE, "acc": acc_b, "macro_f1": mf1_b, "confused": [f"{a}<->{b} ({n})" for (a, b), n in off],
                     "frozen": POS_MODE_DF.loc["frozen"].to_dict(), "finetuned": POS_MODE_DF.loc["fine-tuned"].to_dict()}

# %% [markdown]
# ### 5.2 NER — conlleval, CRF vs. softmax, error analysis

# %%
Xte_n, Yte_n = encode(S_TEST, "ner", NER2I)
def ner_test(model):
    _, _, pr = evaluate(model, Xte_n, Yte_n, "ner")
    gold = [[NER_TAGS[t] for t in s] for s in Yte_n]; pred = [[NER_TAGS[t] for t in s] for s in pr]
    return conlleval(gold, pred), gold, pred
ner_soft, h_ner_soft, f_ner_soft = train_tagger("ner", freeze=(NER_MODE == "frozen"), use_crf=False, tag="NER softmax (no CRF)")
RES_CRF, G_NER, P_NER = ner_test(NER_BEST)
RES_SOFT, _, P_NER_SOFT = ner_test(ner_soft)
print("conlleval — BiLSTM-CRF (test):"); print(conlleval_report(RES_CRF))
print("\nconlleval — BiLSTM + softmax (test):"); print(conlleval_report(RES_SOFT))
with open(os.path.join(DATA_OUT, "ner_test_predictions.conll"), "w", encoding="utf-8") as f:   # conlleval.pl-compatible file
    for s, g, p in zip(S_TEST, G_NER, P_NER):
        for w, a, b in zip(s["tokens"], g, p): f.write(f"{w} {a} {b}\n")
        f.write("\n")
rows = []
for t in ["PER", "LOC", "ORG", "MISC", "overall"]:
    for name, R in [("CRF", RES_CRF), ("softmax", RES_SOFT)]:
        r_ = R.get(t, {"precision": 0, "recall": 0, "f1": 0, "support": 0})
        rows.append({"type": t, "decoder": name, "P": r_["precision"], "R": r_["recall"], "F1": r_["f1"], "support": r_["support"]})
NER_TABLE = pd.DataFrame(rows).pivot(index="type", columns="decoder", values=["P", "R", "F1"]).round(2).reindex(["PER", "LOC", "ORG", "MISC", "overall"])
display(NER_TABLE)

# %%
# ---- error analysis: 5 false positives and 5 false negatives ----
train_ent_counts = collections.Counter()
for s in S_TRAIN:
    for (a, b, t) in chunks(s["ner"]): train_ent_counts[(tuple(s["tokens"][a:b + 1]), t)] += 1
def explain_fp(s, span, gold_chunks):
    a, b, t = span; words = s["tokens"][a:b + 1]
    for (ga, gb, gt) in gold_chunks:
        if ga <= b and a <= gb:
            if gt != t: return f"type confusion – gold is {gt}; the context (neighbouring words) is typical for {t}"
            return f"boundary error – gold span is '{' '.join(s['tokens'][ga:gb+1])}'"
    if any(W2I.get(w, UNK) == UNK for w in words): return "OOV token (<UNK> embedding) predicted as entity from context only"
    return ("model generalised from context (e.g. a title/verb pattern) to a word absent from the gazetteer – may be a real "
            "entity missed by the silver gazetteer annotation")
def explain_fn(s, span, pred_chunks):
    a, b, t = span; words = tuple(s["tokens"][a:b + 1])
    for (pa, pb, pt) in pred_chunks:
        if pa <= b and a <= pb:
            return f"partially/mis-typed – predicted {pt} span '{' '.join(s['tokens'][pa:pb+1])}'"
    n_tr = train_ent_counts.get((words, t), 0)
    oov = [w for w in words if W2I.get(w, UNK) == UNK]
    if oov: return f"OOV words {oov} have the <UNK> embedding, so the model cannot recognise them"
    if n_tr == 0: return "entity never seen as an entity in training data; its embedding alone is not distinctive enough"
    return f"seen {n_tr}× in training but the local context here is ambiguous (e.g. used as modifier)"
FP, FN = [], []
for s, g, p in zip(S_TEST, G_NER, P_NER):
    gc, pc = set(chunks(g)), set(chunks(p))
    for sp_ in sorted(pc - gc): FP.append((s, sp_, gc))
    for sp_ in sorted(gc - pc): FN.append((s, sp_, pc))
def ctx(s, a, b, w=4): return " ".join(s["tokens"][max(0, a - w):a]) + " [[" + " ".join(s["tokens"][a:b + 1]) + "]] " + " ".join(s["tokens"][b + 1:b + 1 + w])
ERR_ROWS = []
print(f"Test set: {len(FP)} false positives, {len(FN)} false negatives (CRF model)\n")
print("— 5 FALSE POSITIVES —")
for s, (a, b, t), gc in FP[:5]:
    e_ = explain_fp(s, (a, b, t), gc); ERR_ROWS.append(("FP", t, ctx(s, a, b), e_)); print(f"  pred {t:4s}: {ctx(s, a, b)}\n     → {e_}")
print("— 5 FALSE NEGATIVES —")
for s, (a, b, t), pc in FN[:5]:
    e_ = explain_fn(s, (a, b, t), pc); ERR_ROWS.append(("FN", t, ctx(s, a, b), e_)); print(f"  gold {t:4s}: {ctx(s, a, b)}\n     → {e_}")
def _etype(e_):
    for k in ["type confusion", "boundary", "partially", "OOV", "never seen", "gazetteer", "seen"]:
        if k in e_: return {"partially": "boundary/partial", "boundary": "boundary/partial", "gazetteer": "context generalisation (not in gazetteer)",
                            "seen": "ambiguous context"}.get(k, k)
    return "other"
ERR_TYPES = collections.Counter([_etype(explain_fp(s, sp_, gc)) for s, sp_, gc in FP] + [_etype(explain_fn(s, sp_, pc)) for s, sp_, pc in FN])
print("\nError categories over all test FP+FN:", dict(ERR_TYPES))
RESULTS["p2_ner"] = {"mode": NER_MODE, "err_types": dict(ERR_TYPES.most_common()), "crf": {k: {m: round(v, 2) for m, v in RES_CRF[k].items()} for k in RES_CRF},
                     "softmax": {k: {m: round(v, 2) for m, v in RES_SOFT[k].items()} for k in RES_SOFT},
                     "n_fp": len(FP), "n_fn": len(FN), "errors": [r_[0] + ": " + r_[3] for r_ in ERR_ROWS]}

# %% [markdown]
# ### 5.3 Ablation study (same data split, one change at a time; baseline = fine-tuned BiLSTM(-CRF))

# %%
ABL = {}
base_pos_test = pos_test_metrics(pos_ft)[1]; base_ner_test = ner_test(ner_ft)[0]["overall"]["f1"]
ABL["Baseline (BiLSTM, dropout, W2V init, CRF for NER)"] = (f_pos_ft, base_pos_test, f_ner_ft, base_ner_test)
for name, kw in [("A1 Unidirectional LSTM", dict(bidirectional=False)), ("A2 No dropout", dict(dropout=0.0)),
                 ("A3 Random embedding init", dict(random_init=True))]:
    mp, _, fp_ = train_tagger("pos", **kw, tag=f"{name} POS"); mn, _, fn_ = train_tagger("ner", **kw, tag=f"{name} NER")
    ABL[name] = (fp_, pos_test_metrics(mp)[1], fn_, ner_test(mn)[0]["overall"]["f1"])
_soft_ft = ner_soft if NER_MODE == "fine-tuned" else train_tagger("ner", use_crf=False, tag="A4 softmax")[0]
ABL["A4 Softmax instead of CRF (NER)"] = (np.nan, np.nan, evaluate(_soft_ft, *encode(S_VAL, "ner", NER2I), "ner")[0], ner_test(_soft_ft)[0]["overall"]["f1"])
ABL_DF = pd.DataFrame(ABL, index=["POS val macro-F1", "POS test macro-F1", "NER val F1", "NER test F1"]).T.round(2)
display(ABL_DF)

def _d(row, col): return ABL_DF.loc[row, col] - ABL_DF.iloc[0][col]
abl_text = []
for row, what in [("A1 Unidirectional LSTM", "backward context"), ("A2 No dropout", "dropout regularisation"),
                  ("A3 Random embedding init", "pre-trained Word2Vec initialisation")]:
    dp, dn = _d(row, "POS test macro-F1"), _d(row, "NER test F1")
    abl_text.append(f"{row}: POS macro-F1 {dp:+.2f}, NER F1 {dn:+.2f} vs. baseline – "
                    + (f"removing {what} hurts, so it contributes to the baseline." if dp + dn < -1.0 else
                       f"removing {what} changes little on this small, rule-annotated data set (differences within run-to-run noise)." if abs(dp + dn) <= 1.0 else
                       f"removing {what} actually helps here (+{dp + dn:.1f} combined); with only ~350 silver-labelled training sentences "
                       f"the model benefits from {'a less constrained input space' if 'embedding' in what else 'more capacity / less regularisation'}."))
dn = _d("A4 Softmax instead of CRF (NER)", "NER test F1")
abl_text.append(f"A4 Softmax instead of CRF: NER F1 {dn:+.2f} – " + ("structured CRF decoding helps by forbidding invalid BIO transitions (e.g. O→I-PER) and keeping multi-word entities together."
                if dn < -1.0 else "CRF and softmax perform about the same here; the BiLSTM already sees both directions, so few invalid BIO sequences remain to be fixed."
                if dn <= 1.0 else "the softmax decoder is better on this run; with little training data the CRF transition matrix is estimated from few multi-token entities."))
for t_ in abl_text: print("•", t_)
RESULTS["p2_ablation"] = {k: [None if (isinstance(v, float) and np.isnan(v)) else float(v) for v in vals] for k, vals in ABL_DF.iterrows()}
RESULTS["p2_ablation_text"] = abl_text
RESULTS["p2_modes"] = MODE_DF.to_dict()
