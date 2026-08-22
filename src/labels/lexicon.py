"""Rule-based extraction of the twelve findings from multilingual reports.

The corpus spans roughly eleven languages -- English (39%), Greek, Croatian/
Serbian, Spanish, Turkish, German, Bulgarian/Russian, Dutch, French, Italian,
Portuguese. Rather than a language x finding cross-product, this module keys on
the Greek and Latin stems that clinical vocabulary shares across all of them
(menisc- / menisk- / μηνίσκ- / мениск-), with language-specific additions only
where the local term is genuinely unrelated (Turkish "kıkırdak" for cartilage).

Each finding resolves to one of three states:

    1     an unnegated mention above the severity floor
    0     mentioned but negated, or present only below the severity floor
    None  never mentioned -- which is *not* the same as absent, because
          reports narrate positives and omit much else

Callers must preserve the None state. Collapsing it to 0 injects systematic
noise into exactly the findings that are least often narrated.
"""

import re
import unicodedata
from typing import Dict, List, Optional

from src import config

# ---------------------------------------------------------------------------
# Cues
# ---------------------------------------------------------------------------
# Negation, across every language present in the corpus. Matched as whole words
# within a bounded window before the finding term.
NEGATION_CUES = [
    # English / Latin-script general
    "no", "not", "without", "absent", "negative", "unremarkable", "intact",
    "normal", "preserved", "free of", "rule out", "ruled out",
    # Spanish / Portuguese
    "sin", "no", "ausencia", "ausente", "conservado", "conservada", "integro",
    "íntegro", "integra", "íntegra", "normales", "sem", "nao", "não",
    # Dutch
    "geen", "niet", "zonder", "intacte", "ongestoord",
    # German
    "kein", "keine", "keinen", "nicht", "ohne", "unauffällig", "regelrecht",
    "unauffaellig",
    # French
    "pas", "sans", "absence", "aucun", "aucune", "normale",
    # Italian
    "non", "senza", "assenza", "nella norma",
    # Turkish
    "yok", "yoktur", "izlenmemiştir", "izlenmemistir", "saptanmamıştır",
    "değil", "degil", "normaldir", "korunmuş", "korunmus",
    # Croatian / Serbian / Bosnian
    "bez", "nema", "nije", "uredan", "uredna", "uredno", "očuvan", "ocuvan",
    # Bulgarian / Russian
    "без", "нет", "няма", "не", "отсутств", "интактн", "сохранен",
    # Greek
    "χωρίς", "δεν", "ουδεμία", "ουδέν", "φυσιολογικ", "ακέραι",
]

# Hedged or sub-threshold mentions. Ground truth was adjudicated against
# explicit severity thresholds, so "trace effusion" and "large effusion" are
# not the same label.
MINIMAL_CUES = [
    "minimal", "mínimo", "minimo", "minima", "mínima", "trace", "tiny",
    "slight", "small", "mild", "leve", "ligero", "ligera", "discreto",
    "discreta", "gering", "geringe", "klein", "kleine", "wenig", "licht",
    "lichte", "petit", "petite", "faible", "lieve", "piccolo", "az", "küçük",
    "kucuk", "hafif", "minimalan", "mali", "blag", "малък", "минимален",
    "незначител", "ελάχιστ", "μικρ", "ήπι",
]

UNCERTAIN_CUES = [
    "possible", "possibly", "suspicious", "suspected", "probable", "probably",
    "questionable", "equivocal", "cannot exclude", "cannot be excluded",
    "may represent", "posible", "sospecha", "sospechoso", "dudoso",
    "verdacht", "möglich", "moglich", "fraglich", "mogelijk", "verdenking",
    "suspect", "possible", "suspicion", "şüpheli", "supheli", "olası", "olasi",
    "sumnja", "moguć", "mogu", "възможн", "подозр", "πιθαν", "ύποπτ",
]

# Negation that *follows* the finding term. Turkish is verb-final, so
# "eklem ici sivi yoktur" negates with a trailing word and a backwards-only
# search misses it entirely -- that is a tenth of this corpus. English and
# several others do the same with trailing adjectives ("meniscus intact").
# Kept deliberately narrower than NEGATION_CUES: a forward search over the full
# cue list would let an unrelated "normal" in the same clause cancel a real
# finding.
POSTFIX_NEGATION_CUES = [
    # Turkish
    "yok", "yoktur", "izlenmemiştir", "izlenmemistir", "saptanmamıştır",
    "saptanmamistir", "değil", "degil", "normaldir", "normaldi", "korunmuş",
    "korunmus", "izlenmedi",
    # English / Latin-script trailing adjectives
    "intact", "unremarkable", "preserved", "normal", "normale", "normales",
    "within normal limits", "is normal", "are normal",
    # Dutch / German
    "intacte", "unauffällig", "unauffaellig", "regelrecht", "ongestoord",
    # Croatian / Serbian
    "uredan", "uredna", "uredno", "očuvan", "ocuvan", "očuvana", "ocuvana",
    # Bulgarian / Russian
    "сохранен", "интактн", "не се визуализира", "непроменена",
    # Greek
    "φυσιολογικ", "ακέραι",
]

# Terms that signal a torn or damaged structure, used by the ligament and
# meniscus findings where the anatomy word alone means nothing.
INJURY_STEMS = [
    "tear", "torn", "rupture", "ruptur", "disrupt", "discontinu", "avuls",
    "sprain", "lesion", "lesión", "lesao", "lesão", "injur", "insufficien",
    "rotura", "ruptura", "roto", "rota", "desgarro", "riss", "einriss",
    "scheur", "letsel", "déchirure", "dechirure", "rupture", "lacerazion",
    "yırtık", "yirtik", "rüptür", "ruptur", "kopma", "puknuć", "puknuc",
    "ruptura", "razdor", "разрыв", "руптура", "скъсване", "увреж",
    "ρήξη", "ρηξη", "ρήγμα",
]


def _norm(text: str) -> str:
    """Lowercase and strip combining marks, keeping non-Latin scripts intact."""
    lowered = text.lower()
    decomposed = unicodedata.normalize("NFD", lowered)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", stripped)


def _stem_pattern(stems: List[str]) -> re.Pattern:
    """Compile stems into one alternation matching at a word boundary."""
    escaped = sorted((re.escape(_norm(s)) for s in stems), key=len, reverse=True)
    return re.compile(r"(?<![^\W\d_])(?:" + "|".join(escaped) + r")", re.UNICODE)


# Degenerative disease stems. The three OA findings differ only by which
# compartment the disease is localised to, so they share this anchor and are
# separated by the localiser lists below. Anchoring on the compartment word
# instead does not work: "medial" appears in almost every report describing
# normal anatomy.
OA_STEMS = [
    "osteoarthr", "osteoarthrit", "arthrosis", "arthrose", "artrosis",
    "artroz", "artroza", "gonarthros", "gonartros", "gonartroz",
    "degenerative change", "degenerative disease", "cambios degenerativos",
    "degenerativne", "degenerativ", "chondropath", "condropat", "chondromalac",
    "condromalac", "chondral loss", "chondral thinning", "cartilage loss",
    "cartilage thinning", "perdida de cartilago", "adelgazamiento del cartilago",
    "knorpelverlust", "knorpelschaden", "kraakbeenverlies", "kikirdak kayb",
    "kikirdak incelme", "hrskavice", "denudacij", "osteofit", "osteophyt",
    "espolon", "spur formation", "остеоартр", "артроз", "остеофит",
    "хондромалац", "изтънен", "οστεοαρθρ", "αρθριτ", "χονδροπαθ",
    "χονδρομαλακ", "οστεοφυτ",
]

MEDIAL_LOCALIZERS = [
    "medial", "media", "mediale", "medialen", "medialer", "medialne",
    "medijaln", "interno", "interna", "innen", "innenseit", "binnen",
    "ic kompartman", "медиальн", "медиалн", "вътреш", "εσω", "εσωτερικ",
]

LATERAL_LOCALIZERS = [
    "lateral", "laterale", "lateralen", "lateraler", "lateraln", "externo",
    "externa", "aussen", "buiten", "dis kompartman", "латеральн", "латералн",
    "външ", "εξω", "εξωτερικ",
]

PF_LOCALIZERS = [
    "patellofemoral", "patelofemoral", "femoropatellar", "femoropatelar",
    "retropatellar", "retropatelar", "patellar", "patelar", "patella",
    "rotulian", "rotuliana", "rotuliano", "trochlea", "troclea",
    "femoropatelarn", "patellofemoraln", "пателлофеморальн", "пателофеморалн",
    "надколенник", "капачка", "επιγονατιδ", "επιγονατιδομηριαι",
]

# ---------------------------------------------------------------------------
# Finding definitions
# ---------------------------------------------------------------------------
# `anatomy` locates the structure; `requires_injury` marks findings where the
# structure being named is not itself abnormal (a meniscus is always present --
# only a torn one is a positive label).
FINDINGS = {
    "ACL": {
        "anatomy": ["acl", "lca", "anterior cruciate", "ligamento cruzado anterior",
                    "voorste kruisband", "vorderes kreuzband", "kreuzband",
                    "ligament croise anterieur", "legamento crociato anteriore",
                    "on capraz bag", "onc apraz", "capraz bag", "prednji ukrizeni",
                    "prednja ukrstena", "predna krstna", "передня крестообразн",
                    "предна кръстна", "крестообразн", "prosthios chiastos",
                    "προσθιο χιαστο", "χιαστο"],
        "requires_injury": True,
    },
    "MCL": {
        "anatomy": ["mcl", "medial collateral", "ligamento colateral medial",
                    "ligamento lateral interno", "mediale collaterale",
                    "innenband", "mediales kollateralband", "mediale band",
                    "ligament collateral medial", "ic kollateral",
                    "medial kollateral", "medijalni kolateralni",
                    "медиальн коллатеральн", "вътрешна колатерал",
                    "εσω πλαγιο", "εσωτερικο πλαγιο"],
        "requires_injury": True,
    },
    "Medial Meniscus": {
        "anatomy": ["medial menisc", "menisco medial", "menisco interno",
                    "mediale meniscus", "innenmeniskus", "medialer meniskus",
                    "menisque interne", "menisque medial", "menisco mediale",
                    "medial menisk", "ic menisk", "medijalni meniskus",
                    "medijalnog meniskusa", "медиальн мениск",
                    "вътрешния менискус", "εσω μηνισκ"],
        "requires_injury": True,
    },
    "Lateral Meniscus": {
        "anatomy": ["lateral menisc", "menisco lateral", "menisco externo",
                    "laterale meniscus", "aussenmeniskus", "lateraler meniskus",
                    "menisque externe", "menisque lateral", "menisco laterale",
                    "lateral menisk", "dis menisk", "lateralni meniskus",
                    "латеральн мениск", "външния менискус", "εξω μηνισκ"],
        "requires_injury": True,
    },
    # The OA family anchors on the disease stem and requires a compartment
    # localiser nearby, within a tighter window than other findings use.
    "Medial OA": {
        "anatomy": OA_STEMS,
        "requires_injury": False,
        "context": MEDIAL_LOCALIZERS,
        "context_window": 55,
        "context_is_localizer": True,
    },
    "Lateral OA": {
        "anatomy": OA_STEMS,
        "requires_injury": False,
        "context": LATERAL_LOCALIZERS,
        "context_window": 55,
        "context_is_localizer": True,
    },
    "PF OA": {
        "anatomy": OA_STEMS,
        "requires_injury": False,
        "context": PF_LOCALIZERS,
        "context_window": 55,
        "context_is_localizer": True,
    },
    "Effusion": {
        "anatomy": ["effusion", "efusion", "derrame", "hydrops", "hidrartros",
                    "gelenkerguss", "erguss", "gewrichtsvocht", "vocht",
                    "epanchement", "versamento", "efuzyon", "eklem ici sivi",
                    "eklem sivisi", "izliv", "zglobni izliv", "wypot",
                    "выпот", "излив", "ставен излив", "συλλογη υγρου",
                    "αρθριτικη συλλογη", "ενδαρθρικη συλλογη"],
        "requires_injury": False,
    },
    "Synovitis": {
        "anatomy": ["synovit", "sinovit", "synovial thickening",
                    "engrosamiento sinovial", "synoviale verdickung",
                    "synoviale proliferation", "synovitis", "sinovitis",
                    "sinovijalis", "sinovijalna", "синовит", "υμενιτιδα",
                    "αρθρικου υμενα"],
        "requires_injury": False,
    },
    "Baker's": {
        "anatomy": ["baker", "popliteal cyst", "quiste poplite", "quistes poplite",
                    "bakerse cyste", "poplitealzyste", "bakerzyste",
                    "kyste poplite", "cisti poplitea", "poplitealna cista",
                    "bakerova cista", "poplitealni", "baker kisti",
                    "киста бейкера", "подколенна киста", "κυστη baker",
                    "κυστη μπεικερ", "ιγνυακη κυστη"],
        "requires_injury": False,
    },
    "Contusion": {
        "anatomy": ["contusion", "contusion osea", "bone contusion",
                    "bone marrow edema", "bone marrow oedema", "marrow edema",
                    "edema oseo", "edema de medula osea", "edema medular",
                    "knochenmarksodem", "knochenodem", "knochenmarködem",
                    "beenmergoedeem", "botoedeem", "oedeme osseux",
                    "edema midollare", "kemik ilig odem", "kemik odem",
                    "kontuzyon", "kostana kontuzija", "edem kosti",
                    "костномозгов отек", "костномозъчен едем", "оστικο οιδημα",
                    "οιδημα μυελου", "μυελικο οιδημα"],
        "requires_injury": False,
    },
    "Fracture": {
        "anatomy": ["fracture", "fractura", "fraktur", "fractuur", "frattura",
                    "kirik", "kırık", "prelom", "fraktura", "perelom",
                    "перелом", "фрактура", "счупване", "καταγμα", "καταγματ"],
        "requires_injury": False,
    },
}

# Window, in characters, searched backwards from a term for a negation cue.
NEGATION_WINDOW = 60
# Window searched around a term for severity and certainty cues.
SEVERITY_WINDOW = 45
# Window searched forward from a term for a trailing negator.
POSTFIX_NEGATION_WINDOW = 40
# Window searched around an anatomy match for the injury or context stem.
CONTEXT_WINDOW = 90

_NEGATION_RE = _stem_pattern(NEGATION_CUES)
_POSTFIX_NEGATION_RE = _stem_pattern(POSTFIX_NEGATION_CUES)
_MINIMAL_RE = _stem_pattern(MINIMAL_CUES)
_UNCERTAIN_RE = _stem_pattern(UNCERTAIN_CUES)
_INJURY_RE = _stem_pattern(INJURY_STEMS)

_COMPILED = {}
for _name, _spec in FINDINGS.items():
    _COMPILED[_name] = {
        "anatomy": _stem_pattern(_spec["anatomy"]),
        "requires_injury": _spec["requires_injury"],
        "context": _stem_pattern(_spec["context"]) if _spec.get("context") else None,
        "context_window": _spec.get("context_window", CONTEXT_WINDOW),
        "context_is_localizer": _spec.get("context_is_localizer", False),
    }

# Clause boundaries. Negation does not carry across them, so "no tear. Effusion
# present" must not read as a negated effusion.
_CLAUSE_SPLIT = re.compile(r"[.;:\n]|\bbut\b|\bhowever\b|\bpero\b|\bmaar\b|\baber\b")


def _clause_bounds(text: str, position: int) -> tuple:
    """Start and end of the clause containing `position`."""
    start = 0
    end = len(text)
    for match in _CLAUSE_SPLIT.finditer(text):
        if match.end() <= position:
            start = match.end()
        elif match.start() > position:
            end = match.start()
            break
    return start, end


def _is_negated(text: str, position: int, length: int = 0) -> bool:
    """True when a negation cue applies to the term at `position`.

    Searches backwards over the full cue list, and forwards over the narrower
    postfix list -- verb-final languages put the negator after the finding.
    Both searches stop at the clause boundary so a negation cannot leak into a
    neighbouring statement.
    """
    clause_start, clause_end = _clause_bounds(text, position)

    window_start = max(clause_start, position - NEGATION_WINDOW)
    if _NEGATION_RE.search(text, window_start, position):
        return True

    term_end = position + length
    window_end = min(clause_end, term_end + POSTFIX_NEGATION_WINDOW)
    return bool(_POSTFIX_NEGATION_RE.search(text, term_end, window_end))


def _severity(text: str, position: int, length: int) -> float:
    """Confidence that the mention clears the adjudication threshold.

    1.0 for an unqualified positive, lower when the report hedges the finding
    or calls it minimal. These scores feed the fusion step as soft targets
    rather than being thresholded here.
    """
    start = max(0, position - SEVERITY_WINDOW)
    end = min(len(text), position + length + SEVERITY_WINDOW)
    window = text[start:end]

    score = 1.0
    if _MINIMAL_RE.search(window):
        score *= 0.35
    if _UNCERTAIN_RE.search(window):
        score *= 0.5
    return score


def _has_context(text: str, position: int, length: int, pattern,
                 window: int = CONTEXT_WINDOW, clause_scoped: bool = True) -> bool:
    """True when a required injury, disease, or localiser stem sits nearby.

    Injury context is clause-scoped: a tear described in one sentence must not
    license a finding named in the next. Compartment localisers are not, because
    reports routinely name the disease in one clause and the compartment in the
    following one ("Gonarthrosis. Most marked laterally."). Enforcing clause
    scope on localisers cost Lateral OA two thirds of its recall.
    """
    if pattern is None:
        return True

    if clause_scoped:
        clause_start, clause_end = _clause_bounds(text, position)
    else:
        clause_start, clause_end = 0, len(text)

    start = max(clause_start, position - window)
    end = min(clause_end, position + length + window)
    return bool(pattern.search(text, start, end))


def extract_finding(report: str, finding: str) -> Dict:
    """Resolve one finding from one report."""
    spec = _COMPILED[finding]
    text = _norm(report)

    context_pattern = spec["context"]
    context_window = spec["context_window"]
    localizer_based = spec["context_is_localizer"]
    if spec["requires_injury"]:
        context_pattern = _INJURY_RE
        context_window = CONTEXT_WINDOW
        localizer_based = False

    mentioned = False
    best = None

    for match in spec["anatomy"].finditer(text):
        position, length = match.start(), match.end() - match.start()
        in_context = _has_context(text, position, length, context_pattern,
                                  context_window,
                                  clause_scoped=not localizer_based)

        if localizer_based and not in_context:
            # Degenerative disease was described, but not in this compartment.
            # That is silence about this finding, not evidence against it --
            # counting it as a mention would report a confident zero for a
            # compartment the radiologist never discussed.
            continue

        mentioned = True

        if not in_context:
            # The structure is named but nothing abnormal is asserted about it,
            # which is a normal-anatomy sentence rather than a finding.
            continue
        if _is_negated(text, position, length):
            continue

        score = _severity(text, position, length)
        clause_start, clause_end = _clause_bounds(text, position)
        candidate = {
            "present": 1,
            "severity": score,
            "span": report[clause_start:clause_end].strip()[:200],
        }
        # Keep the most confident positive mention.
        if best is None or score > best["severity"]:
            best = candidate

    if best is not None:
        return best
    if mentioned:
        return {"present": 0, "severity": 0.0, "span": None}
    return {"present": None, "severity": None, "span": None}


def extract_report(report: str) -> Dict[str, Dict]:
    """Resolve all twelve findings from one report."""
    if not isinstance(report, str) or not report.strip():
        return {name: {"present": None, "severity": None, "span": None}
                for name in config.LABELS}
    return {name: extract_finding(report, name) for name in config.LABELS}


def extract_frame(reports, ids=None):
    """Vectorise extraction over a series of reports into a DataFrame.

    Returns one row per report with `<finding>` (0/1/NaN) and
    `<finding>_severity` columns.
    """
    import pandas as pd

    records = []
    for report in reports:
        result = extract_report(report)
        row = {}
        for name in config.LABELS:
            row[name] = result[name]["present"]
            row[f"{name}_severity"] = result[name]["severity"]
        records.append(row)

    frame = pd.DataFrame(records)
    if ids is not None:
        frame.insert(0, config.ID_COL, list(ids))
    return frame
