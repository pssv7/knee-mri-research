import ast

from src.labels import translate
from tools import prepare_translation_colab

FAITHFUL = {
    "Kein Riss des lateralen Meniskus.": "No tear of the lateral meniscus.",
    "Posible rotura del menisco lateral.": "Possible tear of the lateral meniscus.",
    "Ρήξη του έξω μηνίσκου.": "Tear of the lateral meniscus.",
    "Medial menisküs arka boynuzunda yırtık izlenmemiştir.":
        "No tear is seen in the posterior horn of the medial meniscus.",
    "Edem koštane srži lateralnog femoralnog kondila.": "Bone marrow edema of the lateral femoral condyle.",
    "Хондромалация на пателата II степен.": "Grade II chondromalacia of the patella.",
    "Non si può escludere una contusione ossea.": "A bone contusion cannot be excluded.",
}


def translator(mapping):
    return lambda items: [mapping[text] for text, _ in items]


def test_a_faithful_translator_passes():
    assert translate.self_check(translator(FAITHFUL)) == []


def test_self_check_catches_a_dropped_negation():
    broken = dict(FAITHFUL, **{"Kein Riss des lateralen Meniskus.": "Tear of the lateral meniscus."})
    failures = translate.self_check(translator(broken))
    assert [f["language"] for f in failures] == ["de"]


def test_self_check_catches_a_dropped_hedge_and_a_swapped_side():
    broken = dict(FAITHFUL, **{
        "Posible rotura del menisco lateral.": "Tear of the lateral meniscus.",
        "Ρήξη του έξω μηνίσκου.": "Tear of the medial meniscus.",
        "Non si può escludere una contusione ossea.": "Bone contusion."})
    assert {f["language"] for f in translate.self_check(translator(broken))} == {"es", "el", "it"}


def test_self_check_catches_a_lost_grade():
    broken = dict(FAITHFUL, **{"Хондромалация на пателата II степен.": "Chondromalacia of the patella."})
    assert [f["language"] for f in translate.self_check(translator(broken))] == ["bg"]


def test_an_identity_translator_fails_the_non_latin_checks():
    failures = translate.self_check(lambda items: [text for text, _ in items])
    assert {"el", "bg"} <= {f["language"] for f in failures}


def test_result_checks_find_missing_empty_truncated_and_stray():
    requests = [{"case_id": "K1"}, {"case_id": "K2"}, {"case_id": "K3"}]
    results = [{"case_id": "K1", "translation": "ok", "finish_reason": "stop"},
               {"case_id": "K2", "translation": "  ", "finish_reason": "stop"},
               {"case_id": "K9", "translation": "x", "finish_reason": "length"}]
    problems = translate.check_results(requests, results)
    assert "K3: missing translation" in problems and "K2: empty translation" in problems
    assert "K9: translation for an unknown case" in problems
    truncated = translate.check_results([{"case_id": "K1"}],
                                        [{"case_id": "K1", "translation": "x", "finish_reason": "length"}])
    assert truncated == ["K1: truncated at the token limit"]


def test_clean_strips_a_label_prefix():
    assert translate.clean("Translation: No effusion.") == "No effusion."


def test_notebook_cells_parse_and_embed_the_module_verbatim():
    nb = prepare_translation_colab.make()
    code = ["".join(c["source"]) for c in nb.cells if c["cell_type"] == "code"]
    for cell in code:
        ast.parse("\n".join(l for l in cell.splitlines() if not l.lstrip().startswith("!")))
    embedded = [c for c in code if "TRANSLATE_SOURCE =" in c]
    assert len(embedded) == 1
    namespace = {}
    exec(embedded[0].split("pathlib.Path(")[0].split("import importlib, pathlib, sys")[1], namespace)
    assert namespace["TRANSLATE_SOURCE"] == open(translate.__file__, encoding="utf-8").read()


def test_notebook_self_check_runs_before_any_report_is_translated():
    code = ["".join(c["source"]) for c in prepare_translation_colab.make().cells if c["cell_type"] == "code"]
    check = next(i for i, c in enumerate(code) if "translate.self_check(" in c)
    reports = next(i for i, c in enumerate(code) if 'r["report"]' in c and "generate(" in c)
    assert check < reports
    assert "raise SystemExit" in code[check]


def test_the_committed_notebook_matches_its_generator():
    import json
    nb = prepare_translation_colab.make()
    on_disk = json.load(open(prepare_translation_colab.NOTEBOOK, encoding="utf-8"))
    assert on_disk["cells"] == nb.cells
