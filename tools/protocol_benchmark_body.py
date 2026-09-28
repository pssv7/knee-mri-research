"""Runs after the generated notebook-10 inference definitions on Kaggle."""

import traceback

out = pathlib.Path("/kaggle/working")
report = {"purpose": "fixed 15-model inference benchmark; no training or submission",
          "gpu": torch.cuda.get_device_name(), "members": len(explicit_members),
          "manifest_sha256": hashlib.sha256(pathlib.Path(ENSEMBLE_JSON).read_bytes()).hexdigest(),
          "setup_seconds": time.monotonic() - RUN_STARTED,
          "reference_checks": [], "raw_studies": [], "errors": [],
          "references_complete": False, "raw_complete": False}


def save_report():
    report["elapsed_seconds"] = time.monotonic() - RUN_STARTED
    report["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
    report["peak_reserved_gpu_gb"] = torch.cuda.max_memory_reserved() / 2**30
    temporary = out / "benchmark_report.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(out / "benchmark_report.json")


try:
    assert len(explicit_members) == 15
    assert np.array_equal(np.asarray(member_weights), np.full(15, 1 / 15))
    actual = {f"{m['spec']['name']}:{m['metadata']['fold']}": m['metadata']['sha256'] for m in explicit_members}
    assert actual == {key: value["sha256"] for key, value in EXPECTED_MODELS.items()}
    report["checkpoints"] = actual
    torch.cuda.reset_peak_memory_stats()
    series_frame = pd.read_csv(find_competition_file("train_series.csv"))
    sides = pd.read_parquet(find_one("side_predictions.parquet"))
    frames, shards = [], {}
    for path in find_inputs("cache_train_series_*.npy"):
        suffix = pathlib.Path(path).stem.removeprefix("cache_")
        frames.append(pd.read_parquet(find_one(f"manifest_{suffix}.parquet")).assign(cache_path=path))
        shards[path] = np.load(path, mmap_mode="r")
    assert len(frames) == 3, "All three full cache shards are required"
    index = knee_dataset.StudyIndex(pd.concat(frames, ignore_index=True), series_frame,
                                   labels=train_labels, sides=sides)
    folded = pd.read_parquet(find_one("reconstructed_folds.parquet"))
    assert len(folded) == 4349 and set(index.studies) == set(folded[config.ID_COL])
    gold = splits.load_gold_uids(find_competition_file("train.csv"))
    splits.assert_no_gold_leakage(folded[config.ID_COL], gold)
    counts = series_frame.groupby(config.ID_COL).size().sort_index()
    for member in explicit_members:
        identity = f"{member['spec']['name']}:{member['metadata']['fold']}"
        reference = pd.read_parquet(find_one(EXPECTED_MODELS[identity]["reference"]))
        val = set(folded.loc[folded.fold == member["metadata"]["fold"], config.ID_COL])
        assert reference[config.ID_COL].is_unique and set(reference[config.ID_COL]) == val
        # The most series-rich held-out studies exercise each member's cap and
        # protocol selection, with no labels involved in sample selection.
        chosen = counts.loc[sorted(val)].sort_values(ascending=False, kind="stable").head(8).index
        predictions = []
        for uid in chosen:
            rows = index._by_study[uid]
            volumes = [np.asarray(shards[row.cache_path][int(row.position)]) for _, row in rows.iterrows()]
            predictions.append(artifacts.predict_members([member], volumes, rows, device)[0])
        expected = reference.set_index(config.ID_COL).loc[chosen, config.LABELS].to_numpy()
        delta = float(np.max(np.abs(np.stack(predictions) - expected)))
        report["reference_checks"].append({"identity": identity, "studies": chosen.tolist(), "max_abs_delta": delta})
        save_report()
        assert np.isfinite(delta) and delta < .01, f"OOF reproduction failed for {identity}: {delta}"
        print(f"Reference {identity}: {delta:.6f}", flush=True)
    report["references_complete"] = len(report["reference_checks"]) == 15
    save_report()

    series_meta = series_frame.set_index("SeriesInstanceUID")
    test_root = config.DATA_ROOT / "train_series"
    largest = counts.sort_values(ascending=False, kind="stable").head(5).index.tolist()
    selected = largest + counts.drop(index=largest).sample(n=95, random_state=954).index.tolist()
    report["raw_selected_uids"] = selected
    report["selection"] = "Five studies with most series plus 95 seeded random; same sample as incumbent benchmark"
    predictions = []
    for uid in selected:
        if time.monotonic() - RUN_STARTED >= 3230:
            break
        try:
            torch.cuda.synchronize()
            start = time.monotonic()
            volumes, uids, positions = build_volumes(uid, fast=False)
            assert volumes, "No readable series"
            side = predict_side(volumes, positions, uid, uids=uids)
            torch.cuda.synchronize()
            decoded = time.monotonic()
            probs = tuned_predict(volumes, uids, side, fast=False)
            torch.cuda.synchronize()
            finished = time.monotonic()
            assert probs.shape == (15, 12) and np.isfinite(probs).all()
            prediction = np.average(probs, axis=0, weights=member_weights)
            assert ((prediction >= 0) & (prediction <= 1)).all()
            predictions.append({config.ID_COL: uid, **dict(zip(config.LABELS, prediction.tolist()))})
            report["raw_studies"].append({"uid": uid, "series": len(uids),
                "decode_side_seconds": decoded - start, "forward_seconds": finished - decoded,
                "total_seconds": finished - start})
            print(f"Raw benchmark {len(predictions)}/100: {finished - start:.2f}s", flush=True)
        except Exception:
            report["errors"].append({"uid": uid, "traceback": traceback.format_exc()})
            raise
        save_report()
    assert len(predictions) == 100, "Benchmark did not finish 100 studies within its budget"
    pd.DataFrame(predictions).to_parquet(out / "raw_benchmark_predictions.parquet", index=False)
    seconds = np.array([r["total_seconds"] for r in report["raw_studies"]])
    report.update(raw_complete=True, raw_mean_seconds=float(seconds.mean()),
        raw_p95_seconds=float(np.quantile(seconds, .95)), raw_max_seconds=float(seconds.max()),
        projected_1300_study_hours=float((report["setup_seconds"] + 1300 * seconds.mean()) / 3600),
        runtime_caveat="Training sample; hidden-test composition, CPU allocation and I/O contention may differ")
    report["runtime_gate_passed"] = report["projected_1300_study_hours"] < 6.3
    save_report()
    assert report["runtime_gate_passed"], "Runtime exceeds the 6.3-hour screening threshold"
    print("PROTOCOL_BENCHMARK_PASSED", flush=True)
except Exception:
    report["fatal_error"] = traceback.format_exc()
    save_report()
    raise
finally:
    watchdog.cancel()
