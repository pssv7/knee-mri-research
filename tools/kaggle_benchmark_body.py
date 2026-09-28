"""Embedded after notebook 10 setup by prepare_kaggle_benchmark.py, not standalone."""

import traceback

out = pathlib.Path("/kaggle/working")
report = {"purpose": "private inference benchmark; no submission or training",
          "gpu": torch.cuda.get_device_name(), "members": len(explicit_members),
          "manifest": str(ENSEMBLE_JSON), "errors": [], "raw_studies": [],
          "setup_seconds": time.monotonic() - RUN_STARTED,
          "oof_complete": False, "raw_complete": False}


def save_report():
    report["elapsed_seconds"] = time.monotonic() - RUN_STARTED
    report["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
    temporary = out / "benchmark_report.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(out / "benchmark_report.json")


def has_time(reserve=120):
    return time.monotonic() - RUN_STARTED < 3350 - reserve


def cached_prediction(member, study):
    rows = index._by_study[study]
    volumes = [np.asarray(shards[row.cache_path][int(row.position)]) for _, row in rows.iterrows()]
    return artifacts.predict_members([member], volumes, rows, device)[0]


try:
    torch.cuda.reset_peak_memory_stats()
    assert len(explicit_members) == 10
    series_frame = pd.read_csv(find_competition_file("train_series.csv"))
    sides = pd.read_parquet(find_one("side_predictions.parquet"))
    frames, shards = [], {}
    for path in find_inputs("cache_train_series_*.npy"):
        suffix = pathlib.Path(path).stem.removeprefix("cache_")
        manifest_path = find_one(f"manifest_{suffix}.parquet")
        frame = pd.read_parquet(manifest_path).assign(cache_path=path)
        frames.append(frame)
        shards[path] = np.load(path, mmap_mode="r")
    assert len(frames) == 3, f"Expected three cache shards, found {len(frames)}"
    index = knee_dataset.StudyIndex(pd.concat(frames, ignore_index=True), series_frame,
                                   labels=train_labels, sides=sides)
    folded = pd.read_parquet(find_one("reconstructed_folds.parquet"))
    assert set(index.studies) == set(folded[config.ID_COL]), "Incomplete cache coverage"
    gold = splits.load_gold_uids(find_competition_file("train.csv"))
    splits.assert_no_gold_leakage(folded[config.ID_COL], gold)
    fold0 = next(m for m in explicit_members if m["spec"]["slices"] == 24 and m["metadata"]["fold"] == 0)
    reference = pd.read_parquet(find_one("reference_24_fold0.parquet")).sort_values(config.ID_COL).head(8)
    actual = np.stack([cached_prediction(fold0, uid) for uid in reference[config.ID_COL]])
    delta = float(np.max(np.abs(actual - reference[config.LABELS].to_numpy())))
    report["reference_24_fold0_max_abs_delta"] = delta
    print("Saved OOF reproduction max difference:", delta, flush=True)
    save_report()
    assert delta < .01, "Saved OOF was not reproduced; investigate input/checkpoint provenance"
    member = next(m for m in explicit_members if m["spec"]["slices"] == 24 and m["metadata"]["fold"] == 1)
    report["recovered_checkpoint_sha256"] = member["metadata"].get("sha256")
    report["recovered_input_spec"] = member["spec"]
    val_uids = sorted(folded.loc[folded.fold == 1, config.ID_COL])
    recovered = []
    start = time.monotonic()
    for uid in val_uids:
        if not has_time(600):
            break
        prediction = cached_prediction(member, uid)
        recovered.append({config.ID_COL: uid, "fold": 1, **dict(zip(config.LABELS, prediction.tolist()))})
        if len(recovered) % 50 == 0:
            pd.DataFrame(recovered).to_parquet(out / "oof_24_fold1_partial.parquet", index=False)
            print(f"Recovered fold 1: {len(recovered)}/{len(val_uids)}", flush=True)
    report["oof_studies"] = len(recovered)
    report["oof_expected"] = len(val_uids)
    report["oof_seconds"] = time.monotonic() - start
    report["oof_complete"] = len(recovered) == len(val_uids)
    if recovered:
        name = "oof_24_fold1.parquet" if report["oof_complete"] else "oof_24_fold1_partial.parquet"
        pd.DataFrame(recovered).to_parquet(out / name, index=False)
        (out / (name + ".json")).write_text(json.dumps({
            "checkpoint_sha256": member["metadata"].get("sha256"), "experiment": member["spec"],
            "fold": 1, "complete": report["oof_complete"], "val_uids": val_uids,
            "split_provenance": "reconstructed legacy split; training exclusion not independently verified"
        }, indent=2) + "\n")
        candidate = pd.read_parquet(find_one("candidate_oof_fold1.parquet")).set_index(config.ID_COL)
        fresh = pd.DataFrame(recovered).set_index(config.ID_COL)
        assert set(candidate.index) == set(val_uids), "Candidate OOF has incorrect fold membership"
        differences = np.abs(fresh[config.LABELS].to_numpy() - candidate.loc[fresh.index, config.LABELS].to_numpy())
        report["candidate_fold1_max_abs_delta"] = float(differences.max())
        report["candidate_fold1_mean_abs_delta"] = float(differences.mean())
        report["candidate_fold1_matches_within_0_01"] = bool(report["oof_complete"] and differences.max() < .01)
    save_report()

    # Use the actual submission decoder and model path, without generating submission.csv.
    series_meta = series_frame.set_index("SeriesInstanceUID")
    test_root = config.DATA_ROOT / "train_series"
    counts = series_frame.groupby(config.ID_COL).size().sort_index()
    largest = counts.sort_values(ascending=False, kind="stable").head(5).index.tolist()
    remainder = counts.drop(index=largest).sample(n=95, random_state=954).index.tolist()
    selected = largest + remainder
    report["raw_selected_uids"] = selected
    report["selection"] = "Five studies with most series plus 95 seeded random studies; not AUC validation"
    raw_predictions = []
    for uid in selected:
        if not has_time():
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
            assert probs.shape == (10, 12) and np.isfinite(probs).all()
            prediction = np.average(probs, axis=0, weights=member_weights)
            assert ((prediction >= 0) & (prediction <= 1)).all()
            raw_predictions.append({config.ID_COL: uid, **dict(zip(config.LABELS, prediction.tolist()))})
            report["raw_studies"].append({"uid": uid, "series": len(uids),
                "decode_side_seconds": decoded - start, "forward_seconds": finished - decoded,
                "total_seconds": finished - start})
            print(f"Raw benchmark {len(raw_predictions)}/100: {finished - start:.2f}s, {len(uids)} series", flush=True)
        except Exception:
            report["errors"].append({"uid": uid, "traceback": traceback.format_exc()})
            print(report["errors"][-1], flush=True)
            if len(report["errors"]) >= 3:
                break
        save_report()
    if raw_predictions:
        pd.DataFrame(raw_predictions).to_parquet(out / "raw_benchmark_predictions.parquet", index=False)
        seconds = np.array([r["total_seconds"] for r in report["raw_studies"]])
        report["raw_mean_seconds"] = float(seconds.mean())
        report["raw_p95_seconds"] = float(np.quantile(seconds, .95))
        report["projected_1300_study_hours"] = float((report["setup_seconds"] + 1300 * seconds.mean()) / 3600)
        report["runtime_caveat"] = "Training-study sample; hidden test composition and hardware contention can differ"
    report["raw_complete"] = len(raw_predictions) == 100 and not report["errors"]
    save_report()
    print(json.dumps({k: v for k, v in report.items() if k not in ["raw_studies", "raw_selected_uids"]}, indent=2), flush=True)
except Exception:
    report["fatal_error"] = traceback.format_exc()
    save_report()
    raise
finally:
    watchdog.cancel()
