"""Read competition metadata, rules and submission history; never submit."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiListCompetitionPagesRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/meta/kaggle_audit")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    competition = "rsna-knee-abnormality-detection"
    api = KaggleApi()
    api.authenticate()
    fields = ("ref", "title", "deadline", "new_entrant_deadline", "team_count",
              "max_daily_submissions", "evaluation_metric", "is_kernels_submissions_only")
    found = api.competitions_list(search=competition)
    competitions = getattr(found, "competitions", found)
    summary = {"checked_at": datetime.now(timezone.utc).isoformat(),
               "competitions": [{k: str(getattr(c, k, None)) for k in fields}
                                for c in competitions]}
    submissions = api.competition_submissions(competition, page_size=100)
    summary["submissions"] = [{k: str(getattr(s, k, None)) for k in
                               ("ref", "date", "status", "public_score", "private_score")}
                              for s in submissions]
    leaders = api.competition_leaderboard_view(competition, page_size=20)
    summary["leaderboard"] = [{k: str(getattr(s, k, None)) for k in
                               ("team_name", "score", "submission_date")}
                              for s in leaders]
    with api.build_kaggle_client() as client:
        request = ApiListCompetitionPagesRequest()
        request.competition_name = competition
        pages = client.competitions.competition_api_client.list_competition_pages(request).pages
    for page in pages:
        name = Path(page.name).name
        (out / (name + ".html")).write_text(page.content)
    summary["pages"] = [page.name for page in pages]
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
