"""
Airflow DAG: nightly maintenance for pr-rakshaka (M6).

    validate_labels ─┐
    run_unit_tests ──┼─> refresh_index (one task per repo, in parallel) ─> write_report
                     ┘

Airflow's job here: run the steps on a schedule, in order, retry a flaky one, run the per-repo
index refreshes side by side, and show every run in its web UI. The steps themselves are plain
functions in src/pipeline/nightly.py.

Run it locally (Airflow is NOT a dependency of pr-rakshaka; install it only to use this file):
    pip install apache-airflow
    export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/airflow/dags
    export PYTHONPATH=$(pwd)
    airflow standalone            # then open http://localhost:8080
    airflow dags test pr_rakshaka_nightly      # one run, no scheduler, prints each task's result

It never runs the evaluation (it needs model quota, and the sealed test sets are run once, by hand).
"""
from datetime import datetime, timedelta

try:                                   # Airflow 3
    from airflow.sdk import dag, task
except ImportError:                    # Airflow 2
    from airflow.decorators import dag, task

from src.pipeline import nightly


@dag(dag_id="pr_rakshaka_nightly", schedule="0 14 * * *",          # 01:00 in Sydney
     start_date=datetime(2026, 11, 1), catchup=False, tags=["pr-rakshaka"],
     default_args={"retries": 2, "retry_delay": timedelta(minutes=5)})
def pr_rakshaka_nightly():
    @task
    def validate_labels() -> dict:
        return nightly.validate_labels()

    @task
    def run_unit_tests() -> dict:
        return nightly.run_unit_tests()

    @task
    def list_repos() -> list:
        return nightly.watched_repos()

    @task
    def refresh_index(repo: str) -> dict:
        return nightly.refresh_index(repo)

    @task
    def write_report(index_results: list, label_summary: dict, tests: dict) -> str:
        return str(nightly.write_report(list(index_results), label_summary, tests))

    labels, tests = validate_labels(), run_unit_tests()
    indexed = refresh_index.expand(repo=list_repos())
    labels >> indexed
    tests >> indexed
    write_report(indexed, labels, tests)


pr_rakshaka_nightly()
