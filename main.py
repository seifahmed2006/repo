"""CSE382 project CLI (Milestone 1).

  python main.py download                     fetch H2.csv
  python main.py analyze                      EDA tables + figures  -> outputs/eda/
  python main.py split                        show cleaning log and split sizes
  python main.py baseline --task regression|classification|clustering|all
  python main.py milestone1                   run everything above in order
"""
from __future__ import annotations
import argparse, json, sys
from src.utils import load_config, set_seed, out_dir


def main(argv=None):
    p = argparse.ArgumentParser(description="CSE382 City Hotel (H2) ML project")
    p.add_argument("command", choices=["download", "analyze", "split", "baseline", "milestone1"])
    p.add_argument("--task", choices=["regression", "classification", "clustering", "all"], default="all")
    p.add_argument("--config", default=None)
    a = p.parse_args(argv)
    cfg = load_config(a.config); set_seed(cfg["seed"])

    if a.command == "download":
        from src.data.loader import download_data
        print(download_data(cfg, force=True)); return
    if a.command == "analyze":
        from src.evaluation.eda import run_eda
        run_eda(cfg); return

    from src.data.loader import prepare_data
    splits, log = prepare_data(cfg)
    if a.command == "split":
        print(json.dumps(log, indent=2))
        json.dump(log, open(out_dir() / "cleaning_and_split_log.json", "w"), indent=2); return

    if a.command == "milestone1":
        from src.evaluation.eda import run_eda
        run_eda(cfg)
        json.dump(log, open(out_dir() / "cleaning_and_split_log.json", "w"), indent=2)
    tasks = ["regression", "classification", "clustering"] if a.command == "milestone1" or a.task == "all" else [a.task]
    if "regression" in tasks:
        from src.regression.baseline import run_regression_baseline; run_regression_baseline(splits, cfg)
    if "classification" in tasks:
        from src.classification.baseline import run_classification_baseline; run_classification_baseline(splits, cfg)
    if "clustering" in tasks:
        from src.clustering.baseline import run_clustering_baseline; run_clustering_baseline(splits, cfg)


if __name__ == "__main__":
    sys.exit(main())
