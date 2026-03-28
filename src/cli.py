import argparse
from src.core.config import ExperimentConfig
from src.training.runner import ExperimentRunner
from src.evaluation.evaluator import Evaluator
from src.insights.analyzer import LogAnalyzer
from src.insights.comparator import RunComparator

def main():
    parser = argparse.ArgumentParser(description="LLM Fine-Tuning ML System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Start a training experiment via YAML.")
    train_parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Compute metrics and outputs for a completed run.")
    eval_parser.add_argument("--run-id", type=str, required=True, help="Target experiment identity")

    # Insights command
    insights_parser = subparsers.add_parser("insights", help="Parse local telemetry to generate decision logic.")
    insights_parser.add_argument("--run-id", type=str, required=False, help="Specific experiment identity. Analyzes all if omitted.")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate a comprehensive decision-support report for a run.")
    report_parser.add_argument("--run-id", type=str, required=True, help="Target experiment identity")

    # Compare command
    comp_parser = subparsers.add_parser("compare", help="Diff two experiments via telemetry logs.")
    comp_parser.add_argument("--runs", type=str, nargs=2, required=True, help="Pass exactly two Run IDs")

    args = parser.parse_args()

    if args.command == "train":
        cfg = ExperimentConfig.load_from_yaml(args.config)
        runner = ExperimentRunner(cfg)
        runner.run()

    elif args.command == "evaluate":
        evaluator = Evaluator(args.run_id)
        evaluator.evaluate()

    elif args.command == "insights":
        target = getattr(args, "run_id", None)
        LogAnalyzer.analyze(target)

    elif args.command == "report":
        from src.report import ReportGenerator
        ReportGenerator.generate(args.run_id)

    elif args.command == "compare":
        RunComparator.compare(args.runs[0], args.runs[1])

if __name__ == "__main__":
    main()
