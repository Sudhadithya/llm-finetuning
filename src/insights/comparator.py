from src.core.logger import ExperimentLogger

class RunComparator:
    @staticmethod
    def compare(run_a: str, run_b: str):
        logger_a = ExperimentLogger(run_id=run_a)
        logger_b = ExperimentLogger(run_id=run_b)
        
        data_a = logger_a.get_run_data()
        data_b = logger_b.get_run_data()
        
        if not data_a or not data_b:
            print("Error: Missing telemetry for one or both requested IDs.")
            return
            
        print(f"\nRUN COMPARISON: {run_a} vs {run_b}")
        print("-" * 60)
        
        config_a = data_a.get("config", {})
        config_b = data_b.get("config", {})
        
        metrics_a = data_a.get("metrics", {})
        metrics_b = data_b.get("metrics", {})
        
        print(f"| {'Metric/Config':<20} | {run_a:<15} | {run_b:<15} |")
        print("-" * 60)
        
        keys_to_compare = [
            ("Dataset", config_a.get("dataset_version", "N/A"), config_b.get("dataset_version", "N/A")),
            ("Model", config_a.get("model", "N/A"), config_b.get("model", "N/A")),
        ]
        
        for label, val_a, val_b in keys_to_compare:
            print(f"| {label:<20} | {str(val_a):<15} | {str(val_b):<15} |")
            
        print("-" * 60)
        
        def get_latest(metrics_dict, key):
            if key in metrics_dict and metrics_dict[key]:
                return metrics_dict[key][-1].get("value", "N/A")
            return "N/A"
            
        metric_keys = ["train_loss", "eval_loss", "rouge1", "bleu_score"]
        
        for key in metric_keys:
            v_a = get_latest(metrics_a, key)
            v_b = get_latest(metrics_b, key)
            
            trend = ""
            if isinstance(v_a, (int, float)) and isinstance(v_b, (int, float)):
                diff = v_b - v_a
                if "loss" in key:
                    trend = f"({diff:.2f} \u2193)" if diff < 0 else f"(+{diff:.2f} \u2191)"
                else:
                    trend = f"({diff:.2f} \u2191)" if diff > 0 else f"({diff:.2f} \u2193)"
                    
            print(f"| {key.upper():<20} | {str(v_a):<15} | {str(v_b):<15} | {trend}")
            
        # Extract Nested BERTScore
        v_a = metrics_a.get("bertscore", {}).get("f1", "N/A")
        v_b = metrics_b.get("bertscore", {}).get("f1", "N/A")
        trend = ""
        if isinstance(v_a, (int, float)) and isinstance(v_b, (int, float)):
            diff = v_b - v_a
            pct = (diff / v_a) * 100 if v_a else 0
            trend = f"(+{pct:.1f}% \u2191)" if diff > 0 else f"({pct:.1f}% \u2193)"
        print(f"| {'BERT_F1':<20} | {str(v_a):<15} | {str(v_b):<15} | {trend}")
            
        print("-" * 60)
        print("Comparison sequence concluded.")
