import os
import glob
from src.core.logger import ExperimentLogger

class LogAnalyzer:
    @staticmethod
    def analyze(target_id: str = None):
        log_dir = "experiments/logs"
        if not os.path.exists(log_dir):
            print("Error: Telemetry directory missing.")
            return
            
        log_files = glob.glob(os.path.join(log_dir, "*.json"))
        if not log_files:
            print("Error: No telemetry found.")
            return
            
        print("\nTELEMETRY ANALYSIS")
        print("="*40)
        
        for path in log_files:
            run_id = os.path.basename(path).replace(".json", "")
            
            if target_id and run_id != target_id:
                continue
                
            logger = ExperimentLogger(run_id=run_id, log_dir=log_dir)
            data = logger.get_run_data()
            
            config = data.get("config", {})
            metrics = data.get("metrics", {})
            dataset_version = config.get("dataset_version", "Unknown")
            
            print(f"\nAnalyzing Run: {run_id}")
            print(f"Dataset: {dataset_version}")
            
            insights = []
            
            # 1. Analyze LoRA Impact
            lora_cfg = config.get("lora", {})
            if lora_cfg.get("applied", False):
                insights.append("LoRA acceleration enabled: Hardware utilization expected to drop by 60%.")
                if lora_cfg.get("r", 8) > 16:
                    insights.append("Warning: LoRA Rank is >16. Potential for dimension overfitting.")
                    
            # 2. Analyze Epochs / Overfitting
            eval_loss = metrics.get("eval_loss", [])
            
            if len(eval_loss) >= 3:
                recent = [e["value"] for e in eval_loss[-3:]]
                if recent[-1] >= recent[-2]:
                    insights.append(f"Overfitting Indicator: Eval loss increasing at step {eval_loss[-1]['step']}.")
                elif (recent[-2] - recent[-1]) < 0.05:
                    insights.append(f"Diminishing Returns: Increasing epochs beyond {eval_loss[-1]['step']} yields minimal value.")
                    
            # 3. Analyze NLP Metrics
            rouge = metrics.get("rouge1", [])
            bleu = metrics.get("bleu_score", [])
            bertscore = metrics.get("bertscore", {})
            
            if rouge:
                val = rouge[-1].get("value", 0)
                if val < 0.3:
                    insights.append("Warning: Subpar ROUGE score. Model alignment likely failed.")
                elif val > 0.6:
                    insights.append("Note: Stable ROUGE score indicates successful lexical alignment.")
                    
            # 4. Integrate Semantic vs Lexical
            if bleu and bertscore.get("f1"):
                bleu_val = bleu[-1].get("value", 0)
                bert_f1 = bertscore.get("f1", 0)
                
                if bleu_val < 0.35 and bert_f1 > 0.85:
                    insights.append("High BERTScore F1 with low BLEU: Output is semantically extremely accurate despite deviating from ground-truth exact phrasing.")
                elif bleu_val < 0.35 and bert_f1 < 0.70:
                    insights.append("Low BERTScore and BLEU: Quality degradation. Model lacks both semantic and lexical alignment.")
                elif bert_f1 > 0.90:
                    insights.append(f"Exceptional semantic alignment (BERTScore F1: {bert_f1}). Model reliably captures objective intent.")

            # Append new insights
            if insights:
                data["insights"] = insights
                logger._save(data)
                
                print("Telemetry insights generated:")
                for i, text in enumerate(insights, 1):
                    print(f"  {i}. {text}")
            else:
                print("  No notable friction points detected.")
                
        print("\nTelemetry metadata updated.")
