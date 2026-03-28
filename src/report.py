import os
from src.core.logger import ExperimentLogger
from src.insights.analyzer import LogAnalyzer

class ReportGenerator:
    @staticmethod
    def generate(run_id: str):
        logger = ExperimentLogger(run_id=run_id)
        data = logger.get_run_data()
        
        if not data:
            print(f"Error: Could not find telemetry for run_id '{run_id}'.")
            return
            
        # Trigger insight recalculation 
        LogAnalyzer.analyze(run_id)
        
        # Reload after insights are saved
        data = logger.get_run_data()
        config = data.get("config", {})
        metrics = data.get("metrics", {})
        hardware = data.get("hardware", {})
        failures = data.get("failures", [])
        insights = data.get("insights", [])
        
        print(f"\n📊 LLM Experiment Report — {run_id}")
        print("="*60)
        print(f"Dataset: {config.get('dataset_version', 'Unknown')}")
        print(f"Model: {config.get('model', 'Unknown')}")
        
        print("\n--- Performance ---")
        rouge = metrics.get("rouge1", [{}])[-1].get("value", "N/A")
        bleu = metrics.get("bleu_score", [{}])[-1].get("value", "N/A")
        print(f"BLEU: {bleu}")
        print(f"ROUGE: {rouge}")
        
        print("\n--- Semantic Performance ---")
        bertscore = metrics.get("bertscore", {})
        print(f"BERTScore F1: {bertscore.get('f1', 'N/A')}")
        
        print("\n--- Cost ---")
        print(f"Training Time: {hardware.get('training_time_sec', 'N/A')} seconds")
        print(f"GPU: {hardware.get('gpu', 'N/A')}")
        print(f"Estimated Cost: ${hardware.get('estimated_cost_usd', 'N/A')}")
        
        print("\n--- Failure Analysis ---")
        if failures:
            for fail in failures:
                print(f" * [Query]: '{fail['query']}'")
                print(f"   [Issue]: {fail['reason']}")
        else:
            print(" * No robust failure cases documented.")
            
        print("\n--- Key Insights ---")
        # Deduplicate insights securely
        dedup_insights = list(dict.fromkeys(insights))
        if dedup_insights:
            for insight in dedup_insights:
                print(f" * {insight}")
        else:
            print(" * No significant insights generated from thresholds.")
            
        print("\n--- Decision ---")
        cost = hardware.get("estimated_cost_usd", 0)
        bert_f1 = bertscore.get("f1", 0)
        
        if bert_f1 > 0.85 and cost < 0.20:
            print(" * DEPLOY: LoRA configuration")
            print(f" * JUSTIFICATION: Achieves high semantic accuracy (BERTScore F1: {bert_f1}) at minimal operational cost (${cost}).")
        elif cost > 0.50:
            print(" * HOLD: Configuration exceeds basic cost thresholds.")
            print(f" * JUSTIFICATION: Full fine-tuning is cost-prohibitive (${cost}) for the current performance delta. Evaluate PEFT alternatives.")
        else:
            print(" * HOLD: Insufficient distinct improvements.")
            print(" * JUSTIFICATION: Review thresholds before proceeding.")
            
        print("\n--- Limitations ---")
        if failures:
            for fail in failures:
                reason = fail.get('reason', '')
                if "Domain gap" in reason or "B2B" in reason:
                    print(" * Weak on long or enterprise-level queries.")
                    print(" * Dataset lacks B2B coverage.")
                elif "Length constraint" in reason:
                    print(" * Vulnerable to output length constraints causing hallucination.")
                else:
                    print(f" * {reason}")
        else:
            print(" * No critical limitations detected in current evaluation set.")
            
        print("\n--- Next Steps ---")
        if any("Domain gap" in f.get('reason', '') for f in failures):
            print(" * Expand dataset with enterprise support examples.")
        elif any("Length constraint" in f.get('reason', '') for f in failures):
            print(" * Implement recursive chunking in data pipeline.")
        else:
            print(" * Proceed to staging deployment.")
            
        print("="*60)
