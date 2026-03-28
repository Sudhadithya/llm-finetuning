import json
import os
import time
from typing import Any, Dict
from src.core.config import ExperimentConfig

class ExperimentLogger:
    def __init__(self, run_id: str, log_dir: str = "experiments/logs"):
        self.run_id = run_id
        self.log_dir = log_dir
        self.log_path = os.path.join(self.log_dir, f"{self.run_id}.json")
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        if not os.path.exists(self.log_path):
            self._save({
                "run_id": self.run_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "config": {},
                "metrics": {},
                "insights": []
            })
            
    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                return json.load(f)
        return {}

    def _save(self, data: Dict[str, Any]):
        with open(self.log_path, 'w') as f:
            json.dump(data, f, indent=4)

    def log_config(self, config: ExperimentConfig):
        data = self._load()
        data["config"] = config.model_dump()
        self._save(data)

    def log_metric(self, key: str, value: float, step: int = None):
        data = self._load()
        if key not in data["metrics"]:
            data["metrics"][key] = []
            
        entry = {"value": value}
        if step is not None:
            entry["step"] = step
            
        data["metrics"][key].append(entry)
        self._save(data)
        
    def log_insight(self, insight: str):
        data = self._load()
        data["insights"].append(insight)
        self._save(data)
        
    def log_hardware_metrics(self, duration_sec: float, gpu_name: str, cost_per_hour: float = 0.35):
        data = self._load()
        estimated_cost = (duration_sec / 3600) * cost_per_hour
        
        data["hardware"] = {
            "training_time_sec": round(duration_sec, 2),
            "estimated_cost_usd": round(estimated_cost, 4),
            "gpu": gpu_name
        }
        self._save(data)
        
    def get_run_data(self) -> Dict[str, Any]:
        return self._load()
