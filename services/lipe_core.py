import time
from services.config_loader import load_all
from services.ig_apply import apply_ig_command

class LIPE:
    """Living Intelligence Predictive Engine — orchestration layer."""

    def __init__(self):
        self.refresh_configs()
        self.metrics = {
            "NoiseIndex": None,
            "SyncState": None,
            "DecayScore": None,
            "TruthScore": None,
        }
        self.history = []

    # ---- config control
    def refresh_configs(self):
        self.cfg = load_all()

    # ---- metric updates
    def update_metric(self, key, value):
        if key not in self.metrics:
            raise KeyError(f"Unknown metric {key}")
        self.metrics[key] = value

    # ---- NBC logic
    def next_best_course(self):
        n = self.metrics["NoiseIndex"] or 0.5
        s = self.metrics["SyncState"] or "ALIGNED"
        d = self.metrics["DecayScore"] or 0.5
        t = self.metrics["TruthScore"] or 0.75

        actions = []
        if n > 0.7:  actions.append("EFB: tighten filter +1")
        if s == "CONFLICT": actions.append("CTPS: pause publishing")
        if d > 0.7:  actions.append("PDM+: retire stale patterns")
        if t < self.cfg["fir"]["fir"]["min_truth_score"]:
            actions.append("FIR: root-cause trace")
        if not actions: actions.append("Maintain current course")

        event = {
            "ts": int(time.time()),
            "metrics": self.metrics.copy(),
            "actions": actions,
        }
        self.history.append(event)
        return event

    # ---- IG wrapper
    def command(self, text):
        """Send an IG command directly through LIPE."""
        return apply_ig_command(text)
