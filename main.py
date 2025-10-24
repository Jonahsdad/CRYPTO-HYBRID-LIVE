from services.config_loader import load_all
from services.ig_apply import apply_ig_command

if __name__ == "__main__":
    print("Loading configs...")
    cfg = load_all()
    print(cfg)

    print("\nApplying IG commands...")
    cmds = [
        "IG + ACTIVATE PDM+ [min_half_life=30d] [regime_awareness=on]",
        "IG + ACTIVATE CTPS [macro_weight=0.6] [event_sensitivity=high]",
        "IG + ACTIVATE FIR [min_truth_score=0.72] [quarantine_window=90m]"
    ]
    for c in cmds:
        print(apply_ig_command(c))

    print("\nReloading to verify...")
    cfg2 = load_all()
    print(cfg2)
