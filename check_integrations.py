import torch
import yaml

from models.vjepa.vjepa_model import VisualExecutionModel
from models.vjepa.planning import MCTS


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("config/vjepa_micro.yaml")
    model = VisualExecutionModel(
        encoder_config=cfg["encoder"],
        predictor_config=cfg["predictor"],
        ema_momentum=cfg["training"].get("ema_momentum", 0.996),
        action_dim=cfg.get("action_dim", 128),
    )
    model.eval()

    bsz = 1
    t = cfg["encoder"].get("max_t", 8)
    h = cfg["encoder"].get("img_size", 64)
    w = cfg["encoder"].get("img_size", 64)
    video = torch.randn(bsz, t, 3, h, w)

    pt, ph, pw = cfg["encoder"]["patch_size"]
    seq_len = (t // pt) * (h // ph) * (w // pw)
    num_mask = max(1, seq_len // 4)
    mask = torch.randperm(seq_len)[:num_mask]

    batch = {
        "video": video,
        "mask": mask,
        "delta_t": torch.ones(bsz, 1),
        "action": torch.randn(bsz, cfg.get("action_dim", 128)),
    }

    out = model(batch)
    assert "predicted" in out and "target" in out and "value" in out

    mcts = MCTS(model=model, n_simulations=4)
    root_state = out["all_context"].mean(dim=1)
    actions = torch.randn(8, cfg.get("action_dim", 128))
    chosen = mcts.plan(root_state, actions)
    if isinstance(chosen, tuple):
        chosen = chosen[0]
    assert chosen.shape[-1] == cfg.get("action_dim", 128)

    print("Integration check passed: model forward + MCTS planning wired correctly.")


if __name__ == "__main__":
    main()
