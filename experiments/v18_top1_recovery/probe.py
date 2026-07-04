import json
from pathlib import Path

import torch


def main() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0]], requires_grad=True)
    selected = logits.topk(1, dim=-1).values
    routing_weight = selected.softmax(dim=-1)
    routing_weight.sum().backward()
    gradient = logits.grad.tolist()
    assert routing_weight.item() == 1.0
    assert not logits.grad.any()
    report = {
        "status": "router_only_recovery_rejected",
        "top1_weight": routing_weight.item(),
        "router_gradient": gradient,
        "reason": "softmax over one selected expert is constant, so router-only top-1 recovery has zero gradient",
        "required_next": "train non-router adapters or expert/transformer weights from a trainable checkpoint",
    }
    (Path(__file__).parent / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
