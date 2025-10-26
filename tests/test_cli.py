from xai_proj_b import cli


def test_cli_train_parser():
    parser = cli.build_parser()
    args = parser.parse_args(["train", "--config", "configs/cifar10_baseline.yaml"])
    assert args.command == "train"
    assert args.config.endswith("cifar10_baseline.yaml")


def test_cli_evaluate_parser():
    parser = cli.build_parser()
    args = parser.parse_args(["evaluate", "--config", "configs/cifar10_baseline.yaml", "--checkpoint", "runs/x.pt"])
    assert args.command == "evaluate"
    assert args.checkpoint == "runs/x.pt"
