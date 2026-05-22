from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from radchar_class_config import load_label_schema
from radchar_model import DEVICE, SEQ_LEN, load_checkpoint_bundle
from radchar_training import RadCharDataset, load_encoded_h5, per_class_stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def predict_all(model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds, gts, confs = [], [], []
    for real, imag, labels in tqdm(loader, desc="测试集推理", dynamic_ncols=True, mininterval=0.05):
        real = real.to(DEVICE)
        imag = imag.to(DEVICE)
        logits = model(real, imag)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        conf = probs.gather(1, pred.unsqueeze(1)).squeeze(1)
        preds.append(pred.cpu().numpy())
        gts.append(labels.numpy())
        confs.append(conf.cpu().numpy())
    return np.concatenate(preds), np.concatenate(gts), np.concatenate(confs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RadChar 动态类别测试脚本")
    p.add_argument("--checkpoint", type=str, default=os.path.join(BASE_DIR, "models", "radchar_best.pth"))
    p.add_argument("--test_h5", type=str, default=os.path.join(BASE_DIR, "Bear_data", "RadChar-Test.h5"))
    p.add_argument("--class_config", type=str, default=os.path.join(BASE_DIR, "radchar_classes.json"))
    p.add_argument("--output", type=str, default=os.path.join(BASE_DIR, "results", "radchar_test_detail.csv"))
    p.add_argument("--class_stats_csv", type=str, default=os.path.join(BASE_DIR, "results", "radchar_test_per_class.csv"))
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model, schema, _, _ = load_checkpoint_bundle(args.checkpoint, config_path=args.class_config, strict=False)
    schema = load_label_schema(args.class_config, discovered_labels=schema.raw_labels, auto_create=False)
    data = load_encoded_h5(args.test_h5, schema)
    if data.iq.shape[1] != SEQ_LEN:
        raise ValueError(f"期望 iq 长度 {SEQ_LEN}，实际为 {data.iq.shape[1]}")

    loader = DataLoader(
        RadCharDataset(data.iq, data.labels, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    preds, gts, confs = predict_all(model, loader)
    raw_preds = schema.decode(preds)
    raw_gts = schema.decode(gts)
    acc = 100.0 * np.mean(preds == gts)
    print(f"整体准确率: {acc:.2f}%")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "index",
                "true_model_label",
                "true_raw_label",
                "true_class_name",
                "pred_model_label",
                "pred_raw_label",
                "pred_class_name",
                "correct",
                "pred_confidence",
            ]
        )
        for i in range(len(preds)):
            w.writerow(
                [
                    i,
                    int(gts[i]),
                    int(raw_gts[i]),
                    schema.name_for_model_index(int(gts[i])),
                    int(preds[i]),
                    int(raw_preds[i]),
                    schema.name_for_model_index(int(preds[i])),
                    1 if int(preds[i]) == int(gts[i]) else 0,
                    float(confs[i]),
                ]
            )
    print(f"逐条结果已保存到: {os.path.abspath(args.output)}")

    stats = per_class_stats(gts, preds, schema)
    os.makedirs(os.path.dirname(args.class_stats_csv) or ".", exist_ok=True)
    with open(args.class_stats_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model_label",
                "raw_label",
                "class_name",
                "num_samples_true",
                "num_correct_within_class",
                "accuracy_within_true_class_pct",
                "num_predicted_as_class",
                "precision_when_predicted_as_class_pct",
            ],
        )
        w.writeheader()
        for row in stats:
            safe = dict(row)
            for key in ("accuracy_within_true_class_pct", "precision_when_predicted_as_class_pct"):
                value = safe[key]
                safe[key] = "" if np.isnan(float(value)) else round(float(value), 6)
            w.writerow(safe)
    print(f"分类统计已保存到: {os.path.abspath(args.class_stats_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
