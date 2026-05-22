from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from radchar_class_config import load_label_schema
from radchar_model import DEVICE, SEQ_LEN, load_checkpoint_bundle
from radchar_training import RadCharDataset, load_encoded_h5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ModelEnsemble:
    def __init__(self, model_paths: list[str], config_path: str):
        self.models = []
        self.schema = None
        
        for path in model_paths:
            model, schema, _, _ = load_checkpoint_bundle(path, config_path=config_path, strict=False)
            self.models.append(model)
            if self.schema is None:
                self.schema = schema
        
        print(f"已加载 {len(self.models)} 个模型用于集成")

    @torch.no_grad()
    def predict(self, real: torch.Tensor, imag: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        all_logits = []
        
        for model in self.models:
            logits = model(real, imag)
            all_logits.append(logits)
        
        # 平均所有模型的logits
        avg_logits = torch.stack(all_logits).mean(dim=0)
        probs = F.softmax(avg_logits, dim=1)
        pred = avg_logits.argmax(dim=1)
        conf = probs.gather(1, pred.unsqueeze(1)).squeeze(1)
        
        return pred.cpu().numpy(), conf.cpu().numpy()


@torch.no_grad()
def predict_all(ensemble: ModelEnsemble, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds, gts, confs = [], [], []
    for real, imag, labels in tqdm(loader, desc="测试集集成推理", dynamic_ncols=True, mininterval=0.05):
        real = real.to(DEVICE)
        imag = imag.to(DEVICE)
        pred, conf = ensemble.predict(real, imag)
        preds.append(pred)
        gts.append(labels.numpy())
        confs.append(conf)
    return np.concatenate(preds), np.concatenate(gts), np.concatenate(confs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RadChar 模型集成测试脚本")
    p.add_argument("--model_paths", type=str, nargs="+", default=[
        os.path.join(BASE_DIR, "models", "radchar_best.pth"),
        os.path.join(BASE_DIR, "models", "radchar_checkpoint.pth")
    ])
    p.add_argument("--test_h5", type=str, default=os.path.join(BASE_DIR, "Bear_data", "RadChar-Test.h5"))
    p.add_argument("--class_config", type=str, default=os.path.join(BASE_DIR, "radchar_classes.json"))
    p.add_argument("--output", type=str, default=os.path.join(BASE_DIR, "results", "radchar_ensemble_test_detail.csv"))
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensemble = ModelEnsemble(args.model_paths, args.class_config)
    schema = load_label_schema(args.class_config, discovered_labels=ensemble.schema.raw_labels, auto_create=False)
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

    preds, gts, confs = predict_all(ensemble, loader)
    raw_preds = schema.decode(preds)
    raw_gts = schema.decode(gts)
    acc = 100.0 * np.mean(preds == gts)
    print(f"集成模型整体准确率: {acc:.2f}%")

    # 保存结果
    import csv
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
    print(f"集成模型测试结果已保存到: {os.path.abspath(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
