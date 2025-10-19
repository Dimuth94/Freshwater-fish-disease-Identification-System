import yaml, time
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from src.datamod import build_loaders
from src.utils import set_seed

def make_model(name, num_classes):
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_feat = m.fc.in_features
        m.fc = nn.Linear(in_feat, num_classes)
    else:
        raise ValueError("Unsupported model")
    return m

def train_one_epoch(model, loader, crit, opt, device, scaler):
    model.train(); loss_sum=0.0
    acc = MulticlassAccuracy(num_classes=len(loader.dataset.classes)).to(device)
    for x,y in tqdm(loader, desc="train", leave=False):
        x,y = x.to(device), y.to(device); opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=str(device).split(':')[0], dtype=torch.float16):
            out = model(x); loss = crit(out,y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        loss_sum += loss.item()*x.size(0); acc.update(out.argmax(1), y)
    return loss_sum/len(loader.dataset), acc.compute().item()

@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval(); loss_sum=0.0
    acc = MulticlassAccuracy(num_classes=len(loader.dataset.classes)).to(device)
    f1  = MulticlassF1Score(num_classes=len(loader.dataset.classes), average="macro").to(device)
    for x,y in tqdm(loader, desc="eval", leave=False):
        x,y = x.to(device), y.to(device)
        out = model(x); loss = crit(out,y)
        loss_sum += loss.item()*x.size(0)
        preds = out.argmax(1)
        acc.update(preds, y); f1.update(out, y)
    return loss_sum/len(loader.dataset), acc.compute().item(), f1.compute().item()

@torch.no_grad()
def test_report(model, loader, device, class_names):
    model.eval(); y_true=[]; y_pred=[]
    for x,y in tqdm(loader, desc="test", leave=False):
        out = model(x.to(device))
        y_true += y.tolist(); y_pred += out.argmax(1).cpu().tolist()
    print("\nClassification report:\n", classification_report(y_true,y_pred,target_names=class_names,digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true,y_pred))

def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    set_seed(cfg["seed"])
    device = torch.device(cfg.get("device","cuda") if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["log"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te = build_loaders(
        cfg["data"]["root"], cfg["data"]["img_size"], cfg["data"]["batch_size"], cfg["data"]["num_workers"]
    )
    model = make_model(cfg["train"]["model_name"], len(ds_tr.classes)).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sch = CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"], eta_min=cfg["train"]["lr"]*0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    best_val = 1e9; patience=cfg["train"]["patience"]; patience_ctr=0
    for ep in range(cfg["train"]["epochs"]):
        print(f"\nEpoch {ep+1}/{cfg['train']['epochs']}")
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, crit, opt, device, scaler)
        va_loss, va_acc, va_f1 = evaluate(model, dl_va, crit, device)
        sch.step()
        print(f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f}")

        if va_loss < best_val:
            best_val = va_loss; patience_ctr=0
            torch.save({"model":model.state_dict(),"classes":ds_tr.classes}, out_dir/"best.pt")
            print("Saved best checkpoint.")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping."); break

    ckpt = torch.load(out_dir/"best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    test_report(model, dl_te, device, ds_tr.classes)

if __name__ == "__main__":
    main()
