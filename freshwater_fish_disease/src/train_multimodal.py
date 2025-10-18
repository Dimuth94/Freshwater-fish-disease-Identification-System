import yaml, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2, pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertModel
from tqdm import tqdm
from sklearn.metrics import classification_report

CFG = {
  "csv_train": "data/images_manifest_train.csv",  # build this from your dataset (image_path,label,symptom_text)
  "csv_val":   "data/images_manifest_val.csv",
  "csv_test":  "data/images_manifest_test.csv",
  "img_size": 224, "batch": 16, "epochs": 10, "lr": 2e-4, "max_len": 48, "device": "cuda" if torch.cuda.is_available() else "cpu",
  "out": "outputs_mm"
}

aug_tr = A.Compose([A.LongestMaxSize(CFG["img_size"]), A.PadIfNeeded(CFG["img_size"],CFG["img_size"]),
    A.RandomResizedCrop(CFG["img_size"],CFG["img_size"],scale=(0.85,1.0),p=0.7),
    A.HorizontalFlip(p=0.5),
    A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), ToTensorV2()])
aug_val = A.Compose([A.LongestMaxSize(CFG["img_size"]), A.PadIfNeeded(CFG["img_size"],CFG["img_size"]),
    A.CenterCrop(CFG["img_size"],CFG["img_size"]),
    A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), ToTensorV2()])

class MMDataset(Dataset):
    def __init__(self, csv_path, aug, tok, label2id):
        df = pd.read_csv(csv_path)
        self.items = df.to_dict("records"); self.aug=aug; self.tok=tok; self.label2id=label2id
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        r = self.items[i]
        img = cv2.imread(r["image_path"])[:,:,::-1]
        img = self.aug(image=img)["image"]
        lab = self.label2id[r["label"]]
        toks = self.tok(r.get("symptom_text",""), truncation=True, padding="max_length",
                        max_length=CFG["max_len"], return_tensors="pt")
        toks = {k:v.squeeze(0) for k,v in toks.items()}
        return img, toks, lab

class MMModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.img = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feat = self.img.classifier[1].in_features
        self.img.classifier[1] = nn.Identity()
        self.txt = DistilBertModel.from_pretrained("distilbert-base-uncased")
        fusion_dim = in_feat + self.txt.config.dim
        self.head = nn.Sequential(nn.Linear(fusion_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, n_classes))
    def forward(self, images, input_ids, attention_mask):
        z_img = self.img(images)
        z_txt = self.txt(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0,:]
        z = torch.cat([z_img, z_txt], dim=1)
        return self.head(z)

def run():
    Path(CFG["out"]).mkdir(parents=True, exist_ok=True)
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # build label mapping from train CSV
    df_tr = pd.read_csv(CFG["csv_train"]); labels = sorted(df_tr["label"].unique())
    label2id = {l:i for i,l in enumerate(labels)}; id2label = {i:l for l,i in label2id.items()}

    ds_tr = MMDataset(CFG["csv_train"], aug_tr, tok, label2id)
    ds_va = MMDataset(CFG["csv_val"],   aug_val, tok, label2id)
    ds_te = MMDataset(CFG["csv_test"],  aug_val, tok, label2id)

    dl_tr = DataLoader(ds_tr, batch_size=CFG["batch"], shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=CFG["batch"], shuffle=False, num_workers=2, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=CFG["batch"], shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device(CFG["device"])
    model = MMModel(n_classes=len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    for ep in range(CFG["epochs"]):
        model.train(); tl,correct,total=0.0,0,0
        for img, toks, lab in tqdm(dl_tr, desc=f"train {ep+1}"):
            img,lab = img.to(device), torch.tensor(lab).to(device)
            toks = {k:v.to(device) for k,v in toks.items()}
            opt.zero_grad(set_to_none=True)
            out = model(img, toks["input_ids"], toks["attention_mask"])
            loss = crit(out, lab); loss.backward(); opt.step()
            tl += loss.item()*img.size(0); correct += (out.argmax(1)==lab).sum().item(); total += img.size(0)
        va_correct, va_total = 0, 0
        model.eval()
        with torch.no_grad():
            for img, toks, lab in dl_va:
                img,lab = img.to(device), torch.tensor(lab).to(device)
                toks = {k:v.to(device) for k,v in toks.items()}
                out = model(img, toks["input_ids"], toks["attention_mask"])
                va_correct += (out.argmax(1)==lab).sum().item(); va_total += img.size(0)
        va_acc = va_correct/va_total
        print(f"val acc {va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model":model.state_dict(),"labels":labels}, f"{CFG['out']}/best_mm.pt")
            print("Saved best multimodal model.")
    # test
    ckpt = torch.load(f"{CFG['out']}/best_mm.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    y_true,y_pred=[],[]
    model.eval()
    with torch.no_grad():
        for img, toks, lab in dl_te:
            img = img.to(device); toks = {k:v.to(device) for k,v in toks.items()}
            out = model(img, toks["input_ids"], toks["attention_mask"])
            y_true += lab; y_pred += out.argmax(1).cpu().tolist()
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))

if __name__ == "__main__":
    run()
