# =============================================================================
# 手写 ResNet18 训练 EuroSAT（Kaggle GPU 版，单单元格自包含）
# 若右侧 Internet 已开启且未上传 npz：自动从 HuggingFace 下载数据
# 若已上传 eurosat_rgb.npz 为 Kaggle Dataset：自动读取，无需联网
# =============================================================================
import os, json, time, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- 0. 配置 ----------------
CFG = dict(epochs=int(os.environ.get("EPOCHS", "30")), batch_size=128, lr=0.05,
           weight_decay=5e-4, seed=42, num_classes=10)
OUT = os.environ.get("OUT_DIR") or ("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
print("OUT:", OUT)

random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])

# ---------------- 1. 数据 ----------------
npz_path = os.environ.get("EUROSAT_NPZ")
if not npz_path and os.path.isdir("/kaggle/input"):
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "eurosat_rgb.npz" in files:
            npz_path = os.path.join(root, "eurosat_rgb.npz")
            break

CLASSES_DEFAULT = ["Annual Crop", "Forest", "Herbaceous Vegetation", "Highway", "Industrial Buildings",
                   "Pasture", "Permanent Crop", "Residential Buildings", "River", "SeaLake"]

if npz_path and os.path.exists(npz_path):
    print("load npz:", npz_path)
    z = np.load(npz_path)
    X, y, split_idx = z["X"], z["y"].astype(np.int64), z["split_idx"]
    cj = os.path.join(os.path.dirname(os.path.abspath(npz_path)), "classes.json")
    classes = json.load(open(cj, encoding="utf-8"))["classes"] if os.path.exists(cj) else CLASSES_DEFAULT
else:
    print("download from HuggingFace ...")
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("请先运行: !pip install -q datasets 并开启 Internet")
    ds = load_dataset("blanchon/EuroSAT_RGB")
    classes = list(ds["train"].features["label"].names)
    order = [("train", 0), ("validation", 1), ("test", 2)]
    Xs, ys, ss = [], [], []
    for sp, code in order:
        d = ds[sp]
        for i in range(len(d)):
            ex = d[i]
            Xs.append(np.asarray(ex["image"].convert("RGB"), dtype=np.uint8))
            ys.append(int(ex["label"])); ss.append(code)
    X = np.stack(Xs); y = np.array(ys, dtype=np.int64); split_idx = np.array(ss, dtype=np.int8)
    del ds

print("X:", X.shape, "classes:", len(classes))
tr = np.where(split_idx == 0)[0]; va = np.where(split_idx == 1)[0]; te = np.where(split_idx == 2)[0]
LIMIT = int(os.environ.get("LIMIT", "0"))
LIMIT_TEST = int(os.environ.get("LIMIT_TEST", "0"))
_rng = np.random.default_rng(CFG["seed"])
if LIMIT:
    tr = np.sort(_rng.choice(tr, size=min(LIMIT, len(tr)), replace=False))
    va = np.sort(_rng.choice(va, size=min(max(64, LIMIT // 5), len(va)), replace=False))
if LIMIT_TEST:
    te = np.sort(_rng.choice(te, size=min(LIMIT_TEST, len(te)), replace=False))
print("train/val/test:", len(tr), len(va), len(te))

mean = X[tr].astype(np.float32).mean(axis=(0, 1, 2)) / 255.0
std = X[tr].astype(np.float32).std(axis=(0, 1, 2)) / 255.0
print("mean", np.round(mean, 4), "std", np.round(std, 4))

Xg = torch.from_numpy(X).to(device)
yg = torch.from_numpy(y).to(device)
tr_t = torch.from_numpy(tr).to(device)
va_t = torch.from_numpy(va).to(device)
te_t = torch.from_numpy(te).to(device)
mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)

# ---------------- 2. 手写 ResNet18 ----------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class ResNet(nn.Module):
    def __init__(self, layers, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        in_c = 64; blocks = []
        for out_c, n in zip([64, 128, 256, 512], layers):
            for i in range(n):
                blocks.append(BasicBlock(in_c, out_c, 2 if i == 0 else 1))
                in_c = out_c
        self.body = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.body(self.stem(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

def resnet18(num_classes=10):
    return ResNet([2, 2, 2, 2], num_classes)

model = resnet18(len(classes)).to(device)
print("params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

# ---------------- 3. 训练 ----------------
crit = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=CFG["lr"], momentum=0.9, weight_decay=CFG["weight_decay"], nesterov=True)
steps = max(1, int(np.ceil(len(tr_t) / CFG["batch_size"])))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["lr"], epochs=CFG["epochs"], steps_per_epoch=steps)

def augment(xb):
    bh = torch.rand(xb.size(0), device=xb.device) < 0.5
    bv = torch.rand(xb.size(0), device=xb.device) < 0.5
    if bh.any(): xb[bh] = torch.flip(xb[bh], dims=[3])
    if bv.any(): xb[bv] = torch.flip(xb[bv], dims=[2])
    return xb

def run_batches(indices, train=True):
    order = torch.randperm(len(indices), device=device) if train else torch.arange(len(indices), device=device)
    total = correct = 0; loss_sum = 0.0
    bs = CFG["batch_size"]
    for i in range(0, len(order), bs):
        idx = indices[order[i:i + bs]]
        xb = Xg[idx].float().div_(255.0).permute(0, 3, 1, 2).contiguous()
        yb = yg[idx]
        if train: xb = augment(xb)
        xb = (xb - mean_t) / std_t
        if train:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward(); opt.step(); sched.step()
        else:
            with torch.no_grad():
                logits = model(xb)
                loss = crit(logits, yb)
        loss_sum += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return loss_sum / total, correct / total

best_acc = -1.0; rows = []; t_start = time.time()
for epoch in range(1, CFG["epochs"] + 1):
    t0 = time.time()
    model.train()
    tr_loss, tr_acc = run_batches(tr_t, train=True)
    model.eval()
    val_loss, val_acc = run_batches(va_t, train=False)
    rows.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=val_loss, val_acc=val_acc))
    improved = val_acc > best_acc
    if improved:
        best_acc = val_acc
        torch.save({"model": model.state_dict(), "val_acc": val_acc, "classes": classes},
                   os.path.join(OUT, "resnet18_best.pt"))
    print(f"[{epoch:02d}/{CFG['epochs']}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
          f"val loss={val_loss:.4f} acc={val_acc:.4f} ({time.time() - t0:.0f}s){' *best' if improved else ''}")

# ---------------- 4. 测试集评估 ----------------
model.eval()
test_loss, test_acc = run_batches(te_t, train=False)
print(f"TEST acc = {test_acc:.4f}  (best val = {best_acc:.4f}, 总耗时 {time.time() - t_start:.0f}s)")

preds = []
bs = CFG["batch_size"]
with torch.no_grad():
    for i in range(0, len(te_t), bs):
        idx = te_t[i:i + bs]
        xb = Xg[idx].float().div_(255.0).permute(0, 3, 1, 2).contiguous()
        xb = (xb - mean_t) / std_t
        preds.append(model(xb).argmax(1).cpu())
preds = torch.cat(preds).numpy()
y_true = yg[te_t].cpu().numpy()
K = len(classes)
cm = np.zeros((K, K), dtype=np.int64)
for t, p in zip(y_true, preds):
    cm[t, p] += 1
per = []
for i, c in enumerate(classes):
    tp = int(cm[i, i]); fp = int(cm[:, i].sum() - tp); fn = int(cm[i, :].sum() - tp)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    per.append(dict(class_name=c, precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4), support=int(cm[i, :].sum())))
pd.DataFrame(per).to_csv(os.path.join(OUT, "metrics_resnet18.csv"), index=False, encoding="utf-8")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "log_resnet18.csv"), index=False, encoding="utf-8")
json.dump(dict(test_acc=float(test_acc), best_val_acc=float(best_acc), per_class=per),
          open(os.path.join(OUT, "summary_resnet18.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ---------------- 5. 画图并保存 ----------------
log = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(log.epoch, log.train_loss, label="train"); ax[0].plot(log.epoch, log.val_loss, label="val")
ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(log.epoch, log.train_acc, label="train"); ax[1].plot(log.epoch, log.val_acc, label="val")
ax[1].set_title("Accuracy"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "train_curves.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(9, 8))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(K)); ax.set_yticks(range(K))
ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(classes, fontsize=8)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(f"Confusion Matrix (acc={test_acc:.4f})")
for i in range(K):
    for j in range(K):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=150); plt.close()

print("saved ->", OUT)
print(os.listdir(OUT))
