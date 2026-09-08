"""Task1 reproduction: kidney normal(0) vs abnormal(1), MONAI 3D DenseNet-121.
Input = (Z,128,128) kidney clips, resized to (32,128,128), intensity scaled to [0,1].
Split follows the xlsx Training / Validation assignment (task1_final.csv).
Reports validation AUC / accuracy per epoch; saves best-AUC checkpoint.
"""
import os, csv, argparse, numpy as np, SimpleITK as sitk, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import DenseNet121
from monai.transforms import Resize
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = "/data/qjy"
RESIZE = Resize(spatial_size=(32, 128, 128))


class ClipDS(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        path, lab = self.rows[i]
        a = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
        a = a / 255.0
        a = a[None]                       # add channel -> (1,Z,128,128)
        a = RESIZE(a)                     # -> (1,32,128,128)
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32), int(lab)


def load_rows(split):
    rows = []
    with open(f"{ROOT}/reviewer/repro/task1_final.csv") as f:
        for r in csv.DictReader(f):
            if r['split'] == split and os.path.exists(r['path']):
                rows.append((r['path'], int(r['label'])))
    return rows


def evaluate(net, loader, device):
    net.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            out = torch.softmax(net(x.to(device)), 1)[:, 1]
            ps += out.cpu().tolist(); ys += y.tolist()
    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float('nan')
    acc = accuracy_score(ys, [1 if p >= 0.5 else 0 for p in ps])
    return auc, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='1'); ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--bs', type=int, default=6); ap.add_argument('--lr', type=float, default=1e-4)
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    device = 'cuda'
    tr, va = load_rows('Training'), load_rows('Validation')
    print(f"train={len(tr)} valid={len(va)}  pos_train={sum(l for _,l in tr)} pos_valid={sum(l for _,l in va)}")
    trl = DataLoader(ClipDS(tr), batch_size=a.bs, shuffle=True, num_workers=8, drop_last=True)
    val = DataLoader(ClipDS(va), batch_size=a.bs, shuffle=False, num_workers=8)

    net = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    # class weights for imbalance
    n1 = sum(l for _, l in tr); n0 = len(tr) - n1
    w = torch.tensor([len(tr) / (2 * n0), len(tr) / (2 * n1)], dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs(f"{ROOT}/reviewer/repro/task1_ckpt", exist_ok=True)
    best = 0.0
    for ep in range(a.epochs):
        net.train(); tot = 0
        for x, y in trl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit(net(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        sch.step()
        auc, acc = evaluate(net, val, device)
        flag = ''
        if auc == auc and auc > best:
            best = auc; torch.save(net.state_dict(), f"{ROOT}/reviewer/repro/task1_ckpt/best.pt"); flag = ' *BEST*'
        print(f"ep{ep:02d} loss={tot/max(len(trl),1):.4f} val_AUC={auc:.4f} val_ACC={acc:.4f}{flag}", flush=True)
    print(f"DONE best_val_AUC={best:.4f}")


if __name__ == "__main__":
    main()
