from pathlib import Path
import nbformat as nbf

TEMA2 = Path(__file__).resolve().parents[1]
OUTPUT = TEMA2 / "03-redes-multicapa-convolucionales-vision" / "notebooks" / "03_cnn_mnist_pytorch_base.ipynb"

def md(text): return nbf.v4.new_markdown_cell(text.strip())
def code(text): return nbf.v4.new_code_cell(text.strip())

cells = [
md(r"""
# CNN base sobre MNIST con PyTorch

## Objetivo

Implementar una CNN pequeña y pedagógica: `Conv2D → ReLU → MaxPool → Flatten → Linear`. Se explican filtros, kernels, padding y pooling, usando validation para seleccionar el mejor checkpoint y test una sola vez al final.

> Dependencias: `requirements/common.txt` y `requirements/pytorch.txt`.
"""),
md("## 1. Configuración reproducible y modos `fast/full`"),
code(r"""
import copy, gzip, json, os, platform, random, struct, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

RUN_MODE = os.getenv("RUN_MODE", "full").lower()
if RUN_MODE not in {"fast", "full"}: raise ValueError("RUN_MODE debe ser 'fast' o 'full'")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

def localizar_tema2():
    cwd = Path.cwd().resolve()
    for p in [cwd, cwd / "tema-02-machine-learning", *cwd.parents]:
        if p.name == "tema-02-machine-learning" and (p / ".tema2-root").exists(): return p
    raise FileNotFoundError("No se localizó tema-02-machine-learning")

TEMA2_ROOT = localizar_tema2()
DATA_DIR = TEMA2_ROOT / ".data" / "mnist"
RESULTS_ROOT = TEMA2_ROOT / "03-redes-multicapa-convolucionales-vision" / "results" / "03_cnn_mnist_pytorch_base"
RUN_DIR = RESULTS_ROOT / "experiments" / RUN_MODE / "mnist_cnn_base"
DATA_DIR.mkdir(parents=True, exist_ok=True); RUN_DIR.mkdir(parents=True, exist_ok=True)
PUBLISHABLE = RUN_MODE == "full"
print(f"RUN_MODE={RUN_MODE} | publishable={PUBLISHABLE} | device={DEVICE} | torch={torch.__version__}")
"""),
md("""
## 2. Datos y particiones

El train oficial se divide estratificadamente en train y validation. Normalización: píxeles `[0,255] → [0,1]`. Test permanece aislado.
"""),
code(r"""
URLS = {
 "train-images-idx3-ubyte.gz":"https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
 "train-labels-idx1-ubyte.gz":"https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
 "t10k-images-idx3-ubyte.gz":"https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
 "t10k-labels-idx1-ubyte.gz":"https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"}
def descargar(n,u):
    p=DATA_DIR/n
    if not p.exists():
        t=p.with_suffix(p.suffix+".tmp"); urllib.request.urlretrieve(u,t); t.replace(p)
    return p
def imagenes(p):
    with gzip.open(p,"rb") as f:
        magic,n,r,c=struct.unpack(">IIII",f.read(16)); return np.frombuffer(f.read(),dtype=np.uint8).reshape(n,r,c)
def etiquetas(p):
    with gzip.open(p,"rb") as f:
        magic,n=struct.unpack(">II",f.read(8)); return np.frombuffer(f.read(),dtype=np.uint8)
paths={n:descargar(n,u) for n,u in URLS.items()}
X_all=imagenes(paths["train-images-idx3-ubyte.gz"]); y_all=etiquetas(paths["train-labels-idx1-ubyte.gz"])
X_test=imagenes(paths["t10k-images-idx3-ubyte.gz"]); y_test=etiquetas(paths["t10k-labels-idx1-ubyte.gz"])
X_train,X_val,y_train,y_val=train_test_split(X_all,y_all,test_size=.15,stratify=y_all,random_state=SEED)
def limitar(X,y,n,seed):
    if len(y)<=n:return X,y
    a,_,b,_=train_test_split(X,y,train_size=n,stratify=y,random_state=seed); return a,b
if RUN_MODE=="fast":
    X_train,y_train=limitar(X_train,y_train,5000,SEED)
    X_val,y_val=limitar(X_val,y_val,1200,SEED+1)
    X_test,y_test=limitar(X_test,y_test,1500,SEED+2)
def dataset(X,y):
    return TensorDataset(torch.from_numpy(X.astype(np.float32)/255).unsqueeze(1),torch.from_numpy(y.astype(np.int64)))
train_ds,val_ds,test_ds=dataset(X_train,y_train),dataset(X_val,y_val),dataset(X_test,y_test)
def loader(ds,shuffle,seed):
    return DataLoader(ds,batch_size=128,shuffle=shuffle,generator=torch.Generator().manual_seed(seed),num_workers=0,pin_memory=DEVICE.type=="cuda")
print(f"Train={len(train_ds)} | Validation={len(val_ds)} | Test={len(test_ds)}")
"""),
md("""
## 3. Arquitectura base

`Conv2D` aplica 16 filtros 3×3; `padding=1` conserva 28×28. ReLU introduce no linealidad, MaxPool reduce a 14×14, Flatten crea un vector y Linear produce diez logits. `CrossEntropyLoss` recibe logits, sin softmax previo.
"""),
code(r"""
class CNNBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(1,16,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(2)
        self.fc=nn.Linear(16*14*14,10)
    def forward(self,x):
        x=self.pool(torch.relu(self.conv(x)))
        return self.fc(torch.flatten(x,1))
model=CNNBase().to(DEVICE)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
parameters=sum(p.numel() for p in model.parameters())
print(model); print("Parámetros:",parameters)
"""),
md("## 4. Entrenamiento y selección por validation"),
code(r"""
def evaluar(model,data_loader):
    model.eval(); loss_sum=0.; true=[]; pred=[]
    with torch.no_grad():
        for X,y in data_loader:
            X,y=X.to(DEVICE),y.to(DEVICE); logits=model(X)
            loss_sum+=criterion(logits,y).item()*len(y)
            true.extend(y.cpu().numpy()); pred.extend(logits.argmax(1).cpu().numpy())
    true,pred=np.asarray(true),np.asarray(pred)
    return {"loss":loss_sum/len(true),"accuracy":accuracy_score(true,pred),"macro_f1":f1_score(true,pred,average="macro"),"true":true,"pred":pred}
epochs=2 if RUN_MODE=="fast" else 12
train_loader,val_loader=loader(train_ds,True,SEED),loader(val_ds,False,SEED)
history=[]; best_loss=np.inf; best_epoch=0; best_state=None; start=time.perf_counter()
for epoch in range(1,epochs+1):
    model.train(); total=0.; yt=[]; yp=[]
    for X,y in train_loader:
        X,y=X.to(DEVICE),y.to(DEVICE); optimizer.zero_grad(set_to_none=True)
        logits=model(X); loss=criterion(logits,y); loss.backward(); optimizer.step()
        total+=loss.item()*len(y); yt.extend(y.cpu().numpy()); yp.extend(logits.argmax(1).detach().cpu().numpy())
    val=evaluar(model,val_loader)
    row={"epoch":epoch,"train_loss":total/len(train_ds),"validation_loss":val["loss"],"train_accuracy":accuracy_score(yt,yp),"validation_accuracy":val["accuracy"],"validation_macro_f1":val["macro_f1"]}
    history.append(row)
    if row["validation_loss"]<best_loss:
        best_loss=row["validation_loss"]; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict())
    print(f"{epoch:02d}/{epochs} train_loss={row['train_loss']:.4f} val_loss={row['validation_loss']:.4f} val_acc={row['validation_accuracy']:.4f}")
training_seconds=time.perf_counter()-start
model.load_state_dict(best_state); history=pd.DataFrame(history)
"""),
md("## 5. Test final, métricas y artefactos"),
code(r"""
test=evaluar(model,loader(test_ds,False,SEED)); y_true,y_pred=test["true"],test["pred"]
report=pd.DataFrame(classification_report(y_true,y_pred,output_dict=True,zero_division=0)).T
metrics={"experiment_id":"mnist_cnn_base","task":"multiclass_classification","classes":[str(i) for i in range(10)],
"train_samples":len(train_ds),"validation_samples":len(val_ds),"test_samples":len(test_ds),
"epochs_or_iterations_completed":epochs,"best_epoch_or_iteration":best_epoch,"parameters":parameters,
"training_seconds":training_seconds,"best_validation_accuracy":float(history.loc[best_epoch-1,"validation_accuracy"]),
"test_accuracy":accuracy_score(y_true,y_pred),"test_precision":precision_score(y_true,y_pred,average="macro",zero_division=0),
"test_recall":recall_score(y_true,y_pred,average="macro",zero_division=0),"test_f1":f1_score(y_true,y_pred,average="macro",zero_division=0),
"test_macro_precision":precision_score(y_true,y_pred,average="macro",zero_division=0),"test_macro_recall":recall_score(y_true,y_pred,average="macro",zero_division=0),
"test_macro_f1":f1_score(y_true,y_pred,average="macro",zero_division=0),"test_balanced_accuracy":recall_score(y_true,y_pred,average="macro"),"test_specificity":None}
(RUN_DIR/"metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding="utf-8")
history.to_csv(RUN_DIR/"training_history.csv",index=False); report.to_csv(RUN_DIR/"classification_report.csv")
display(pd.Series(metrics,name="valor").to_frame()); display(report)
"""),
code(r"""
def guardar(fig,n): fig.savefig(RUN_DIR/n,dpi=140,bbox_inches="tight"); plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(12,4))
ax[0].plot(history.epoch,history.train_loss,label="train"); ax[0].plot(history.epoch,history.validation_loss,label="validation"); ax[0].axvline(best_epoch,ls="--",c="gray"); ax[0].set(title="Loss",xlabel="Época"); ax[0].legend()
ax[1].plot(history.epoch,history.train_accuracy,label="train"); ax[1].plot(history.epoch,history.validation_accuracy,label="validation"); ax[1].set(title="Accuracy",xlabel="Época"); ax[1].legend(); fig.tight_layout(); guardar(fig,"learning_curves.png")
cm=confusion_matrix(y_true,y_pred,labels=list(range(10))); fig,axis=plt.subplots(figsize=(8,7)); ConfusionMatrixDisplay(cm,display_labels=list(range(10))).plot(cmap="Blues",ax=axis,values_format="d"); guardar(fig,"confusion_matrix.png")
def ejemplos(indices,title,name):
    fig,axes=plt.subplots(2,5,figsize=(11,5))
    for a in axes.ravel():a.axis("off")
    for a,i in zip(axes.ravel(),indices[:10]):a.imshow(X_test[i],cmap="gray");a.set_title(f"R={y_true[i]} P={y_pred[i]}");a.axis("off")
    fig.suptitle(title);fig.tight_layout();guardar(fig,name)
ejemplos(np.flatnonzero(y_true==y_pred),"Correctos","sample_predictions.png")
ejemplos(np.flatnonzero(y_true!=y_pred),"Errores","misclassified_examples.png")
"""),
md("""
## 6. Interpretación

Esta CNN es deliberadamente pequeña. Underfitting aparece si train y validation siguen bajos; overfitting si se separan. La matriz identifica clases confundidas. El notebook mejorado añadirá profundidad, BatchNorm, Dropout y augmentation, por lo que cualquier mejora será multicausal.
"""),
code(r"""
try: git_commit=subprocess.run(["git","rev-parse","HEAD"],cwd=TEMA2_ROOT.parent,capture_output=True,text=True,check=True).stdout.strip()
except Exception: git_commit="UNAVAILABLE"
summary={"notebook_id":"03-03","source_notebook":"2_red_convolucional_mnist.ipynb","canonical_notebook":"03_cnn_mnist_pytorch_base.ipynb",
"module":"03-redes-multicapa-convolucionales-vision","status":"COMPLETED","run_mode":RUN_MODE,"publishable":PUBLISHABLE,
"seed":SEED,"device":str(DEVICE),"frameworks":["pytorch"],"dataset":"MNIST","run_timestamp_utc":datetime.now(timezone.utc).isoformat(),
"git_commit":git_commit,"experiments":["mnist_cnn_base"]}
(RESULTS_ROOT/("run_summary.json" if RUN_MODE=="full" else "run_summary.fast.json")).write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
(RESULTS_ROOT/("environment.txt" if RUN_MODE=="full" else "environment.fast.txt")).write_text("\n".join([f"platform={platform.platform()}",f"python={sys.version}",f"torch={torch.__version__}",f"numpy={np.__version__}",f"device={DEVICE}"]),encoding="utf-8")
print(json.dumps(summary,indent=2,ensure_ascii=False))
"""),
md("""
## 7. Conclusiones

La CNN base demuestra cómo filtros locales preservan estructura espacial con una arquitectura interpretable. Conceptualmente podría inspirar inspección visual en Marina del Sol, pero requiere datos reales, seguridad, monitorización y validación productiva. Este ejercicio académico no está listo para producción.
""")
]

metadata={"kernelspec":{"display_name":"Python - Master IA Agentica","language":"python","name":"master-ia-agentica"},"language_info":{"name":"python","version":"3.10"}}
nb=nbf.v4.new_notebook(cells=cells,metadata=metadata); nbf.validate(nb); OUTPUT.parent.mkdir(parents=True,exist_ok=True); nbf.write(nb,OUTPUT)
print(f"Notebook generado: {OUTPUT}")
