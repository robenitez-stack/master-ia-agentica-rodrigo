from pathlib import Path
import nbformat as nbf

TEMA2=Path(__file__).resolve().parents[1]
OUTPUT=TEMA2/"03-redes-multicapa-convolucionales-vision"/"notebooks"/"05_cnn_mnist_pytorch_mejorada.ipynb"
def md(x):return nbf.v4.new_markdown_cell(x.strip())
def code(x):return nbf.v4.new_code_cell(x.strip())
cells=[
md("""# CNN mejorada sobre MNIST con PyTorch

Modelo profundo con BatchNorm, Dropout, augmentation exclusivo de train, scheduler, checkpoint y early stopping. Test se evalúa una sola vez al final.

> Dependencias: `requirements/common.txt` y `requirements/pytorch.txt`."""),
md("## 1. Configuración reproducible"),
code(r"""
import copy,gzip,json,os,platform,random,struct,subprocess,sys,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score,classification_report,confusion_matrix,f1_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
RUN_MODE=os.getenv("RUN_MODE","full").lower()
if RUN_MODE not in {"fast","full"}:raise ValueError("RUN_MODE inválido")
SEED=42;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True,warn_only=True)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends,"mps") and torch.backends.mps.is_available() else "cpu")
def localizar():
 c=Path.cwd().resolve()
 for p in [c,c/"tema-02-machine-learning",*c.parents]:
  if p.name=="tema-02-machine-learning" and (p/".tema2-root").exists():return p
 raise FileNotFoundError
ROOT=localizar();DATA=ROOT/".data"/"mnist";RESULTS=ROOT/"03-redes-multicapa-convolucionales-vision"/"results"/"05_cnn_mnist_pytorch_mejorada"
RUN=RESULTS/"experiments"/RUN_MODE/"mnist_cnn_improved";DATA.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
PUBLISHABLE=RUN_MODE=="full";print(RUN_MODE,PUBLISHABLE,DEVICE,torch.__version__)
"""),
md("## 2. Datos: augmentation solo en train"),
code(r"""
URLS={"train-images-idx3-ubyte.gz":"https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz":"https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz","t10k-images-idx3-ubyte.gz":"https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz":"https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"}
def get(n,u):
 p=DATA/n
 if not p.exists():t=p.with_suffix(".tmp");urllib.request.urlretrieve(u,t);t.replace(p)
 return p
def imgs(p):
 with gzip.open(p,"rb") as f:_,n,r,c=struct.unpack(">IIII",f.read(16));return np.frombuffer(f.read(),dtype=np.uint8).reshape(n,r,c)
def labs(p):
 with gzip.open(p,"rb") as f:_,n=struct.unpack(">II",f.read(8));return np.frombuffer(f.read(),dtype=np.uint8)
p={n:get(n,u) for n,u in URLS.items()};Xa=imgs(p["train-images-idx3-ubyte.gz"]);ya=labs(p["train-labels-idx1-ubyte.gz"]);Xt=imgs(p["t10k-images-idx3-ubyte.gz"]);yt=labs(p["t10k-labels-idx1-ubyte.gz"])
Xtr,Xv,ytr,yv=train_test_split(Xa,ya,test_size=.15,stratify=ya,random_state=SEED)
def limit(X,y,n,s):
 if len(y)<=n:return X,y
 a,_,b,_=train_test_split(X,y,train_size=n,stratify=y,random_state=s);return a,b
if RUN_MODE=="fast":Xtr,ytr=limit(Xtr,ytr,5000,SEED);Xv,yv=limit(Xv,yv,1200,SEED+1);Xt,yt=limit(Xt,yt,1500,SEED+2)
train_tf=transforms.Compose([transforms.RandomRotation(10),transforms.RandomAffine(0,translate=(.1,.1)),transforms.ToTensor()])
eval_tf=transforms.ToTensor()
class ArrayDS(Dataset):
 def __init__(self,X,y,transform):self.X,self.y,self.transform=X,y,transform
 def __len__(self):return len(self.y)
 def __getitem__(self,i):return self.transform(Image.fromarray(self.X[i])),int(self.y[i])
tr,va,te=ArrayDS(Xtr,ytr,train_tf),ArrayDS(Xv,yv,eval_tf),ArrayDS(Xt,yt,eval_tf)
def loader(ds,shuffle,s):return DataLoader(ds,batch_size=128,shuffle=shuffle,generator=torch.Generator().manual_seed(s),num_workers=0,pin_memory=DEVICE.type=="cuda")
print(len(tr),len(va),len(te))
"""),
md("## 3. Arquitectura afinada"),
code(r"""
class ImprovedCNN(nn.Module):
 def __init__(self):
  super().__init__()
  self.features=nn.Sequential(
   nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.Conv2d(32,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),nn.Dropout(.25),
   nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),nn.Dropout(.25))
  self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(64*7*7,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(.5),nn.Linear(128,10))
 def forward(self,x):return self.classifier(self.features(x))
model=ImprovedCNN().to(DEVICE);criterion=nn.CrossEntropyLoss();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=.5,patience=1,min_lr=1e-5)
parameters=sum(p.numel() for p in model.parameters());print(model);print(parameters)
"""),
md("## 4. Entrenamiento, scheduler y early stopping por validation"),
code(r"""
def evaluate(m,dl):
 m.eval();ls=0.;true=[];pred=[]
 with torch.no_grad():
  for X,y in dl:X,y=X.to(DEVICE),y.to(DEVICE);z=m(X);ls+=criterion(z,y).item()*len(y);true.extend(y.cpu().numpy());pred.extend(z.argmax(1).cpu().numpy())
 true,pred=np.asarray(true),np.asarray(pred);return {"loss":ls/len(true),"accuracy":accuracy_score(true,pred),"f1":f1_score(true,pred,average="macro"),"true":true,"pred":pred}
epochs=2 if RUN_MODE=="fast" else 25;patience=4;wait=0;best=np.inf;best_epoch=0;state=None;hist=[];start=time.perf_counter()
train_loader,val_loader=loader(tr,True,SEED),loader(va,False,SEED)
for epoch in range(1,epochs+1):
 model.train();loss_sum=0.;true=[];pred=[]
 for X,y in train_loader:
  X,y=X.to(DEVICE),y.to(DEVICE);optimizer.zero_grad(set_to_none=True);z=model(X);loss=criterion(z,y);loss.backward();optimizer.step()
  loss_sum+=loss.item()*len(y);true.extend(y.cpu().numpy());pred.extend(z.argmax(1).detach().cpu().numpy())
 val=evaluate(model,val_loader);scheduler.step(val["loss"])
 row={"epoch":epoch,"train_loss":loss_sum/len(tr),"validation_loss":val["loss"],"train_accuracy":accuracy_score(true,pred),"validation_accuracy":val["accuracy"],"validation_macro_f1":val["f1"],"learning_rate":optimizer.param_groups[0]["lr"]};hist.append(row)
 if val["loss"]<best:best=val["loss"];best_epoch=epoch;state=copy.deepcopy(model.state_dict());wait=0
 else:wait+=1
 print(epoch,row)
 if RUN_MODE=="full" and wait>=patience:print("Early stopping");break
seconds=time.perf_counter()-start;model.load_state_dict(state);history=pd.DataFrame(hist)
"""),
md("## 5. Test final y artefactos"),
code(r"""
out=evaluate(model,loader(te,False,SEED));y_true,y_pred=out["true"],out["pred"];report=pd.DataFrame(classification_report(y_true,y_pred,output_dict=True,zero_division=0)).T
metrics={"experiment_id":"mnist_cnn_improved","task":"multiclass_classification","classes":[str(i) for i in range(10)],"train_samples":len(tr),"validation_samples":len(va),"test_samples":len(te),"epochs_or_iterations_completed":len(history),"best_epoch_or_iteration":best_epoch,"parameters":parameters,"training_seconds":seconds,"best_validation_accuracy":float(history.loc[best_epoch-1,"validation_accuracy"]),"test_accuracy":accuracy_score(y_true,y_pred),"test_precision":precision_score(y_true,y_pred,average="macro",zero_division=0),"test_recall":recall_score(y_true,y_pred,average="macro",zero_division=0),"test_f1":f1_score(y_true,y_pred,average="macro",zero_division=0),"test_macro_precision":precision_score(y_true,y_pred,average="macro",zero_division=0),"test_macro_recall":recall_score(y_true,y_pred,average="macro",zero_division=0),"test_macro_f1":f1_score(y_true,y_pred,average="macro",zero_division=0),"test_balanced_accuracy":recall_score(y_true,y_pred,average="macro"),"test_specificity":None}
(RUN/"metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding="utf-8");history.to_csv(RUN/"training_history.csv",index=False);report.to_csv(RUN/"classification_report.csv");display(pd.Series(metrics,name="valor").to_frame());display(report)
"""),
code(r"""
def save(fig,n):fig.savefig(RUN/n,dpi=140,bbox_inches="tight");plt.close(fig)
fig,a=plt.subplots(1,2,figsize=(12,4));a[0].plot(history.epoch,history.train_loss,label="train");a[0].plot(history.epoch,history.validation_loss,label="validation");a[0].legend();a[1].plot(history.epoch,history.train_accuracy,label="train");a[1].plot(history.epoch,history.validation_accuracy,label="validation");a[1].legend();save(fig,"learning_curves.png")
cm=confusion_matrix(y_true,y_pred);fig,ax=plt.subplots(figsize=(8,7));ConfusionMatrixDisplay(cm).plot(cmap="Blues",ax=ax,values_format="d");save(fig,"confusion_matrix.png")
def examples(ids,title,n):
 fig,axs=plt.subplots(2,5,figsize=(11,5))
 for ax in axs.ravel():ax.axis("off")
 for ax,i in zip(axs.ravel(),ids[:10]):ax.imshow(Xt[i],cmap="gray");ax.set_title(f"R={y_true[i]} P={y_pred[i]}");ax.axis("off")
 fig.suptitle(title);save(fig,n)
examples(np.flatnonzero(y_true==y_pred),"Correctos","sample_predictions.png");examples(np.flatnonzero(y_true!=y_pred),"Errores","misclassified_examples.png")
"""),
md("""## 6. Interpretación y comparación

La mejora combina profundidad, BatchNorm, Dropout, augmentation y gestión del learning rate; no es válido atribuir el cambio a una sola técnica. La comparación definitiva con MLP y CNN base se genera desde resultados `full`."""),
code(r"""
try:commit=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT.parent,capture_output=True,text=True,check=True).stdout.strip()
except Exception:commit="UNAVAILABLE"
summary={"notebook_id":"03-05","source_notebook":"3_red_convolucional_mnist_mejorada.ipynb","canonical_notebook":"05_cnn_mnist_pytorch_mejorada.ipynb","module":"03-redes-multicapa-convolucionales-vision","status":"COMPLETED","run_mode":RUN_MODE,"publishable":PUBLISHABLE,"seed":SEED,"device":str(DEVICE),"frameworks":["pytorch"],"dataset":"MNIST","run_timestamp_utc":datetime.now(timezone.utc).isoformat(),"git_commit":commit,"experiments":["mnist_cnn_improved"]}
(RESULTS/("run_summary.json" if RUN_MODE=="full" else "run_summary.fast.json")).write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
(RESULTS/("environment.txt" if RUN_MODE=="full" else "environment.fast.txt")).write_text(f"platform={platform.platform()}\npython={sys.version}\ntorch={torch.__version__}\ndevice={DEVICE}",encoding="utf-8");print(summary)
"""),
md("""## 7. Conclusiones

La CNN mejorada aplica regularización y augmentation con una metodología no contaminada. Conceptualmente puede inspirar visión artificial en Marina del Sol, pero no está lista para producción.""")
]
metadata={"kernelspec":{"display_name":"Python - Master IA Agentica","language":"python","name":"master-ia-agentica"},"language_info":{"name":"python","version":"3.10"}}
nb=nbf.v4.new_notebook(cells=cells,metadata=metadata);nbf.validate(nb);OUTPUT.parent.mkdir(parents=True,exist_ok=True);nbf.write(nb,OUTPUT);print(OUTPUT)
