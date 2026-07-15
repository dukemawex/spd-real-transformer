"""NEXT STEP (implemented): retrain the real mod-add transformer, then decompose its MLP with a
STRONGER annealed SPD (more components C=48, longer schedule) and test whether components now
recover MORE THAN ONE of the model's ~4 true frequencies (vs the collapse-to-1 in the README)."""
import torch,torch.nn as nn,torch.nn.functional as F,numpy as np,json
from modadd_model import TinyTransformer, all_data, train
m,acc=train(P=53,steps=6000,seed=0); print(f"retrained, acc={acc:.3f}")
P=53;a,b,y=all_data(P)
acts=[]
hh=m.mlp[0].register_forward_hook(lambda mod,i,o:acts.append(i[0].detach()))
with torch.no_grad():_=m(a,b)
hh.remove(); Xin=acts[0][:,-1,:]; d_model=Xin.shape[1]
W1=m.mlp[0].weight.detach().t();b1=m.mlp[0].bias.detach();din,dh=W1.shape
class ROC(nn.Module):
    def __init__(s,di,do,C):
        super().__init__();s.U=nn.Parameter(torch.randn(C,di)*0.05);s.V=nn.Parameter(torch.randn(C,do)*0.05)
        s.ci=nn.Sequential(nn.Linear(di,64),nn.ReLU(),nn.Linear(64,C));nn.init.constant_(s.ci[-1].bias,2.0)
    def gates(s,x):return torch.sigmoid(s.ci(x))
    def full(s):return torch.einsum("ci,co->io",s.U,s.V)
    def forward(s,x,mask=None):
        comp=torch.einsum("ci,co->cio",s.U,s.V);g=s.gates(x)
        if mask is not None:g=g*mask
        return torch.einsum("bc,bco->bo",g,torch.einsum("bi,cio->bco",x,comp))
C=48;torch.manual_seed(0);roc=ROC(din,dh,C);opt=torch.optim.Adam(roc.parameters(),lr=1e-3)
STEPS=12000
for s in range(STEPS):
    frac=s/STEPS;impc=0.0 if frac<0.4 else 1e-2*((frac-0.4)/0.6)
    idx=torch.randint(0,Xin.shape[0],(1024,));x=Xin[idx]
    Lf=((roc.full()-W1)**2).mean()
    with torch.no_grad():t=x@W1+b1
    g=roc.gates(x);mk=torch.bernoulli(g.clamp(0,1)).detach()+g-g.detach()
    Ls=((roc(x,mask=mk)+b1-t)**2).mean();Li=g.abs().mean()
    (1e4*Lf+Ls+impc*Li).backward();opt.step();opt.zero_grad()
E=m.embed.weight.detach()[:P].numpy();Efft=np.abs(np.fft.rfft(E,axis=0));domdim=Efft.argmax(0)
U=roc.U.detach().numpy();alive=[c for c in range(C) if np.abs(U[c]).max()>1e-2]
cf=np.array([np.bincount(domdim,weights=np.abs(U[c]),minlength=Efft.shape[0]).argmax() for c in alive])
uniq,counts=np.unique(cf,return_counts=True)
# true top frequencies
pw=(Efft**2).sum(1);pw[0]=0;truetop=set(pw.argsort()[::-1][:4].tolist())
recovered=truetop & set(uniq.tolist())
print(f"alive components: {len(alive)}")
print(f"component frequencies: {dict(zip(uniq.tolist(),counts.tolist()))}")
print(f"true top-4 freqs: {sorted(truetop)} | recovered by components: {sorted(recovered)} ({len(recovered)}/4)")
json.dump({"acc":round(acc,3),"C":C,"alive":len(alive),
           "distinct_component_freqs":int(len(uniq)),
           "true_top4":sorted(truetop),"recovered":sorted(recovered),
           "n_recovered":len(recovered)}, open("freq_recovery_results.json","w"),indent=2)
print("saved")
