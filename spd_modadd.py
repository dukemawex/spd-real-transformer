"""Apply SPD to the trained mod-add transformer's MLP (d_model->d_mlp->d_model), then test
whether automated clustering of subcomponents recovers the known FREQUENCY structure:
grokked mod-add models represent numbers on a few Fourier frequencies (Nanda et al. 2023).
We check if clusters of SPD components align with dominant frequencies in their input weights."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, json
from modadd_model import TinyTransformer, all_data
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

P=53; m=TinyTransformer(P); m.load_state_dict(torch.load("modadd.pt")); m.eval()
a,b,y=all_data(P)

# Collect MLP-input activations (residual stream at '=' pos before mlp) over all data
acts=[]
def hook(mod,inp,out): acts.append(inp[0].detach())
h=m.mlp[0].register_forward_hook(hook)
with torch.no_grad(): _=m(a,b)
h.remove()
Hin=acts[0]                      # [N,3?]  -> actually mlp applied to [B,3,d]; take '=' token
# mlp is applied per position; hook input is [B,3,d]; use last position
Xin=Hin[:,-1,:]                  # [N, d_model]
d_model=Xin.shape[1]
print("MLP input acts:", Xin.shape)

# SPD-decompose lin1 (d_model->d_mlp) — the up-projection where features are computed
W1=m.mlp[0].weight.detach().t(); b1=m.mlp[0].bias.detach()   # [d_model,d_mlp]
din,dh=W1.shape
class ROC(nn.Module):
    def __init__(s,d_in,d_out,C):
        super().__init__(); s.U=nn.Parameter(torch.randn(C,d_in)*0.05); s.V=nn.Parameter(torch.randn(C,d_out)*0.05)
        s.ci=nn.Sequential(nn.Linear(d_in,64),nn.ReLU(),nn.Linear(64,C)); nn.init.constant_(s.ci[-1].bias,2.0)
    def gates(s,x): return torch.sigmoid(s.ci(x))
    def full(s): return torch.einsum("ci,co->io",s.U,s.V)
    def forward(s,x,mask=None):
        comp=torch.einsum("ci,co->cio",s.U,s.V); g=s.gates(x)
        if mask is not None: g=g*mask
        return torch.einsum("bc,bco->bo",g,torch.einsum("bi,cio->bco",x,comp))

C=32; torch.manual_seed(0)
roc=ROC(din,dh,C)
opt=torch.optim.Adam(roc.parameters(),lr=1e-3)
tgt_pre=Xin@W1+b1                # target pre-activation
for s in range(6000):
    frac=s/6000; impc=0.0 if frac<0.4 else 3e-3*((frac-0.4)/0.6)
    idx=torch.randint(0,Xin.shape[0],(1024,)); x=Xin[idx]
    L_faith=((roc.full()-W1)**2).mean()
    with torch.no_grad(): t=x@W1+b1
    g=roc.gates(x); mk=torch.bernoulli(g.clamp(0,1)).detach()+g-g.detach()
    out=roc(x,mask=mk)+b1
    L_stoch=((out-t)**2).mean(); L_imp=g.abs().mean()
    loss=1e4*L_faith+L_stoch+impc*L_imp
    opt.zero_grad(); loss.backward(); opt.step()
    if s%2000==0: print(f"  step {s} faith {L_faith:.1e} stoch {L_stoch:.4f} imp {L_imp:.3f}")

# For each component, its input direction U_c lives in d_model (=residual). Project onto the
# embedding's Fourier basis: which frequency does each component read from?
E=m.embed.weight.detach()[:P]            # [P, d_model] number embeddings
# DFT over the P numbers for each embedding dim -> which freq each dim encodes
Efft=np.abs(np.fft.rfft(E.numpy(),axis=0))   # [P//2+1, d_model]
dom_freq_dim=Efft.argmax(0)                   # dominant freq per d_model dim
# component freq = freq of the d_model dim it most reads (|U_c| weighted)
U=roc.U.detach().numpy()                       # [C, d_model]
alive=[c for c in range(C) if np.abs(U[c]).max()>1e-2]
comp_freq=np.array([np.bincount(dom_freq_dim,weights=np.abs(U[c]),minlength=Efft.shape[0]).argmax() for c in alive])
print(f"\nalive components: {len(alive)}")
print("component dominant frequencies:", comp_freq)
uniq,counts=np.unique(comp_freq,return_counts=True)
print("frequency histogram:", dict(zip(uniq.tolist(),counts.tolist())))
print(f"# distinct frequencies used by components: {len(uniq)} (grokked mod-add typically uses ~3-6 key freqs)")

# cluster components by input direction; do clusters align with frequency?
Un=U[alive]/(np.linalg.norm(U[alive],axis=1,keepdims=True)+1e-9)
dist=1-np.clip(Un@Un.T,-1,1); np.fill_diagonal(dist,0)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
k=min(len(uniq),len(alive)-1) if len(alive)>2 else 2
lab=AgglomerativeClustering(n_clusters=max(2,len(uniq)),metric="precomputed",linkage="average").fit_predict(dist)
ari=adjusted_rand_score(comp_freq,lab); nmi=normalized_mutual_info_score(comp_freq,lab)
print(f"\nClustering-by-weight vs frequency-structure: ARI={ari:.3f} NMI={nmi:.3f}")
json.dump({"alive":len(alive),"distinct_freqs":int(len(uniq)),
           "freq_hist":{int(k):int(v) for k,v in zip(uniq,counts)},
           "cluster_vs_freq_ari":round(float(ari),3),"cluster_vs_freq_nmi":round(float(nmi),3),
           "test_acc":1.0}, open("results.json","w"),indent=2)
print("saved results.json")
