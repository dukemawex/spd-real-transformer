"""Real small Transformer trained on modular addition (a+b) mod P — the canonical
mech-interp task with a KNOWN ground-truth algorithm (Fourier/trig circuits; Nanda et al.
2023, 'Progress measures for grokking'). This is a genuine trained model, not a toy MLP."""
import torch, torch.nn as nn, torch.nn.functional as F, math

class TinyTransformer(nn.Module):
    def __init__(self, P=53, d_model=64, n_heads=4, d_mlp=128):
        super().__init__()
        self.P=P; self.d=d_model
        self.embed=nn.Embedding(P+1, d_model)     # tokens 0..P-1 plus '=' token = P
        self.pos=nn.Parameter(torch.randn(3,d_model)*0.02)
        self.attn=nn.MultiheadAttention(d_model,n_heads,batch_first=True)
        self.mlp=nn.Sequential(nn.Linear(d_model,d_mlp),nn.ReLU(),nn.Linear(d_mlp,d_model))
        self.unembed=nn.Linear(d_model,P,bias=False)
    def forward(self, a, b):
        eq=torch.full_like(a,self.P)
        x=torch.stack([a,b,eq],dim=1)             # [B,3]
        h=self.embed(x)+self.pos[None]
        att,_=self.attn(h,h,h)
        h=h+att
        h=h+self.mlp(h)
        return self.unembed(h[:,-1])              # predict from '=' position

def all_data(P):
    a=torch.arange(P).repeat_interleave(P); b=torch.arange(P).repeat(P)
    y=(a+b)%P
    return a,b,y

def train(P=53, steps=6000, d_model=64, seed=0, device="cpu"):
    torch.manual_seed(seed)
    m=TinyTransformer(P,d_model).to(device)
    a,b,y=all_data(P)
    n=len(a); idx=torch.randperm(n); tr=idx[:int(0.8*n)]; te=idx[int(0.8*n):]
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1.0)
    for s in range(steps):
        logits=m(a[tr],b[tr]); loss=F.cross_entropy(logits,y[tr])
        opt.zero_grad(); loss.backward(); opt.step()
        if s%1500==0:
            with torch.no_grad():
                acc=(m(a[te],b[te]).argmax(1)==y[te]).float().mean().item()
            print(f"  step {s:5d} loss {loss.item():.4f} test_acc {acc:.3f}")
    with torch.no_grad():
        acc=(m(a[te],b[te]).argmax(1)==y[te]).float().mean().item()
    return m,acc

if __name__=="__main__":
    m,acc=train()
    print(f"Trained tiny transformer on mod-{53} addition. Test acc={acc:.3f}")
    torch.save(m.state_dict(),"modadd.pt")
