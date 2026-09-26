import argparse, time
import torch
from torch.utils.data import DataLoader, Subset
from data.processed import ItemData, SeqData, RecDataset
from data.utils import batch_to
from evaluate.metrics import TopKAccumulator
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.model import EncoderDecoderRetrievalModel

p=argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--num-samples", type=int, default=128)
p.add_argument("--batch-size", type=int, default=2)
a=p.parse_args()

device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(2026)
root="dataset/amazon"
print("device:", device)
print("checkpoint:", a.checkpoint)

items=ItemData(root=root,dataset=RecDataset.AMAZON,force_process=False,split="beauty")
train=SeqData(root=root,dataset=RecDataset.AMAZON,is_train=True,subsample=True,split="beauty")
ev=SeqData(root=root,dataset=RecDataset.AMAZON,is_train=False,subsample=False,split="beauty")
n=min(a.num_samples,len(ev))
loader=DataLoader(Subset(ev,range(n)),batch_size=a.batch_size,shuffle=False)

tok=SemanticIdTokenizer(input_dim=768,hidden_dims=[512,256,128],output_dim=32,
 codebook_size=256,n_layers=3,n_cat_feats=0,
 rqvae_weights_path="trained_models/rqvae_amazon_beauty/checkpoint_high_entropy.pt",
 rqvae_codebook_normalize=False,rqvae_sim_vq=False).to(device)
print("precomputing corpus semantic IDs...")
tok.precompute_corpus_ids(items)

model=EncoderDecoderRetrievalModel(embedding_dim=128,attn_dim=512,dropout=0.3,
 num_heads=8,n_layers=8,num_embeddings=256,
 inference_verifier_fn=lambda x: tok.exists_prefix(x),
 sem_id_dim=tok.sem_ids_dim,max_pos=train.max_seq_len*tok.sem_ids_dim,
 jagged_mode=False).to(device)

ckpt=torch.load(a.checkpoint,map_location=device,weights_only=False)
model.load_state_dict(ckpt["model"])
print("loaded decoder checkpoint iter:",ckpt["iter"])
model.eval(); model.enable_generation=True

acc=TopKAccumulator(ks=[1,5,10])
t0=time.perf_counter(); seen=0
with torch.no_grad():
    for batch in loader:
        data=batch_to(batch,device); x=tok(data)
        g=model.generate_next_sem_id(x,top_k=True,temperature=1)
        acc.accumulate(actual=x.sem_ids_fut,top_k=g.sem_ids)
        seen += x.sem_ids_fut.shape[0]
        if seen % 100 == 0 or seen == n:
            print(f"processed {seen}/{n}", flush=True)

r=acc.reduce()
print("elapsed_seconds:",round(time.perf_counter()-t0,2))
print("FULL_ID_METRICS")
for k in (1,5,10):
    key=f"h@{k}_slice_:4"
    print(f"{key}: {r[key]:.8f}")
print("SUBSET_EVAL=PASS")
