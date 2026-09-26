import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data.processed import ItemData, SeqData, RecDataset
from data.utils import batch_to
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.model import EncoderDecoderRetrievalModel


p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--num-samples", type=int, default=22363)
p.add_argument("--batch-size", type=int, default=64)
a = p.parse_args()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(2026)

root = "dataset/amazon"
print("device:", device)
print("checkpoint:", a.checkpoint)

items = ItemData(
    root=root,
    dataset=RecDataset.AMAZON,
    force_process=False,
    split="beauty",
)

train = SeqData(
    root=root,
    dataset=RecDataset.AMAZON,
    is_train=True,
    subsample=True,
    split="beauty",
)

ev = SeqData(
    root=root,
    dataset=RecDataset.AMAZON,
    is_train=False,
    subsample=False,
    split="beauty",
)

n = min(a.num_samples, len(ev))
loader = DataLoader(
    Subset(ev, range(n)),
    batch_size=a.batch_size,
    shuffle=False,
)

tok = SemanticIdTokenizer(
    input_dim=768,
    hidden_dims=[512, 256, 128],
    output_dim=32,
    codebook_size=256,
    n_layers=3,
    n_cat_feats=0,
    rqvae_weights_path=(
        "trained_models/rqvae_amazon_beauty/"
        "checkpoint_high_entropy.pt"
    ),
    rqvae_codebook_normalize=False,
    rqvae_sim_vq=False,
).to(device)

print("precomputing corpus semantic IDs...")
tok.precompute_corpus_ids(items)

model = EncoderDecoderRetrievalModel(
    embedding_dim=128,
    attn_dim=512,
    dropout=0.3,
    num_heads=8,
    n_layers=8,
    num_embeddings=256,
    inference_verifier_fn=lambda x: tok.exists_prefix(x),
    sem_id_dim=tok.sem_ids_dim,
    max_pos=train.max_seq_len * tok.sem_ids_dim,
    jagged_mode=False,
).to(device)

ckpt = torch.load(
    a.checkpoint,
    map_location=device,
    weights_only=False,
)
model.load_state_dict(ckpt["model"])

print("loaded decoder checkpoint iter:", ckpt["iter"])

# Important: keep generation disabled so forward() returns token losses/logits.
model.eval()
model.enable_generation = False

D = tok.sem_ids_dim
loss_sum = torch.zeros(D, dtype=torch.float64)
correct = torch.zeros(D, dtype=torch.long)
count = torch.zeros(D, dtype=torch.long)

t0 = time.perf_counter()
seen = 0

with torch.no_grad():
    for batch in loader:
        data = batch_to(batch, device)
        x = tok(data)

        out = model(x)

        # In padded mode:
        # logits[:, :-1, :] corresponds to sem_ids_fut.
        logits = out.logits[:, :-1, :]
        target = x.sem_ids_fut

        if logits.shape[:2] != target.shape:
            raise RuntimeError(
                f"shape mismatch: logits={logits.shape}, "
                f"target={target.shape}"
            )

        per_token_loss = F.cross_entropy(
            logits.flatten(end_dim=1),
            target.flatten(end_dim=1),
            reduction="none",
            ignore_index=-1,
        ).reshape(target.shape)

        pred = logits.argmax(dim=-1)

        for d in range(D):
            valid = target[:, d] != -1
            c = int(valid.sum().item())

            if c > 0:
                loss_sum[d] += (
                    per_token_loss[:, d][valid]
                    .sum()
                    .detach()
                    .cpu()
                    .double()
                )
                correct[d] += (
                    (pred[:, d][valid] == target[:, d][valid])
                    .sum()
                    .detach()
                    .cpu()
                )
                count[d] += c

        seen += target.shape[0]

        if seen % 1000 == 0 or seen == n:
            print(f"processed {seen}/{n}", flush=True)


names = ["RQ1", "RQ2", "RQ3", "DEDUP"]

print("elapsed_seconds:", round(time.perf_counter() - t0, 2))
print("POSITION_WISE_METRICS")

for d in range(D):
    avg_loss = (loss_sum[d] / count[d]).item()
    accuracy = (correct[d].double() / count[d]).item()

    name = names[d] if d < len(names) else f"POS{d}"

    print(
        f"{d} {name}: "
        f"loss={avg_loss:.8f} "
        f"accuracy={accuracy:.8f} "
        f"count={count[d].item()}"
    )

print("TOKEN_LOSS_ANALYSIS=PASS")
