import torch
from torch.utils.data import DataLoader

from data.processed import ItemData, SeqData, RecDataset
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.model import EncoderDecoderRetrievalModel
from data.utils import batch_to


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RQ_PATH = "trained_models/rqvae_amazon_beauty/checkpoint_high_entropy.pt"
DECODER_PATH = "out/decoder/amazon_smoke/checkpoint_99.pt"
DATA_ROOT = "dataset/amazon"
BATCH_SIZE = 2

print(f"device: {DEVICE}")

item_dataset = ItemData(
    root=DATA_ROOT,
    dataset=RecDataset.AMAZON,
    force_process=False,
    split="beauty",
)
train_dataset = SeqData(
    root=DATA_ROOT,
    dataset=RecDataset.AMAZON,
    is_train=True,
    subsample=True,
    split="beauty",
)
eval_dataset = SeqData(
    root=DATA_ROOT,
    dataset=RecDataset.AMAZON,
    is_train=False,
    subsample=False,
    split="beauty",
)

tokenizer = SemanticIdTokenizer(
    input_dim=768,
    hidden_dims=[512, 256, 128],
    output_dim=32,
    codebook_size=256,
    n_layers=3,
    n_cat_feats=0,
    rqvae_weights_path=RQ_PATH,
    rqvae_codebook_normalize=False,
    rqvae_sim_vq=False,
).to(DEVICE)

print("precomputing corpus semantic IDs...")
tokenizer.precompute_corpus_ids(item_dataset)

model = EncoderDecoderRetrievalModel(
    embedding_dim=128,
    attn_dim=512,
    dropout=0.3,
    num_heads=8,
    n_layers=8,
    num_embeddings=256,
    inference_verifier_fn=lambda x: tokenizer.exists_prefix(x),
    sem_id_dim=tokenizer.sem_ids_dim,
    max_pos=train_dataset.max_seq_len * tokenizer.sem_ids_dim,
    jagged_mode=False,
).to(DEVICE)

checkpoint = torch.load(DECODER_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model"])
print(f"loaded decoder checkpoint iter: {checkpoint['iter']}")

model.eval()
model.enable_generation = True

loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
batch = batch_to(next(iter(loader)), DEVICE)
tokenized = tokenizer(batch)

print("input sem_ids shape:", tuple(tokenized.sem_ids.shape))
print("actual future sem_ids shape:", tuple(tokenized.sem_ids_fut.shape))
print("running Top-K generation...")

with torch.no_grad():
    generated = model.generate_next_sem_id(
        tokenized,
        top_k=True,
        temperature=1,
    )

print("generated sem_ids shape:", tuple(generated.sem_ids.shape))
print("generated log_probas shape:", tuple(generated.log_probas.shape))
print("first actual semantic ID:", tokenized.sem_ids_fut[0].detach().cpu().tolist())
print("first 3 generated semantic IDs:", generated.sem_ids[0, :3].detach().cpu().tolist())
print("GENERATION_SMOKE_TEST=PASS")
