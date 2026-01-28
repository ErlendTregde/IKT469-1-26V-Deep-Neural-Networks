import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

sentence = "Darth Vader is the villain"
tokens = sentence.replace(",", "").split()

dc = {s: i for i, s in enumerate(sorted(tokens))}
sentence_int = torch.tensor([dc[w] for w in tokens], dtype=torch.long)

print("Tokens:", tokens)
print("Word->index:", dc)
print("Sentence ids:", sentence_int)

vocab_size = 50000
embed_dim = 3

torch.manual_seed(123)
embed = nn.Embedding(vocab_size, embed_dim)

X = embed(sentence_int).detach()
T, d = X.shape

print("\nEmbedded sentence X shape:", X.shape)
print(X)

d_q, d_k, d_v = 2, 2, 4
torch.manual_seed(123)

W_Q = torch.rand(d, d_q)
W_K = torch.rand(d, d_k)
W_V = torch.rand(d, d_v)

Q = X @ W_Q  
K = X @ W_K  
V = X @ W_V  

print("\nQ shape:", Q.shape, "K shape:", K.shape, "V shape:", V.shape)

attention_scores = (Q @ K.T) / math.sqrt(d_k)     
attention_weights = F.softmax(attention_scores, dim=-1)     
context_vector = attention_weights @ V                   

print("\nFull self-attention context shape:", context_vector.shape)
print(context_vector)

# Visualize the attention weights
plt.figure(figsize=(8, 6))
plt.imshow(attention_weights.detach().cpu().numpy(), aspect="auto", cmap="viridis")
plt.xticks(range(T), tokens, rotation=45, ha="right")
plt.yticks(range(T), tokens)
plt.colorbar()
plt.title("Self-Attention Weights")
plt.xlabel("Keys")
plt.ylabel("Queries")
plt.tight_layout()
plt.savefig("attention_heatmap.png", dpi=200)
print("\nVisualization saved as 'attention_heatmap.png'")
