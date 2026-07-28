import torch 
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iterations = 5000
eval_intervals = 500
learning_rate = 1e-3
eval_iterations = 200
n_embd = 32
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    data = f.read()

chars = sorted(list(set(data)))
vocab_size = len(chars)

s_to_i = {ch:i for i,ch in enumerate(chars)} # encrypt mapping
i_to_s = {i:ch for i,ch in enumerate(chars)} # decrypt mapping

def encode(s):
    return [s_to_i[c] for c in s]
def decode(i):
    return ''.join([i_to_s[c] for c in i])

encrypted_data = torch.tensor(encode(data), dtype= torch.long)

n = int(0.9*len(encrypted_data))

train_data = encrypted_data[:n]
test_data = encrypted_data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else test_data

    ix = torch.randint(len(encrypted_data)-block_size, (batch_size,))
    x = torch.stack([encrypted_data[i:i+block_size] for i in ix] )
    y = torch.stack([encrypted_data[i+1:i+block_size+1] for i in ix])
    return x,y

def estimated_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose (-2,-1) * C**-0.5 # (B,T,16) @ (B,16,T) --> (B,T,T)

# weights = torch.zeros((T,T))
        weights = weights.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        weights = F.softmax(weights, dim= -1)
        v = self.value(x)
        out = weights@ v
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self,x):
        return torch.cat([h(x) for h in self.heads], dim = -1)
    
# super simple bigram model
class bigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embd_table = nn.Embedding(block_size, n_embd)
        self.sa_head = MultiHeadAttention(4, n_embd//4)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        B,T = idx.shape

        tok_emb = self.token_embedding_table(idx) # B T C
        pos_emb =  self.position_embd_table(torch.arange(T))
        x = tok_emb + pos_emb
        x = self.sa_head(x)
        logits = self.lm_head(x) # B T vocab_size

        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):# idx is (B,T) array of indices
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get the prediction
            logits, loss = self(idx_cond)
            # focus only on the last step
            logits = logits[:,-1,:]# (B,C)
            # apply softmax for probabilities
            prob = F.softmax(logits, dim= -1)
            # sample from the distribution
            idx_next = torch.multinomial(prob, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    

model = bigramLanguageModel()

# create a pytorch optimizer 
optimizer= torch.optim.Adam(model.parameters(), lr=1e-3)

for iters in range(max_iterations):
    if iters % eval_intervals == 0:
        losses = estimated_loss()
        print(f"step {iters}: train loss{losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1) ,dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))