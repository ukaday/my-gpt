import torch
from torch.nn import functional
torch.autograd.set_detect_anomaly(True)

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters = 200
num_embed = 32


class Head(torch.nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = torch.nn.Linear(num_embed, head_size, bias=False)
        self.query = torch.nn.Linear(num_embed, head_size, bias=False)
        self.value = torch.nn.Linear(num_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        b, t, c = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weights = weights.masked_fill(self.tril[:t, :t] == 0, float('-inf'))  # (B, T, T)
        weights = functional.softmax(weights, dim=-1)  # (B, T, T)
     
        v = self.value(x)  # (B,T,hs)
        out = weights @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(head_size) for x in range(num_heads)])
        self.projection = torch.nn.Linear(num_embed, num_embed)

    def forward(self, x):
        concatenation = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.projection(concatenation)


class FeedForward(torch.nn.Module):
    def __init__(self, num_embed):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(num_embed, 4 * num_embed),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * num_embed, num_embed),
        )

    def forward(self, x):
        return self.net(x)


class Block(torch.nn.Module):
    def __init__(self, num_embed, num_heads):
        super().__init__()
        head_size = num_embed // num_heads
        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward(num_embed)
        self.layer_norm1 = torch.nn.LayerNorm(num_embed)
        self.layer_norm2 = torch.nn.LayerNorm(num_embed)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class BLM(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, num_embed)
        self.position_embedding_table = torch.nn.Embedding(block_size, num_embed)
        # self.self_attention_heads = MultiHeadAttention(4, num_embed // 4)  # 4 heads and 32 / 4 head size
        # self.feed_forward = FeedForward(num_embed)
        self.blocks = torch.nn.Sequential(
            Block(num_embed, num_heads=4),
            Block(num_embed, num_heads=4),
            Block(num_embed, num_heads=4),
            torch.nn.LayerNorm(num_embed)
        )
        self.language_modeling_head = torch.nn.Linear(num_embed, vocab_size)

    def forward(self, inputs, targets=None):
        b, t = inputs.shape

        token_embeddings = self.token_embedding_table(inputs)  # (B,T,num_embed)
        position_embeddings = self.position_embedding_table(torch.arange(t, device=device))  # (T, C)
        x = token_embeddings + position_embeddings
        # x = self.self_attention_heads(x)
        # x = self.feed_forward(x)
        x = self.blocks(x)
        logits = self.language_modeling_head(x)  # (B,T, vocab_size)

        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b*t, c)
            targets = targets.view(b*t)
            loss = functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, inputs, max_new_tokens):
        for i in range(max_new_tokens):
            input_crop = inputs[:, -block_size:]
            logits, loss = self(input_crop)
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = functional.softmax(logits, dim=-1)  # (B, C)
            input_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            inputs = torch.cat((inputs, input_next), dim=1)  # (B, T+1)
        return inputs


def get_batch(part):
    d = training_data if part == 'train' else validation_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for part in ['train', 'validate']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(part)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[part] = losses.mean()
    model.train()
    return out


with open('tolkein.txt', 'r', encoding='utf-8') as file:
    text = file.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

ctoi = {character: integer for integer, character in enumerate(chars)}
itoc = {integer: character for integer, character in enumerate(chars)}
encode = lambda string: [ctoi[character] for character in string]
decode = lambda array: ''.join([itoc[integer] for integer in array])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(.9 * len(data))
training_data = data[:n]
validation_data = data[n:]

model = BLM(vocab_size)
gpu_model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for i in range(max_iters):

    if i % eval_interval == 0 or i == max_iters - 1:
        losses = estimate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['validate']:.4f}")

    inputs, targets = get_batch('train')

    logits, loss = model(inputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(gpu_model.generate(context, max_new_tokens=500)[0].tolist()))