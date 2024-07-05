import torch
from torch.nn import functional

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200


class BLM(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs, targets=None):
        logits = self.token_embedding_table(inputs) #(B,T,C)

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
            # get the predictions
            logits, loss = self(inputs)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = functional.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            input_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
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
    inputs, targets = get_batch('train')

    logits, loss = model(inputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(gpu_model.generate(context, max_new_tokens=500)[0].tolist()))