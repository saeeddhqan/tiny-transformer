
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

block_size = 256
learning_rate = 9e-4
eval_interval = 300 # Every n step, we do an evaluation.
iterations = 5000 # Like epochs
eval_iters = 100
batch_size = 64
embeds_size = 195
num_heads = 5
num_layers = 5
drop_prob = 0.15

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1234)

with open('shakespeare.txt') as fp:
	text = fp.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}
encode = lambda s: [stoi[x] for x in s]
decode = lambda e: ''.join([itos[x] for x in e])

data = torch.tensor(encode(text), dtype=torch.long).to(device)

train_split = int(0.9 * len(data))
train_data = data[:train_split]
test_data = data[train_split:]


def get_batch(split='train', block_size=block_size, batch_size=batch_size):
	'''
		Create a random batch and returning batch along with targets.
	'''
	data = train_data if split == 'train' else test_data
	ix = torch.randint(len(data) - block_size, (batch_size,))
	x = torch.stack([data[i:i + block_size] for i in ix])
	y = torch.stack([data[i+1:i + block_size + 1] for i in ix])
	return x, y

@torch.no_grad()
def estimate_loss():
	'''
		We select eval_iters chunks from both train and val data
		and save their losses. All in all, evaluating the perf
		of the model on train and test data.
	'''
	out = {}
	for split in ('train', 'test'):
		# A tensor to capture the losses
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			X, Y = get_batch(split)
			logits, loss = model(X, Y)
			losses[k] = loss.item()
		out[split] = losses.mean()
	return out


class head(nn.Module):
	'''
		Communication between tokens happen here.
	'''
	def __init__(self, head_size=8):
		super().__init__()
		self.key = nn.Linear(embeds_size, head_size, bias=False)
		self.query = nn.Linear(embeds_size, head_size, bias=False)
		self.value = nn.Linear(embeds_size, head_size, bias=False)
		self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
		self.dropout = nn.Dropout(drop_prob)

	def forward(self, x):
		B,T,C = x.shape
		# What am I looking for?
		q = self.query(x)
		# What do I have?
		k = self.key(x)
		# What is the representation value of me?
		# Or: what's my personality in the group?
		# Or: what mask do I have when I'm in a group?
		v = self.value(x)
		scores = q @ k.transpose(-2, -1) * (1 / math.sqrt(C)) # (B,T,head_size) @ (B,head_size,T) --> (B,T,T)
		scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
		scores = F.softmax(scores, dim=-1)
		scores = self.dropout(scores)
		out = scores @ v
		return out


class multihead(nn.Module):
	'''
		I have multiple personalities(v), tendencies and needs (q), and valuable things (k) in different groups.
	'''
	def __init__(self, num_heads=4, head_size=8):
		super().__init__()
		self.multihead = nn.ModuleList([head(head_size) for _ in range(num_heads)])
		self.output_linear = nn.Linear(embeds_size, embeds_size)
		self.dropout = nn.Dropout(drop_prob)

	def forward(self, hidden_state):
		hidden_state = torch.cat([head(hidden_state) for head in self.multihead], dim=-1)
		hidden_state = self.output_linear(hidden_state)
		hidden_state = self.dropout(hidden_state)
		return hidden_state


class transformer_block(nn.Module):
	def __init__(self):
		super().__init__()
		self.head_count = embeds_size // num_heads
		self.n_heads = multihead(num_heads, self.head_count)
		self.ffn = nn.Sequential(
			nn.Linear(embeds_size, 4 * embeds_size),
			nn.ReLU(),
			nn.Linear(4 * embeds_size, embeds_size),
			nn.Dropout(drop_prob),
		)
		self.ln1 = nn.LayerNorm(embeds_size)
		self.ln2 = nn.LayerNorm(embeds_size)

	def forward(self, hidden_state):
		hidden_state = hidden_state + self.n_heads(self.ln1(hidden_state))
		hidden_state = hidden_state + self.ffn(self.ln2(hidden_state))
		return hidden_state


# We do feed-forward n times where n is block_size
class transformer(nn.Module):
	def __init__(self):
		super().__init__()
		self.stack = nn.ModuleDict(dict(
			tok_embs=nn.Embedding(vocab_size, embeds_size),
			pos_embs=nn.Embedding(block_size, embeds_size),
			dropout=nn.Dropout(drop_prob),
			blocks=nn.Sequential(
				transformer_block(),
				transformer_block(),
				transformer_block(),
				transformer_block(),
				transformer_block(),
			),
			ln=nn.LayerNorm(embeds_size),
			lm_head=nn.Linear(embeds_size, vocab_size),
		))

	def forward(self, seq, targets=None):
		B, T = seq.shape
		tok_emb = self.stack.tok_embs(seq) # (batch, block_size, embed_dim) (B,T,C)
		pos_emb = self.stack.pos_embs(torch.arange(T, device=device))
		x = tok_emb + pos_emb
		x = self.stack.dropout(x)
		x = self.stack.blocks(x)
		x = self.stack.ln(x)
		logits = self.stack.lm_head(x) # (B, block_size, vocab_size)

		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B * T, C)
			targets = targets.view(B * T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss

	def autocomplete(self, idx, _len=10):
		for _ in range(_len):
			idx_cond = idx[:, -block_size:] # crop it
			logits, _ = self(idx_cond)
			logits = logits[:, -1, :] # we only care about the last probability
			probs = F.softmax(logits, dim=-1)
			# It selects samples from probs. The higher the prob, the more the chance of being selected
			next_idx = torch.multinomial(probs, num_samples=1) # (B, 1) one prediction for each batch
			idx = torch.cat((idx, next_idx), dim=1)
		return idx


model = transformer()
if input('Would you like to load the model[y|n]?') == 'y':
	model.load_state_dict(torch.load('model.pth'))
model.to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

def generate(_len=100):
	sample = torch.zeros((1, 1), dtype=torch.long, device=device)
	generated = model.autocomplete(sample, _len)
	decoded = decode(generated[0].tolist())
	return decoded

print(generate())
for epoch in range(iterations):
	X, y = get_batch()
	pred, loss = model(X, y)
	optimizer.zero_grad(set_to_none=True)
	loss.backward()
	optimizer.step()
	if epoch % eval_interval == 0:
		print(f"epoch: {loss.item()}")
print(generate(1000))

if input('Would you like to save the model[y|n]?') == 'y':
	torch.save(model.state_dict(), 'model.pth')
