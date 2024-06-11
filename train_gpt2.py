# Main architectural differences between GPT and GPT-2 are,
        #a. Layer norm is swapped to before Feedforward MLP
        #b. And one additional layer norm before the linear layer
        #c. gpt2 uses no bias in the final classification layer



from dataclasses import dataclass
import torch
import torch.nn as nn

from torch.nn import functional as F

class SelfAttention(nn.Module):
   def __init__(self, config) -> None:
       super().__init__()
       assert config.n_embd % config.n_head == 0
       self.config = config
       self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb)
       self.c_attn = nn.Linear(config.n_emb, config.n_emb)
       self.n_head = config.n_head
       self.n_emb = config.n_emb

       self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))


   def forward(self, x):
        B, T, C = x.shape # batch, sequence length, embedding_dimenionality

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y


class MLP(nn.Module):
   def __init__(self, config) -> None:
       super().__init__()
       self.config = config
       self.c_fc = nn.Linear(config.n_emb, 4 * config.n_emb)
       self.gelu = nn.GELU(approximate='tanh')
       self.c_proj = nn.Linear(4 * config.n_emb, config.n_emb)

   def forward(self, x):
       x = self.c_fc(x)
       x = self.gelu(x)
       x = self.c_proj(x)
       return x
       

class Block(nn.Module):
   
   def __init__(self, config) -> None:
       super().__init__()
       self.config = config
       self.ln_1 = nn.LayerNorm(config.n_emb)
       self.attn = SelfAttention(config)
       self.ln_2 = nn.LayerNorm(config.n_emb)
       self.mlp = MLP(config)

   def forward(self, x):
       #gpt2 differs in the order layernorm is order in the block
       x = x + self.attn(self.ln_1(x)) #norm --> attn --> add
       x = x + self.mlp(self.ln_2(x))
       return x

@dataclass
class GPTConfig:
   block_size: int = 1024 #max sequence length
   vocab_size: int = 50257
   n_layer: int = 12
   n_head: int = 12
   n_emb:int = 768

class GPT(nn.Module):
   
   def __init__(self, config) -> None:
       super().__init__()
       self.config = config

       self.transfomer = nn.ModuleDict(
               dict(
                   wte = nn.Embedding(config.vocab_size, config.n_emb),
                   wpe = nn.Embedding(config.vocab_size, config.n_emb),
                   h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                   ln_f = nn.LayerNorm(config.n_emb),
               )
       )
       self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
       # weight sharing scheme
       self.transformer.wte.weight = self.lm_head.weight
       # init params
       self.apply(self._init_weights)

   # this is used to intialize the weights with some variance as followed by gpt-2 paper     
   def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

   def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
        

   @classmethod
   def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model



class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
