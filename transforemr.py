import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# FlashAttention implementation
def flash_attention(q, k, v, mask=None, dropout=None, block_size=1024):
    # Extract info from query tensor
    batch_size, num_heads, seq_len, d_k = q.size()
    # Scaling factor for the dot product attention
    scale = 1 / math.sqrt(d_k)
    # Initialize tensors to store final output and attention weights
    output = torch.zeros_like(q)
    attention_weights = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=q.device)

    # Iterate over the sequence in blocks
    for block_start in range(0, seq_len, block_size):
        # Calculate the end of the block
        block_end = min(block_start + block_size, seq_len)
        
        # Extract blocks for query, key, and values
        local_q = q[:, :, block_start:block_end]
        local_k = k[:, :, :block_end]
        local_v = v[:, :, :block_end]

        # Local attention scaled dot product
        local_attn = torch.matmul(local_q, local_k.transpose(-1, -2)) * scale

        # Apply mask if provided
        if mask is not None:
            local_attn = local_attn + mask[:, :, block_start:block_end, :block_end]

        # Apply softmax
        local_attn = F.softmax(local_attn, dim=-1)

        # Apply dropout if provided
        if dropout is not None:
            local_attn = dropout(local_attn)

        # Compute output of current block
        local_output = torch.matmul(local_attn, local_v)

        # Store output of the current block
        output[:, :, block_start:block_end] = local_output
        attention_weights[:, :, block_start:block_end, :block_end] = local_attn

    return output, attention_weights    

# Helper function to apply FlashAttention
def flash_scaled_dot_product(q, k, v, mask=None):
    values, attention = flash_attention(q, k, v, mask)
    return values, attention

# Rotary Embedding class
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, seq_len):
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device='cuda') * -emb_scale)
        emb = torch.outer(torch.arange(seq_len, device='cuda'), emb)
        emb = torch.stack([torch.cos(emb), torch.sin(emb)], dim=-1)
        return emb.unsqueeze(1)

# Rotary Position Encoding class
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.rotary_emb = RotaryEmbedding(self.d_model // 2)

    def forward(self, x):
        pos_emb = self.rotary_emb(self.max_sequence_length)
        cos_pos, sin_pos = pos_emb.unbind(dim=-1)
        x_even, x_odd = x.chunk(2, dim=-1)
        x_even_new = x_even * cos_pos - x_odd * sin_pos
        x_odd_new = x_odd * cos_pos + x_even * sin_pos
        x_new = torch.cat([x_even_new, x_odd_new], dim=-1)
        return x_new

# Decoder layer class
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = flash_attention
        self.norm1 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Decoder class
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_sequence_length, dropout=0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.rotary_pe = RotaryPositionalEncoding(d_model, max_sequence_length)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.token_embeddings(x)
        x = self.rotary_pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.layer_norm(x)
        logits = self.output_projection(x)
        return logits

# Layer Normalization class
class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.epsilon)
        out = self.gamma * x_norm + self.beta
        return out

# FeedForward network class
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = getattr(F, activation)
        self.dropout_layer = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.linear2(x)
        return x
#LLM definition with the generate function    
class LLM(nn.Module):
    def __init__(self,vocab_size,d_model,num_heads,num_layers,d_ff,max_seq_leangth,dropout=0.1):
        super().__init__()
        self.token_embedding=nn.Embedding(vocab_size,d_model)
        self.rotary_pe=RotaryPositionalEncoding(d_model,max_seq_leangth)
        self.decoder=Decoder(d_model=d_model,vocab_size=vocab_size,max_sequence_length=max_seq_leangth,num_heads=num_heads,num_layers=num_layers,d_ff=d_ff,dropout=dropout)
        self.final_layer_norm=LayerNorm(d_model)
        self.ouput_projection=nn.Linear(d_model,vocab_size)

    def forward(self,input_ids,attention_mask=None):
        x=self.token_embeding(input_ids)
        x=self.rotary_pe(x)
        x=self.decoder(x,mask=attention_mask)
        x=self.final_layer_norm(x)
        logits=self.ouput_projection(x)
        return logits   
    def generate(self,input_ids,max_length,temperature=1.0):
        self.eval()
        with torch.no_grad():
            for _ in range(max_length-input_ids.size(1)):
                outputs=self(input_ids)
                next_token_logits=outputs[:,-1,:]/temperature
                next_token_probs=torch.softmax(next_token_logits,dim=-1)
                next_token=torch.multinomial(next_token_probs,num_samples=1)
                input_ids=torch.cat([input_ids,next_token],dim=-1)
                if next_token.item()==self.eos_token_id:
                    break
        return input_ids  

def load_model(model,path):
    model.load_state_dict(torch.load(path))
    return model
def save_model(model,path):
    torch.save(model.state_dict(),path)
def initialize_weights(model):
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
def train_step(model,optimizer,batch,loss_fn):
    model.train()
    optimizer.zero_grad()
    input_ids=batch['input_ids']
    labels=batch['labels']
    attention_mask=batch['attention_mask']
    outputs=model(input_ids,attention_mask=attention_mask)
    loss=loss_fn(outputs.view(-1,outputs.size(-1)),labels.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()
def evaluate(model,dataloader,loss_fn):
    model.eval()
    total_loss=0
    total_count=0

    with torch.no_grad():
        for batch in dataloader:
            input_ids=batch['input_ids']
            labels=batch['labels']
            attention_mask=batch['attention_mask']
            outputs=model(input_ids,attention_mask=attention_mask)
            loss=loss_fn(outputs.view(-1,outputs.size(-1),labels.view(-1)))
            total_loss+=loss.item()*input_ids.size(0)
            total_count+=input_ids.size(0)
    return total_loss/total_count            