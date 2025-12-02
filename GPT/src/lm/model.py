import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lm.utils import count_params

"""
Dimension symbols:
    B - batch size
    S - sequence length
    D - hidden dimension (n_embd)
    H - number of attention heads (n_head)
    HD - hidden dimension of a single attention head (d // n_head)
    V - size of the vocabulary
"""

# ==========================================
# Llama Components (New Architecture)
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, 1, *freqs_cis.shape) # Broadcast
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SwiGLU(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        hidden_dim = int(4 * n_embd) 
        hidden_dim = int(2 * hidden_dim / 3) 
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class LlamaAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        attn_hidden_dim = n_embd // n_head
        self.q_attn = nn.Linear(n_embd, n_embd, bias=False) # Llama usually no bias
        self.k_attn = nn.Linear(n_embd, n_embd, bias=False)
        self.v_attn = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(p_dropout)
        scale_factor = 1 / torch.sqrt(torch.tensor(attn_hidden_dim))
        self.register_buffer("scale_factor", scale_factor)

    def forward(self, x, freqs_cis, attention_mask=None):
        B, S, D = x.shape
        q = self.q_attn(x)
        k = self.k_attn(x)
        v = self.v_attn(x)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.n_head)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.n_head)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.n_head)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        if attention_mask is None:
            out = F.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True 
            )
        else:
            attn_mask = attention_mask[:, None, None, :] # (B, 1, 1, S)
            out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True
            )

        out = rearrange(out, "b h s d -> b s (h d)")
        return self.proj(out)

class LlamaBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.rms_1 = RMSNorm(n_embd)
        self.mha = LlamaAttention(n_embd, n_head)
        self.rms_2 = RMSNorm(n_embd)
        self.ff = SwiGLU(n_embd)

    def forward(self, x, freqs_cis, attention_mask):
        h = x + self.mha(self.rms_1(x), freqs_cis, attention_mask)
        out = h + self.ff(self.rms_2(h))
        return out

class LlamaLM(nn.Module):
    """Llama-style Decoder Language Model."""
    def __init__(self, n_vocab, n_embd, n_head, n_positions, n_layer, p_dropout=0.1):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_positions = n_positions
        self.n_layer = n_layer
        
        self.token_embeddings = nn.Embedding(n_vocab, n_embd)
        self.head_dim = n_embd // n_head
        self.freqs_cis = precompute_freqs_cis(self.head_dim, n_positions * 2)
        
        self.blocks = nn.ModuleList([LlamaBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln = RMSNorm(n_embd)
        self.dropout = nn.Dropout(p_dropout)
        
        self.apply(self._init_weights)

        self.flops_per_token = (
            6 * count_params(self) + 12 * n_layer * n_embd * n_positions
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        x = self.dropout(self.token_embeddings(input_ids))
        
        b, s = input_ids.shape
        if self.freqs_cis.device != x.device:
            self.freqs_cis = self.freqs_cis.to(x.device)
        freqs_cis = self.freqs_cis[:s]

        for block in self.blocks:
            x = block(x, freqs_cis, attention_mask)
            
        x = self.ln(x)
        logits = F.linear(x, self.token_embeddings.weight)
        return logits


# ==========================================
# Original Components (GPT-2 Style)
# ==========================================

class MultiHeadAttention(nn.Module):
    """The multi-head attention module in a decoder block."""

    def __init__(self, n_embd: int, n_head: int, p_dropout: float = 0.1):
        super().__init__()
        """Initialize the modules used by multi-head attention."""

        self.n_head = n_head
        attn_hidden_dim = n_embd // n_head

        self.q_attn = nn.Linear(n_embd, n_embd)
        self.k_attn = nn.Linear(n_embd, n_embd)
        self.v_attn = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(p_dropout)

        scale_factor = 1 / torch.sqrt(torch.tensor(attn_hidden_dim))
        self.register_buffer("scale_factor", scale_factor)

    def q_kT_v(
        self, x: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Project the hidden states to q, kT, v prior to computing attention.

        Args:
            x: embeddings or hidden states (B x S x D) from the previous decoder block

        Returns:
            q: The query vector used by multi-head attention (B x H x S x HD)
            kT: The transpose of the key vector used by multi-head attention (B x H x HD x S)
            v: The value vector used by multi-head attention (B x H x S x HD)
        """
        q = self.q_attn(x)
        k = self.k_attn(x)
        v = self.v_attn(x)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.n_head)
        kT = rearrange(k, "b s (h d) -> b h d s", h=self.n_head)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.n_head)

        return q, kT, v

    def self_attention(
        self,
        q: torch.FloatTensor,
        kT: torch.FloatTensor,
        v: torch.FloatTensor,
        attention_mask: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """Compute multi-head attention over the inputs.

        Args:
            q: The query vector used by multi-head attention (B x H x S x HD)
            kT: The transpose of the key vector used by multi-head attention (B x H x HD x S)
            v: The value vector used by multi-head attention (B x H x S x HD)
            attention_mask (optional): Mask indicating tokens that shouldn't
              be included in self-attention (B x S). 1 stands for a token that is
              included, and 0 stands for a token that isn't.

        Returns:
            attn: Outputs of applying multi-head attention to the inputs (B x S x D)
        """

        B, H, S, HD = q.shape

        # compute the attention weights using q and kT
        qkT = torch.matmul(q, kT) #(B H S S)
        unmasked_attn_logits = qkT * self.scale_factor

        """
        In decoder models, attention logits are masked such that computation at
        each position does not involve embeddings / hidden states of future
        positions.

        This boolean mask should have shape (S x S) and has value True iff
        position i is allowed to attend to position j (i.e., j <= i).

        Example (S = 5):
        causal_mask = tensor([
         [ True, False, False, False, False],
         [ True,  True, False, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True]
        ])
        
        Note that `causal mask` needs to be on the same device as the input
        tensors (q, kT, v). You can move a tensor to the right device by calling
        `tensor.to(q.device)`.

        Hint: torch.triu or torch.tril
        """
        causal_mask = torch.tril(torch.ones((S, S), device=q.device, dtype=torch.bool))

        """
        Sometimes, we want to pad the input sequences so that they have the same
        length and can fit into the same batch. These padding tokens should not
        have any effect on the output of self-attention. To achieve this, we
        need to mask out the logits that correspond to those tokens.

        Example (B = 2, S = 5):
        causal_mask = tensor([
         [ True, False, False, False, False],
         [ True,  True, False, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True]
        ])

        attention_mask = tensor([
         [0., 0., 1., 1., 1.],
         [1., 1., 1., 1., 1.]
        ])

        mask = tensor([
        [[[False, False, False, False, False],
          [False, False, False, False, False],
          [False, False,  True, False, False],
          [False, False,  True,  True, False],
          [False, False,  True,  True,  True]]],

        [[[ True, False, False, False, False],
          [ True,  True, False, False, False],
          [ True,  True,  True, False, False],
          [ True,  True,  True,  True, False],
          [ True,  True,  True,  True,  True]]]
        ])

        Note that `mask` needs to be on the same device as the input tensors
        q, kT and v.
        """

        if attention_mask is None:
            mask = causal_mask
        else:
            mask = causal_mask & (attention_mask[:, None, None, :] == 1)

        """
        Fill unmasked_attn_logits with float_min wherever causal mask has value False.

        Hint: torch.masked_fill
        """
        float_min = torch.finfo(q.dtype).min
        attn_logits = unmasked_attn_logits.masked_fill(~mask, float_min)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # scale value by the attention weights.
        attn = torch.matmul(attn_weights, v)  # (B H S HD)
        attn = rearrange(attn, "b h s d -> b s (h d)")

        if attention_mask is not None:
            # zero out outputs that correspond to masked input tokens
            attn = attn * attention_mask[:, :, None]
        
        return attn

    def projection(self, attn: torch.FloatTensor) -> torch.FloatTensor:
        """Apply a dropout and a linear projection to outputs of attention"""
        return self.dropout(self.proj(attn))

    def forward(
        self, x: torch.FloatTensor, attention_mask: torch.FloatTensor | None = None
    ) -> torch.FloatTensor:
        """A full forward pass of the multi-head attention module.

        Args:
            x: embeddings or hidden states (B x S x D) from the previous decoder block

        Returns:
            y: outputs (B x S x D) of the multi-head attention module
        """
        q, kT, v = self.q_kT_v(x)
        attn = self.self_attention(q, kT, v, attention_mask)
        y = self.projection(attn)
        return y


class FeedForward(nn.Module):
    """The feedforward attention module in a decoder block."""

    def __init__(self, n_embd: int, p_dropout: float = 0.1):
        """Initialize the modules used by feedforward."""
        super().__init__()

        middle_dim = 4 * n_embd  # stick to what GPT-2 does
        self.linear_in = nn.Linear(n_embd, middle_dim)
        self.linear_out = nn.Linear(middle_dim, n_embd)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """A full forward pass of the feedforward module.

        Args:
            x: outputs (B x S x D) of the first Add & Norm operation

        Returns:
            z: outputs (B x S x D) of the feedforward module

        Different from what you saw in class which uses ReLU as the activation,
        we are going to follow GPT-2 which uses GeLU. You should also apply
        self.dropout to the output.
        """

        y = F.gelu(self.linear_in(x))
        z = self.dropout(self.linear_out(y))
        return z


class DecoderBlock(nn.Module):
    """A single decoder block in a decoder language model."""

    def __init__(self, n_embd: int, n_head: int):
        """Initialize the modules used in a decoder block."""
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embd)
        self.mha = MultiHeadAttention(n_embd, n_head)
        self.ff = FeedForward(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

    def forward(
        self, x: torch.FloatTensor, attention_mask: torch.FloatTensor | None
    ) -> torch.FloatTensor:
        """A full forward pass of the decoder block.

        Args:
            x: embeddings or hidden states (B x S x D) from the previous decoder block
            attention_mask (optional): Mask indicating tokens that shouldn't
              be included in self-attention (B x S). 1 stands for a token that is
              included, and 0 stands for a token that isn't.
        Returns:
            y: outputs of the current decoder block

        Different from what you saw in class which uses ReLU as the activation,
        we are going to follow GPT-2 which uses GeLU. You should also apply
        self.dropout to the output.

        A note on where to place layer normalization (LN): in the lecture, you
        saw "post-LN", which applies LN to the outputs of MHA / FF modules after
        the residual is added. Another approach to do this is "pre-LN", which
        appiles LN to the inputs of the attention and feedforward modules. Both
        implementations should pass the tests. See explanations here:
        https://sh-tsang.medium.com/review-pre-ln-transformer-on-layer-normalization-in-the-transformer-architecture-b6c91a89e9ab
        """
        residual = x
        x = self.ln_1(x)
        x = self.mha(x, attention_mask)
        x = x + residual
        
        residual = x
        x = self.ln_2(x)
        x = self.ff(x)
        x = x + residual
        return x


class DecoderLM(nn.Module):
    """The decoder language model."""

    def __init__(
        self,
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        n_layer: int,
        p_dropout: float = 0.1,
    ):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_positions = n_positions
        self.n_layer = n_layer
        self.p_dropout = p_dropout

        self.token_embeddings = nn.Embedding(n_vocab, n_embd)
        self.position_embeddings = nn.Embedding(n_positions, n_embd)
        self.blocks = nn.ModuleList(
            [DecoderBlock(n_embd, n_head) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(self.p_dropout)

        # initialize weights according to nanoGPT
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / torch.sqrt(2 * n_layer))

        # count flops per token according to nanoGPT
        self.flops_per_token = (
            6 * count_params(self) + 12 * n_layer * n_embd * n_positions
        )

    def embed(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """Convert input_ids to embeddings (token_embeddings + positional_embeddings).

        Args:
            input_ids: tokens ids with shape (B x S)
            attention_mask (optional): Mask indicating whether tokens should be
              ignored.

        Returns:
            embeddings: token representations with shape (B x S x D)
        """

        """
        Position ids are indices of tokens in the sequence. When attention_mask
        isn't provided, they are simply [0, 1, 2, ...] for every sequence in the
        batch. When they are provided, you should ignore tokens with attention_mask
        equal to 0.
        
        Example (B = 2, S = 5):
        
        attention_mask = tensor([
         [0., 0., 1., 1., 1.],
         [1., 1., 1., 1., 1.]
        ])

        position_ids = tensor([
         [0, 0, 0, 1, 2],
         [0, 1, 2, 3, 4]
        ])

        Note that the position ids for masked out tokens do not matter, as long
        as they don't trigger out-of-bounds errors when fed into the embedding
        layer. I.e., they should be within [0, n_positions).

        Hint: torch.cumsum
        """

        assert input_ids.shape[1] <= self.n_positions
        B, S = input_ids.shape
        if attention_mask is None:
            position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).repeat(B, 1)
        else:
            position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0).long()

        token_embeddings = self.token_embeddings(input_ids)
        positional_embeddings = self.position_embeddings(position_ids)

        return self.dropout(token_embeddings + positional_embeddings)

    def token_logits(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Project the final hidden states of the model to token logits.

        Args:
            x: hidden states produced by the final decoder block (B x S x D)

        Returns:
            logits: logits corresponding to the predicted next token likelihoods (B x S x V)

        Hint: Question 2.2.
        """

        logits = F.linear(x, self.token_embeddings.weight)
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """A forward pass of the decoder LM, converting input_ids to token logits.

        Args:
            input_ids: tokens ids with shape (B x S)
            attention_mask (optional): Mask indicating whether tokens should be
              ignored.

        Returns:
            logits: logits corresponding to the predicted next token likelihoods (B x S x V)
        """

        x = self.embed(input_ids, attention_mask)
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.ln(x)
        logits = self.token_logits(x)

        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
