import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchsummary import summary

try:
    from .attn import *
    from .embed import DataEmbedding, TokenEmbedding
    from .base_models import MLP, resnet1d18
except ImportError:
    from attn import *
    from embed import DataEmbedding, TokenEmbedding
    from base_models import MLP, resnet1d18

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, unmask_indices=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask, unmask_indices=unmask_indices
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, unmask_indices=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask, unmask_indices=unmask_indices)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class cnn_extractor(nn.Module):
    def __init__(self, dim, input_plane):
        super(cnn_extractor, self).__init__()
        self.cnn = resnet1d18(input_channels=dim, inplanes=input_plane)

    def forward(self, x):
        x = self.cnn(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class AnomalyTransformer(nn.Module):
    def __init__(self, seq_len: int, patch_len: int, channel_num: int, dim: int, n_heads: int = 8, e_layers=3,
                 dropout: float = 0.2, activation: str = 'gelu', emb_dropout: float = 0.1, **kwargs):
        """
        :param dim: int, dimension of latent layers
        :param e_layers: int, depth of encoder layers
        """
        assert seq_len > 0
        assert patch_len > 0
        assert channel_num > 0
        assert dim > 0
        assert e_layers > 0
        assert seq_len % (2 * patch_len) == 0, 'The seq_len should be 2 * n * patch_len'

        super(AnomalyTransformer, self).__init__()

        num_patches = seq_len // patch_len
        pixel_values_per_patch = channel_num * patch_len

        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.mask_token = nn.Parameter(torch.randn(1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.to_patch = nn.Sequential(Rearrange('b c (n p1) -> b n c p1', p1=patch_len),
                                      Rearrange('b n c p1 -> (b n) c p1'))

        # Encoding
        self.cnn = cnn_extractor(dim=channel_num, input_plane=dim // 8)  # For temporal data
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        encoder_layers = [
            EncoderLayer(
                AttentionLayer(
                    AnomalyAttention(num_patches + 1,
                                     False,
                                     attention_dropout=dropout,
                                     output_attention=True),
                    dim, n_heads),
                dim,
                dropout=dropout,
                activation=activation) for _ in range(e_layers)]
        self.encoder = Encoder(
            encoder_layers,
            norm_layer=torch.nn.LayerNorm(dim)
        )

        # Decoder
        self.decoder_pos_emb = nn.Embedding(self.num_patches, dim)
        self.decoder = Transformer(dim=dim,
                                   depth=e_layers,
                                   heads=n_heads,
                                   dim_head=n_heads,
                                   mlp_dim=dim)
        self.to_pixels = nn.ModuleList([nn.Linear(dim, pixel_values_per_patch) for i in range(1)])
        self.projs = nn.ModuleList([nn.Linear(dim, dim) for i in range(2)]) # for IDC Loss

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.5, is_ssl: bool = True):
        """
        patching ->
        @param x: with shape of [BATCH_SIZE, CHANNEL_DIM, TIME_LEN]
        @return: output, series, prior, _
        """
        batch, channel_num, time_steps = x.shape
        device = x.device
        patches = self.to_patch[0](x)

        patch2seq = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                  Rearrange('(b n) c 1 -> b n c', b=batch))
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=batch)
        if is_ssl:
            num_masked = int(mask_ratio * self.num_patches)
            rand_indices = torch.randperm(self.num_patches, device=device)
            masked_indices = rand_indices[: num_masked].sort()[0]
            unmasked_indices = rand_indices[num_masked:].sort()[0]
            masked_num = masked_indices.shape[0]
            unmasked_num = unmasked_indices.shape[0]
            unmasked_patches = patches[:, unmasked_indices, :, :]
            tokens = self.to_patch[1](unmasked_patches)
            tokens = self.cnn(tokens)
            Flat = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                 Rearrange('(b n) c 1 -> b n c', b=batch))
            tokens = Flat(tokens)
            ori_tokens = tokens.clone()
            # Add cls_tokens before tokens
            tokens = torch.cat((cls_tokens, tokens), dim=1)
            pos_embedding = torch.cat((self.pos_embedding[:, 0:1, :],
                                       self.pos_embedding[:, unmasked_indices + 1, :]), dim=1)
            tokens = tokens + pos_embedding
            encoded_tokens, series, prior, sigmas = self.encoder(tokens, unmask_indices=unmasked_indices)  # tokens.shape=(BATCH_SIZE ,6, 64)
            decoder_tokens = encoded_tokens
            # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
            mask_tokens = repeat(self.mask_token[0], 'd -> b n d', b=batch, n=masked_num)
            # mask_tokens = repeat(self.mask_token[0], 'd -> b n d', b=batch, n=masked_num_f+masked_num_t)
            decoder_pos_emb = self.decoder_pos_emb(masked_indices)
            mask_tokens = mask_tokens + decoder_pos_emb
            # concat the masked tokens to the decoder tokens and attend with decoder
            decoder_tokens = torch.cat((decoder_tokens, mask_tokens), dim=1)
            decoded_tokens = self.decoder(decoder_tokens)
            pred_pixel_values = self.to_pixels[0](decoder_tokens[:, 1:])

            recon_loss = F.mse_loss(pred_pixel_values, rearrange(patches[:, rand_indices], 'b n c p -> b n (c p)'))
            # ----- ----- -----
            return recon_loss, series, prior, sigmas,

        else:
            tokens = self.to_patch[1](patches)
            tokens = self.cnn(tokens)
            x = torch.cat((cls_tokens[:, 0:1, :], patch2seq(tokens)), dim=1)
            x += self.pos_embedding
            x = self.dropout(x)
            x, series, prior, sigmas = self.encoder(x)

            # compute anomaly scores
            prior_norm = [p / torch.sum(p, dim=-1, keepdim=True) for p in prior]
            series_norm = [s / torch.sum(s, dim=-1, keepdim=True) for s in series]

            # compute symmetric KL divergence for each layer
            ass_discrepancy = 0.0
            for s, p in zip(series_norm, prior_norm):
                kl_1 = p * (torch.log(p + 1e-5) - torch.log(s + 1e-5))
                kl_2 = s * (torch.log(s + 1e-5) - torch.log(p + 1e-5))
                kl = torch.sum(kl_1 + kl_2, dim=-1)  # sum over attention targets
                ass_discrepancy += kl

            ass_discrepancy = ass_discrepancy / len(series)  # average over layers
            anomaly_score = ass_discrepancy[:, :]  # skip cls token
            return x[:, 0, :], anomaly_score


if __name__ == '__main__':
    # ----- Test AnomalyTransformer ----
    model = AnomalyTransformer(seq_len=200, patch_len=10, channel_num=64, dim=64).to('cuda')
    x = torch.rand([32, 64, 200]).to('cuda')
    y = model(x, is_ssl=False)
    print(y)
    # ----- ----- -----

    # # ----- Test Encoder ----
    # model = Encoder(
    #         [
    #             EncoderLayer(
    #                 AttentionLayer(
    #                     AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
    #                     d_model, n_heads),
    #                 d_model,
    #                 d_ff,
    #                 dropout=dropout,
    #                 activation=activation
    #             ) for l in range(e_layers)
    #         ],
    #         norm_layer=torch.nn.LayerNorm(d_model)
    #     ).to('cuda')
    # x = torch.rand([32, 6, 64]).to('cuda')
    # y = model(x)
    # print(y.shape)
    # summary(model, (200, 64), device='cuda', )
    # # ----- ----- -----

    # # ----- Test Encoder ----
    # model = Encoder(
    #         [
    #             EncoderLayer(
    #                 AttentionLayer(
    #                     AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
    #                     d_model, n_heads),
    #                 d_model,
    #                 d_ff,
    #                 dropout=dropout,
    #                 activation=activation
    #             ) for l in range(e_layers)
    #         ],
    #         norm_layer=torch.nn.LayerNorm(d_model)
    #     ).to('cuda')
    # x = torch.rand([32, 6, 64]).to('cuda')
    # y = model(x)
    # print(y.shape)
    # summary(model, (200, 64), device='cuda', )
    # # ----- ----- -----