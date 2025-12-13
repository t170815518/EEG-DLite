from loguru import logger
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torchsummary import summary

try:
    from base_models import MLP, resnet1d18
    from anomaly_transformer import *
except ModuleNotFoundError:
    from .base_models import MLP, resnet1d18
    from .anomaly_transformer import *


class cnn_extractor(nn.Module):
    def __init__(self, dim, input_plane):
        super(cnn_extractor, self).__init__()
        self.cnn = resnet1d18(input_channels=dim, inplanes=input_plane)

    def forward(self, x):
        x = self.cnn(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_pathces=None, dropout=0., is_anomaly: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if is_anomaly:
                self.layers.append(AnomalyAttentionLayer(n_heads=heads, d_model=dim, dropout=dropout,
                                                         num_patches=num_pathces, d_keys=dim, d_values=dim))
            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))

        self.is_anomaly = is_anomaly

    def forward(self, x, unmask_indices=None):
        if self.is_anomaly:
            # x [B, L, D]
            series_list = []
            prior_list = []
            sigma_list = []
            for attn_layer in self.layers:
                x, series, prior, sigma = attn_layer(x, x, x, attn_mask=None, unmask_indices=unmask_indices)
                series_list.append(series)
                prior_list.append(prior)
                sigma_list.append(sigma)

            association_dict = {'series': series_list, 'prior': prior_list, 'sigma': sigma_list}
            return x, association_dict
        else:
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
            return x, {}


class TFR(nn.Module):
    def __init__(self, seq_len, patch_len, num_classes, dim, depth, heads, mlp_dim, channels=12, dim_head=64,
                 dropout=0., emb_dropout=0., is_anomaly: bool = False):
        """
        The encoder of CRT
        """
        super().__init__()

        assert seq_len % (4 * patch_len) == 0, \
            'The seq_len should be 4 * n * patch_len, or there must be patch with both magnitude and phase data.'

        num_patches = seq_len // patch_len
        patch_dim = channels * patch_len

        self.to_patch = nn.Sequential(Rearrange('b c (n p1) -> b n c p1', p1=patch_len),
                                      Rearrange('b n c p1 -> (b n) c p1'))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 3, dim))
        self.modal_embedding = nn.Parameter(torch.randn(3, 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 3, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, num_patches, dropout, is_anomaly)

        self.cnn1 = cnn_extractor(dim=channels, input_plane=dim // 8)  # For temporal data
        self.cnn2 = cnn_extractor(dim=channels, input_plane=dim // 8)  # For magnitude data
        self.cnn3 = cnn_extractor(dim=channels, input_plane=dim // 8)  # For phase data

    def forward(self, x):
        batch, _, time_steps = x.shape
        # t, m, p refers to temporal features, magnitude featuers, phase features respectively
        # Assuming that the length of temporal data is L, then the magnitude and phase data are set as L // 2 here. The length can be adjusted by users.
        t, m, p = x[:, :, :time_steps // 2], x[:, :, time_steps // 2: time_steps * 3 // 4], x[:, :, -time_steps // 4:]
        t, m, p = self.to_patch(t), self.to_patch(m), self.to_patch(p)
        patch2seq = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                  Rearrange('(b n) c 1 -> b n c', b=batch))

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=batch)

        x = torch.cat((cls_tokens[:, 0:1, :], patch2seq(self.cnn1(t)),
                       cls_tokens[:, 1:2, :], patch2seq(self.cnn2(m)),
                       cls_tokens[:, 2:3, :], patch2seq(self.cnn3(p))), dim=1)

        b, t, c = x.shape  # t = time_steps + 3
        time_steps = t - 3
        t_token_idx, m_token_idx, p_token_idx = 0, time_steps // 2 + 1, time_steps * 3 // 4 + 2
        x[:m_token_idx] += self.modal_embedding[:1]
        x[m_token_idx: p_token_idx] += self.modal_embedding[1:2]
        x[p_token_idx:] += self.modal_embedding[2:]
        x += self.pos_embedding[:, : t]
        x = self.dropout(x)
        x = self.transformer(x)
        if isinstance(x[1], dict):
            x = x[0]
        t_token, m_token, p_token = x[:, t_token_idx + 1: m_token_idx], x[:, m_token_idx + 1: p_token_idx], x[:, p_token_idx + 1:]
        avg = (t_token.sum(dim=1) + m_token.sum(dim=1) + p_token.sum(dim=1)) / time_steps
        return avg


class TransformerEEG(nn.Module):
    """
    Args:
        seq_len (int): Length of the input EEG sequence.
        patch_len (int): Length of each patch for dividing the sequence.
        num_classes (int): Number of output classes (not used in this implementation).
        dim (int): Embedding dimension for Transformer input.
        depth (int): Number of Transformer layers.
        heads (int): Number of attention heads in Transformer.
        mlp_dim (int): Dimension of the feed-forward layer in Transformer.
        channels (int, optional): Number of EEG channels. Default is 12.
        dim_head (int, optional): Dimension per attention head. Default is 64.
        dropout (float, optional): Dropout rate for Transformer. Default is 0.
        emb_dropout (float, optional): Dropout rate for embedding layers. Default is 0.

    Forward Args:
        x (torch.Tensor): Input tensor of shape `(batch_size, channels, seq_len)`,
            where `batch_size` is the number of samples, `channels` is the number
            of EEG channels, and `seq_len` is the number of time steps.

    Returns:
        torch.Tensor: Output tensor of shape `(batch_size, num_patches + 3, dim)`,
            where `num_patches` is `seq_len // patch_len`, and `dim` is the Transformer embedding size.
    """

    def __init__(self, seq_len, patch_len, num_classes, dim, depth, heads, mlp_dim, channels=12,
                 dim_head=64, dropout=0., emb_dropout=0., is_anomaly_transformer: bool = False):
        """
        The encoder of CRT
        """
        super().__init__()

        assert seq_len % (4 * patch_len) == 0, \
            'The seq_len should be 4 * n * patch_len, or there must be patch with both magnitude and phase data.'

        num_patches = seq_len // patch_len
        patch_dim = channels * patch_len

        self.to_patch = nn.Sequential(Rearrange('b c (n p1) -> b n c p1', p1=patch_len),
                                      Rearrange('b n c p1 -> (b n) c p1'))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 3, dim))
        self.modal_embedding = nn.Parameter(torch.randn(3, 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 3, dim))
        self.dropout = nn.Dropout(emb_dropout)
        if is_anomaly_transformer:
            attention_layers = [EncoderLayer(AttentionLayer(
                AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                d_model, n_heads), d_model, d_ff, dropout=dropout, activation=activation) for l in range(e_layers)]

            self.transformer = AnomalyEncoder(attention_layers, norm_layer=torch.nn.LayerNorm(dim_head))
        else:
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.cnn1 = cnn_extractor(dim=channels, input_plane=dim // 8)  # For temporal data

    def forward(self, x):
        batch, _, time_steps = x.shape

        t = self.to_patch(x)
        patch2seq = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                  Rearrange('(b n) c 1 -> b n c', b=batch))

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=batch)

        x = torch.cat((cls_tokens[:, 0:1, :], patch2seq(self.cnn1(t))), dim=1)

        b, t, c = x.shape
        t_token_idx = 0
        x[:] += self.modal_embedding[:1]
        x += self.pos_embedding[:, : t]
        x = self.dropout(x)
        t_token = self.transformer(x)
        return t_token


def TFR_Encoder(seq_len, patch_len, dim, num_class, in_dim, is_unidomain, is_anomaly):
    if is_unidomain:
        vit = TransformerEEG(seq_len=seq_len,
                             patch_len=patch_len,
                             num_classes=num_class,
                             dim=dim,
                             depth=6,
                             heads=8,
                             mlp_dim=dim,
                             dropout=0.2,
                             emb_dropout=0.1,
                             channels=in_dim)
    else:
        vit = TFR(seq_len=seq_len,
                  patch_len=patch_len,
                  num_classes=num_class,
                  dim=dim,
                  depth=6,
                  heads=8,
                  mlp_dim=dim,
                  dropout=0.2,
                  emb_dropout=0.1,
                  channels=in_dim,
                  is_anomaly=is_anomaly)
    logger.debug("Transformer Encoder = {}".format(vit))
    return vit


class CRT(nn.Module):
    def __init__(
            self,
            encoder,
            decoder_dim,
            decoder_depth=2,
            decoder_heads=8,
            decoder_dim_head=64,
            patch_len=20,
            in_dim=12,
            is_unidomain: bool = False,
            is_anomaly_attention: bool = False
    ):
        super().__init__()
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch = encoder.to_patch
        pixel_values_per_patch = in_dim * patch_len

        self.is_unidomain = is_unidomain
        self.is_anomaly_attention = is_anomaly_attention

        # decoder parameters
        self.modal_embedding = self.encoder.modal_embedding
        self.mask_token = nn.Parameter(torch.randn(3, decoder_dim))
        self.decoder = Transformer(dim=decoder_dim,
                                   depth=decoder_depth,
                                   heads=decoder_heads,
                                   dim_head=decoder_dim_head,
                                   mlp_dim=decoder_dim)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.ModuleList([nn.Linear(decoder_dim, pixel_values_per_patch) for i in range(3)])
        self.projs = nn.ModuleList([nn.Linear(decoder_dim, decoder_dim) for i in range(2)])

    def IDC_loss(self, tokens, encoded_tokens):
        '''
        :param tokens: tokens before Transformer
        :param encoded_tokens: tokens after Transformer
        :return:
        '''
        B, T, D = tokens.shape
        tokens, encoded_tokens = F.normalize(tokens, dim=-1), F.normalize(encoded_tokens, dim=-1)
        encoded_tokens = encoded_tokens.transpose(2, 1)
        cross_mul = torch.exp(torch.matmul(tokens, encoded_tokens))
        mask = (1 - torch.eye(T)).unsqueeze(0).to(tokens.device)
        cross_mul = cross_mul * mask
        return torch.log(cross_mul.sum(-1).sum(-1)).mean(-1)

    def forward(self, x, mask_ratio=0, beta=1e-4):
        idc_loss, recon_loss, association_dict = self.__forward_multimodal(mask_ratio, x) if not self.is_unidomain \
            else self.__forward_unimodal(mask_ratio, x)
        return recon_loss + beta * idc_loss, association_dict

    def __forward_multimodal(self, mask_ratio, x):
        device = x.device
        patches = self.to_patch[0](x)
        batch, num_patches, c, length = patches.shape
        num_masked = int(mask_ratio * num_patches)
        # masked_indices1: masked index of temporal features
        # masked_indices2: masked index of spectral features
        rand_indices1 = torch.randperm(num_patches // 2, device=device)
        masked_indices1 = rand_indices1[: num_masked // 2].sort()[0]
        unmasked_indices1 = rand_indices1[num_masked // 2:].sort()[0]
        rand_indices2 = torch.randperm(num_patches // 4, device=device)
        masked_indices2, unmasked_indices2 = rand_indices2[: num_masked // 4].sort()[0], \
            rand_indices2[num_masked // 4:].sort()[0]
        rand_indices = torch.cat((masked_indices1, unmasked_indices1,
                                  masked_indices2 + num_patches // 2, unmasked_indices2 + num_patches // 2,
                                  masked_indices2 + num_patches // 4 * 3, unmasked_indices2 + num_patches // 4 * 3))
        masked_num_t, masked_num_f = masked_indices1.shape[0], 2 * masked_indices2.shape[0]
        unmasked_num_t, unmasked_num_f = unmasked_indices1.shape[0], 2 * unmasked_indices2.shape[0]
        # t, m, p refer to temporal, magnitude, phase
        tpatches = patches[:, : num_patches // 2, :, :]
        mpatches, ppatches = patches[:, num_patches // 2: num_patches * 3 // 4, :, :], patches[:, -num_patches // 4:, :,
                                                                                       :]
        # 1. Generate tokens from patches via CNNs.
        unmasked_tpatches = tpatches[:, unmasked_indices1, :, :]
        unmasked_mpatches, unmasked_ppatches = mpatches[:, unmasked_indices2, :, :], ppatches[:, unmasked_indices2, :,
                                                                                     :]
        t_tokens, m_tokens, p_tokens = self.to_patch[1](unmasked_tpatches), self.to_patch[1](unmasked_mpatches), \
            self.to_patch[1](unmasked_ppatches)
        # token shape = B,
        t_tokens, m_tokens, p_tokens = self.encoder.cnn1(t_tokens), self.encoder.cnn2(m_tokens), self.encoder.cnn3(
            p_tokens)
        Flat = nn.Sequential(nn.AdaptiveAvgPool1d(1),  # fixme: 64怎么来的？
                             Rearrange('(b n) c 1 -> b n c', b=batch))
        t_tokens, m_tokens, p_tokens = Flat(t_tokens), Flat(m_tokens), Flat(p_tokens)
        ori_tokens = torch.cat((t_tokens, m_tokens, p_tokens), 1).clone()
        # 2. Add three cls_tokens before temporal, magnitude and phase tokens.
        cls_tokens = repeat(self.encoder.cls_token, '() n d -> b n d', b=batch)
        tokens = torch.cat((cls_tokens[:, 0:1, :], t_tokens,
                            cls_tokens[:, 1:2, :], m_tokens,
                            cls_tokens[:, 2:3, :], p_tokens), dim=1)
        # 3. Generate Positional Embeddings.
        t_idx, m_idx, p_idx = num_patches // 2 - 1, num_patches * 3 // 4 - 1, num_patches - 1
        pos_embedding = torch.cat((self.encoder.pos_embedding[:, 0:1, :],
                                   self.encoder.pos_embedding[:, unmasked_indices1 + 1, :],
                                   self.encoder.pos_embedding[:, t_idx + 2: t_idx + 3],
                                   self.encoder.pos_embedding[:, unmasked_indices2 + t_idx + 3, :],
                                   self.encoder.pos_embedding[:, m_idx + 3: m_idx + 4],
                                   self.encoder.pos_embedding[:, unmasked_indices2 + m_idx + 4, :]), dim=1)
        # 4. Generate Domain-type Embedding
        modal_embedding = torch.cat((repeat(self.modal_embedding[0], '1 d -> 1 n d', n=unmasked_num_t + 1),
                                     repeat(self.modal_embedding[1], '1 d -> 1 n d', n=unmasked_num_f // 2 + 1),
                                     repeat(self.modal_embedding[2], '1 d -> 1 n d', n=unmasked_num_f // 2 + 1)), dim=1)
        tokens = tokens + pos_embedding + modal_embedding
        encoded_tokens, association_dict = self.encoder.transformer(tokens, unmask_indices=unmasked_indices1)
        t_idx, m_idx, p_idx = unmasked_num_t, unmasked_num_f // 2 + unmasked_num_t + 1, -1
        idc_loss = self.IDC_loss(self.projs[0](ori_tokens),
                                 self.projs[1](torch.cat(([encoded_tokens[:, 1: t_idx + 1],
                                                           encoded_tokens[:, t_idx + 2: m_idx + 1],
                                                           encoded_tokens[:, m_idx + 2:]]), dim=1)))
        decoder_tokens = encoded_tokens
        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens1 = repeat(self.mask_token[0], 'd -> b n d', b=batch, n=masked_num_t)
        mask_tokens2 = repeat(self.mask_token[1], 'd -> b n d', b=batch, n=masked_num_f // 2)
        mask_tokens3 = repeat(self.mask_token[2], 'd -> b n d', b=batch, n=masked_num_f // 2)
        mask_tokens = torch.cat((mask_tokens1, mask_tokens2, mask_tokens3), dim=1)
        # mask_tokens = repeat(self.mask_token[0], 'd -> b n d', b=batch, n=masked_num_f+masked_num_t)
        decoder_pos_emb = self.decoder_pos_emb(torch.cat(
            (masked_indices1, masked_indices2 + num_patches // 2, masked_indices2 + num_patches * 3 // 4)))
        mask_tokens = mask_tokens + decoder_pos_emb
        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((decoder_tokens, mask_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)
        if isinstance(decoded_tokens, tuple):
            decoded_tokens = decoded_tokens[0]
        pred_pixel_values_t = self.to_pixels[0](decoder_tokens[:, 1: t_idx + 1])
        pred_pixel_values_m = self.to_pixels[1](decoder_tokens[:, t_idx + 2: m_idx + 1])
        pred_pixel_values_p = self.to_pixels[2](decoder_tokens[:, m_idx + 2:])
        pred_pixel_values = torch.cat((pred_pixel_values_t, pred_pixel_values_m, pred_pixel_values_p), dim=1)
        # logger.debug(f'pred_pixel_values.min()={pred_pixel_values.min()}, '
        #              f'pred_pixel_values.max()={pred_pixel_values.max()},'
        #              f'patches.min()={patches.min()}, '
        #              f'patches.max()={patches.max()}')
        recon_loss = F.mse_loss(pred_pixel_values, rearrange(patches[:, rand_indices], 'b n c p -> b n (c p)'))

        final_representation = (encoded_tokens[:, 1: t_idx + 1].sum(dim=1) +
                                encoded_tokens[:, t_idx + 2: m_idx + 1].sum(dim=1) +
                                encoded_tokens[:, m_idx + 2:].sum(dim=1)) / (unmasked_num_f + unmasked_num_t)
        association_dict['final_representation'] = final_representation
        association_dict['local_input'] = ori_tokens
        association_dict['global_input'] = torch.mean(ori_tokens, dim=1)

        return idc_loss, recon_loss, association_dict

    def __forward_unimodal(self, mask_ratio, x):
        device = x.device
        patches = self.to_patch[0](x)
        batch, num_patches, c, length = patches.shape

        # ----- Generate masks -----
        num_masked = int(mask_ratio * num_patches)
        rand_indices = torch.randperm(num_patches // 2, device=device)
        masked_indices = rand_indices[: num_masked // 2].sort()[0]
        unmasked_indices = rand_indices[num_masked // 2:].sort()[0]
        masked_num = masked_indices.shape[0]
        unmasked_num = unmasked_indices.shape[0]
        # ----- ----- ----- -----

        unmasked_patches = patches[:, unmasked_indices, :, :]
        tokens = self.to_patch[1](unmasked_patches)
        tokens = self.encoder.cnn1(tokens)
        Flat = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                             Rearrange('(b n) c 1 -> b n c', b=batch))
        tokens = Flat(tokens)
        ori_tokens = tokens.clone()
        # 2. Add three cls_tokens before temporal, magnitude and phase tokens.
        cls_tokens = repeat(self.encoder.cls_token, '() n d -> b n d', b=batch)
        tokens = torch.cat((cls_tokens[:, 0:1, :], tokens), dim=1)
        pos_embedding = torch.cat((self.encoder.pos_embedding[:, 0:1, :],
                                   self.encoder.pos_embedding[:, unmasked_indices + 1, :]), dim=1)
        tokens = tokens + pos_embedding
        encoded_tokens = self.encoder.transformer(tokens)  # tokens.shape=(BATCH_SIZE ,6, 64)
        idc_loss = self.IDC_loss(self.projs[0](ori_tokens),
                                 self.projs[1](encoded_tokens[:, 1:]))
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
        # print(pred_pixel_values.min(), max(), patches.min(), max())
        recon_loss = F.mse_loss(pred_pixel_values, rearrange(patches[:, rand_indices], 'b n c p -> b n (c p)'))
        return idc_loss, recon_loss


class Model(nn.Module):
    """
    Main wrapper
    """

    def __init__(self, seq_len, patch_len, dim, num_class, in_dim: int, is_unidomain: bool = False,
                 is_anomaly: bool = False, **kwargs):
        """
        :param in_dim: input channel number of CNN extractors
        :param seq_len: fixme: what does it mean?
        """
        super(Model, self).__init__()
        self.encoder = TFR_Encoder(seq_len=seq_len,
                                   patch_len=patch_len,
                                   dim=dim,
                                   num_class=num_class,
                                   in_dim=in_dim,
                                   is_unidomain=is_unidomain, is_anomaly=is_anomaly)
        self.crt = CRT(encoder=self.encoder,
                       decoder_dim=dim,
                       in_dim=in_dim,
                       patch_len=patch_len,
                       is_unidomain=is_unidomain)

    def forward(self, x, ssl=True, ratio=0):
        """
        :param x: input, where the amplitude and spectral have been concatenated
        """
        if ssl == False:
            features = self.encoder(x)
            return features
        return self.crt(x, mask_ratio=ratio)


if __name__ == '__main__':
    # ----- Test main wrapper ----
    is_unidomain = False
    is_anomaly = False

    model = Model(seq_len=400, patch_len=10, dim=64, num_class=2, in_dim=61, is_unidomain=is_unidomain,
                  is_anomaly=is_anomaly).cuda()
    x = torch.rand([64, 61, 200]) if is_unidomain else torch.randn([64, 61, 400]).cuda()
    y = model(x, ssl=True)
    print(y.shape)
    # kwargs = {'ssl': True}
    # summary(model, (61, 200), device='cpu')
    # ----- ----- -----

    # # ----- Test Transformer architecture only -----
    # transformer_encoder = TransformerEEG(seq_len=200, patch_len=10, num_classes=-1, dim=64, depth=6, heads=8,
    #                                      mlp_dim=64,
    #                                      dropout=0.2, emb_dropout=0.1, channels=61, is_anomaly_transformer=True)
    # x = torch.rand([64, 61, 200])
    # y = transformer_encoder(x)
    # # ----- ----------
