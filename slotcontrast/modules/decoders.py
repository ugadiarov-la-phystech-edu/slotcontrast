from typing import Dict, List, Optional, Tuple, Union

import einops
import timm.layers.pos_embed
import torch
from torch import nn

from slotcontrast.modules import networks, utils
from slotcontrast.modules.utils import Conv2dBlock, conv2d, PositionalEmbedding
from slotcontrast.utils import config_as_kwargs, make_build_fn


@make_build_fn(__name__, "decoder")
def build(config, name: str):
    if name == "SpatialBroadcastDecoder":
        output_transform = None
        if config.get("output_transform"):
            output_transform = utils.build_module(config.output_transform)

        return SpatialBroadcastDecoder(
            backbone=utils.build_module(config.backbone, default_group="networks"),
            output_transform=output_transform,
            **config_as_kwargs(config, ("backbone", "output_transform")),
        )
    elif name == "SpatialBroadcastDecoderV2":
        return SpatialBroadcastDecoderV2(image_size=config.image_size, obs_channels=config.outp_dim,
                                         hidden_size=config.hidden_size, slot_size=config.inp_dim,
                                         n_background_slots=config.n_background_slots)
    elif name == "SlotMixerDecoder":
        output_transform = None
        if config.get("output_transform"):
            output_transform = utils.build_module(config.output_transform)

        return SlotMixerDecoder(
            allocator=utils.build_module(config.allocator, default_group="networks"),
            renderer=utils.build_module(config.renderer, default_group="networks"),
            output_transform=output_transform,
            pos_embed_mode=config.get("pos_embed_mode", "add"),
            **config_as_kwargs(
                config, ("allocator", "renderer", "output_transform", "pos_embed_mode")
            ),
        )
    else:
        return None


class MLPDecoder(nn.Module):
    """Decoder that reconstructs independently for every position and slot."""

    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        hidden_dims: List[int],
        n_patches: int,
        activation: str = "relu",
        eval_output_size: Optional[Tuple[int]] = None,
        frozen: bool = False,
        n_background_slots: int = 0,
    ):
        super().__init__()
        self.outp_dim = outp_dim
        self.n_patches = n_patches
        self.eval_output_size = list(eval_output_size) if eval_output_size else None
        self.n_background_slots = n_background_slots

        self.mlp = networks.MLP(
            inp_dim, outp_dim + 1, hidden_dims, activation=activation, frozen=frozen
        )
        self.pos_emb = nn.Parameter(torch.randn(1, 1, n_patches, inp_dim) * inp_dim**-0.5)
        if frozen:
            self.pos_emb.requires_grad = False

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        bs, n_slots, dims = slots.shape

        if not self.training and self.eval_output_size is not None:
            pos_emb = timm.layers.pos_embed.resample_abs_pos_embed(
                self.pos_emb.squeeze(1),
                new_size=self.eval_output_size,
                num_prefix_tokens=0,
            ).unsqueeze(1)
        else:
            pos_emb = self.pos_emb

        slots = slots.view(bs, n_slots, 1, dims).expand(bs, n_slots, pos_emb.shape[2], dims)
        slots = slots + pos_emb

        recons, alpha = self.mlp(slots).split((self.outp_dim, 1), dim=-1)

        if self.n_background_slots > 0:
            alpha = alpha.clone()
            alpha[:, -self.n_background_slots:] = 0

        masks = torch.softmax(alpha, dim=1)
        recon = torch.sum(recons * masks, dim=1)

        return {"reconstruction": recon, "masks": masks.squeeze(-1)}


class SpatialBroadcastDecoder(nn.Module):
    """Decoder that reconstructs a spatial map independently per slot."""

    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        backbone: nn.Module,
        initial_size: Union[int, Tuple[int, int]] = 8,
        backbone_dim: Optional[int] = None,
        pos_embed: Optional[nn.Module] = None,
        output_transform: Optional[nn.Module] = None,
        n_background_slots: int = 0,
    ):
        super().__init__()
        self.outp_dim = outp_dim
        self.n_background_slots = n_background_slots
        if isinstance(initial_size, int):
            self.initial_size = (initial_size, initial_size)
        else:
            self.initial_size = initial_size

        if pos_embed is None:
            self.pos_embed = utils.CoordinatePositionEmbed(inp_dim, initial_size)
        else:
            self.pos_embed = pos_embed

        self.backbone = backbone

        if output_transform is None:
            if backbone_dim is None:
                raise ValueError("Need to provide backbone dim if output_transform is unspecified")
            self.output_transform = nn.Conv2d(backbone_dim, outp_dim + 1, 1, 1)
        else:
            self.output_transform = output_transform

        self.init_parameters()

    def init_parameters(self):
        if isinstance(self.output_transform, nn.Conv2d):
            utils.init_parameters(self.output_transform)

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        bs, n_slots, _ = slots.shape

        slots = einops.repeat(
            slots, "b s d -> (b s) d h w", h=self.initial_size[0], w=self.initial_size[1]
        )

        slots = self.pos_embed(slots)
        features = self.backbone(slots)
        outputs = self.output_transform(features)

        outputs = einops.rearrange(outputs, "(b s) ... -> b s ...", b=bs, s=n_slots)
        recons, alpha = einops.unpack(outputs, [[self.outp_dim], [1]], "b s * h w")

        if self.n_background_slots > 0:
            alpha = alpha.clone()
            alpha[:, -self.n_background_slots:] = 0

        masks = torch.softmax(alpha, dim=1)
        recon = torch.sum(recons * masks, dim=1)

        return {"reconstruction": recon, "masks": masks.squeeze(2), "slot_recons": recons}


class SpatialBroadcastDecoderV2(nn.Module):
    def __init__(self, image_size, obs_channels, hidden_size, slot_size, n_background_slots=0):
        super().__init__()
        self._obs_size = image_size
        self._obs_channels = obs_channels
        self.n_background_slots = n_background_slots
        self._decoder = nn.Sequential(
            Conv2dBlock(slot_size, hidden_size, 5, 1, 2),
            Conv2dBlock(hidden_size, hidden_size, 5, 1, 2),
            Conv2dBlock(hidden_size, hidden_size, 5, 1, 2),
            conv2d(hidden_size, obs_channels + 1, 3, 1, 1),
        )
        self._pos_emb = PositionalEmbedding(image_size, slot_size)

    def _spatial_broadcast(self, slots):
        slots = slots.unsqueeze(-1).unsqueeze(-1)
        return slots.repeat(1, 1, self._obs_size, self._obs_size)

    def forward(self, slots):
        B, N, _ = slots.shape
        # [batch_size * num_slots, d_slots]
        slots = slots.flatten(0, 1)
        # [batch_size * num_slots, d_slots, obs_size, obs_size]
        slots = self._spatial_broadcast(slots)
        out = self._decoder(self._pos_emb(slots))
        img_slots, masks = out[:, : self._obs_channels], out[:, -1:]
        img_slots = img_slots.view(
            B, N, self._obs_channels, self._obs_size, self._obs_size
        )
        masks = masks.view(B, N, 1, self._obs_size, self._obs_size)
        if self.n_background_slots > 0:
            masks = masks.clone()
            masks[:, -self.n_background_slots:] = 0
        masks = masks.softmax(dim=1)
        return {
            "reconstruction": torch.sum(img_slots * masks, dim=1),
            "masks": masks.squeeze(dim=2),
            "slot_recons": img_slots,
        }


class SlotMixerDecoder(nn.Module):
    """Slot mixer decoder reconstructing jointly over all slots, but independent per position.

    Introduced in Sajjadi et al., 2022: Object Scene Representation Transformer,
    http://arxiv.org/abs/2206.06922
    """

    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        embed_dim: int,
        n_patches: int,
        allocator: nn.Module,
        renderer: nn.Module,
        renderer_dim: Optional[int] = None,
        output_transform: Optional[nn.Module] = None,
        pos_embed_mode: Optional[str] = None,
        use_layer_norms: bool = False,
        norm_memory: bool = True,
        temperature: Optional[float] = None,
        eval_output_size: Optional[Tuple[int]] = None,
    ):
        super().__init__()
        self.allocator = allocator
        self.renderer = renderer
        self.eval_output_size = list(eval_output_size) if eval_output_size else None

        att_dim = max(embed_dim, inp_dim)
        self.scale = att_dim**-0.5 if temperature is None else temperature**-1
        self.to_q = nn.Linear(embed_dim, att_dim, bias=False)
        self.to_k = nn.Linear(inp_dim, att_dim, bias=False)

        if use_layer_norms:
            self.norm_k = nn.LayerNorm(inp_dim, eps=1e-5)
            self.norm_q = nn.LayerNorm(embed_dim, eps=1e-5)
            self.norm_memory = norm_memory
            if norm_memory:
                self.norm_memory = nn.LayerNorm(inp_dim, eps=1e-5)
            else:
                self.norm_memory = nn.Identity()
        else:
            self.norm_k = nn.Identity()
            self.norm_q = nn.Identity()
            self.norm_memory = nn.Identity()

        if output_transform is None:
            if renderer_dim is None:
                raise ValueError("Need to provide render_mlp_dim if output_transform is unspecified")
            self.output_transform = nn.Linear(renderer_dim, outp_dim)
        else:
            self.output_transform = output_transform

        if pos_embed_mode is not None and pos_embed_mode not in ("none", "add", "concat"):
            raise ValueError("If set, `pos_embed_mode` should be 'none', 'add' or 'concat'")
        self.pos_embed_mode = pos_embed_mode
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, embed_dim) * embed_dim**-0.5)
        self.init_parameters()

    def init_parameters(self):
        layers = [self.to_q, self.to_k]
        if isinstance(self.output_transform, nn.Linear):
            layers.append(self.output_transform)
        utils.init_parameters(layers, "xavier_uniform")

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.training and self.eval_output_size is not None:
            pos_emb = timm.layers.pos_embed.resample_abs_pos_embed(
                self.pos_emb,
                new_size=self.eval_output_size,
                num_prefix_tokens=0,
            )
        else:
            pos_emb = self.pos_emb

        pos_emb = pos_emb.expand(len(slots), -1, -1)
        memory = self.norm_memory(slots)
        query_features = self.allocator(pos_emb, memory=memory)
        q = self.to_q(self.norm_q(query_features))  # B x P x D
        k = self.to_k(self.norm_k(slots))  # B x S x D

        dots = torch.einsum("bpd, bsd -> bps", q, k) * self.scale
        attn = dots.softmax(dim=-1)

        mixed_slots = torch.einsum("bps, bsd -> bpd", attn, slots)  # B x P x D

        if self.pos_embed_mode == "add":
            mixed_slots = mixed_slots + pos_emb
        elif self.pos_embed_mode == "concat":
            mixed_slots = torch.cat((mixed_slots, pos_emb), dim=-1)

        features = self.renderer(mixed_slots)
        recons = self.output_transform(features)

        return {"reconstruction": recons, "masks": attn.transpose(-2, -1)}
