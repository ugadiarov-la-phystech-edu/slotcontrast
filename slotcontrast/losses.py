from typing import Any, Dict, Optional, Tuple

import einops
import torch
from torch import nn

from slotcontrast import modules, utils


@utils.make_build_fn(__name__, "loss")
def build(config, name: str):
    target_transform = None
    if config.get("target_transform"):
        target_transform = modules.build_module(config.get("target_transform"))

    cls = utils.get_class_by_name(__name__, name)
    if cls is not None:
        return cls(
            target_transform=target_transform,
            **utils.config_as_kwargs(config, ("target_transform",)),
        )
    else:
        raise ValueError(f"Unknown loss `{name}`")


class Loss(nn.Module):
    """Base class for loss functions.

    Args:
        video_inputs: If true, assume inputs contain a time dimension.
        patch_inputs: If true, assume inputs have a one-dimensional patch dimension. If false,
            assume inputs have height, width dimensions.
        pred_dims: Dimensions [from, to) of prediction tensor to slice. Useful if only a
            subset of the predictions should be used in the loss, i.e. because the other dimensions
            are used in other losses.
        remove_last_n_frames: Number of frames to remove from the prediction before computing the
            loss. Only valid with video inputs. Useful if the last frame does not have a
            correspoding target.
        target_transform: Transform that can optionally be applied to the target.
    """

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        video_inputs: bool = False,
        patch_inputs: bool = True,
        keep_input_dim: bool = False,
        pred_dims: Optional[Tuple[int, int]] = None,
        remove_last_n_frames: int = 0,
        target_transform: Optional[nn.Module] = None,
        input_key: Optional[str] = None,
    ):
        super().__init__()
        self.pred_path = pred_key.split(".")
        self.target_path = target_key.split(".")
        self.video_inputs = video_inputs
        self.patch_inputs = patch_inputs
        self.keep_input_dim = keep_input_dim
        self.input_key = input_key
        self.n_expected_dims = (
            2 + (1 if patch_inputs or keep_input_dim else 2) + (1 if video_inputs else 0)
        )

        if pred_dims is not None:
            assert len(pred_dims) == 2
            self.pred_dims = slice(pred_dims[0], pred_dims[1])
        else:
            self.pred_dims = None

        self.remove_last_n_frames = remove_last_n_frames
        if remove_last_n_frames > 0 and not video_inputs:
            raise ValueError("`remove_last_n_frames > 0` only valid with `video_inputs==True`")

        self.target_transform = target_transform
        self.to_canonical_dims = self.get_dimension_canonicalizer()

    def get_dimension_canonicalizer(self) -> torch.nn.Module:
        """Return a module which reshapes tensor dimensions to (batch, n_positions, n_dims)."""
        if self.video_inputs:
            if self.patch_inputs:
                pattern = "B F P D -> B (F P) D"
            elif self.keep_input_dim:
                return torch.nn.Identity()
            else:
                pattern = "B F D H W -> B (F H W) D"
        else:
            if self.patch_inputs:
                return torch.nn.Identity()
            else:
                pattern = "B D H W -> B (H W) D"

        return einops.layers.torch.Rearrange(pattern)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        target = utils.read_path(outputs, elements=self.target_path, error=False)
        if target is None:
            target = utils.read_path(inputs, elements=self.target_path)

        target = target.detach()

        if self.target_transform:
            with torch.no_grad():
                if self.input_key is not None:
                    target = self.target_transform(target, inputs[self.input_key])
                else:
                    target = self.target_transform(target)

        # Convert to dimension order (batch, positions, dims)
        target = self.to_canonical_dims(target)

        return target

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        prediction = utils.read_path(outputs, elements=self.pred_path)
        if prediction.ndim != self.n_expected_dims:
            raise ValueError(
                f"Prediction has {prediction.ndim} dimensions (and shape {prediction.shape}), but "
                f"expected it to have {self.n_expected_dims} dimensions."
            )

        if self.video_inputs and self.remove_last_n_frames > 0:
            prediction = prediction[:, : -self.remove_last_n_frames]

        # Convert to dimension order (batch, positions, dims)
        prediction = self.to_canonical_dims(prediction)

        if self.pred_dims:
            prediction = prediction[..., self.pred_dims]

        return prediction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Implement in subclasses")

    def _expand_frame_mask_to_positions(
        self, frame_padding_mask: torch.Tensor, n_positions: int
    ) -> torch.Tensor:
        """Broadcast a (B, T) padding mask to the canonical (B, n_positions) position mask.

        Returns a float mask with 1.0 at valid positions and 0.0 at padded positions, with shape
        ``(B, n_positions)``. Caller adds the final dim for broadcasting over feature/class dims.
        """
        if not self.video_inputs:
            raise ValueError("frame_padding_mask is only supported with video_inputs=True.")
        b, t = frame_padding_mask.shape
        if n_positions % t != 0:
            raise ValueError(
                f"Cannot broadcast frame_padding_mask of shape (B={b}, T={t}) to {n_positions} "
                f"canonical positions: not divisible by T."
            )
        per_frame = n_positions // t
        valid = (~frame_padding_mask).to(torch.float32)
        return valid[:, :, None].expand(b, t, per_frame).reshape(b, n_positions)


class TorchLoss(Loss):
    """Wrapper around PyTorch loss functions."""

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        loss: str,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        loss_kwargs = loss_kwargs if loss_kwargs is not None else {}
        if hasattr(torch.nn, loss):
            self.loss_fn = getattr(torch.nn, loss)(reduction="mean", **loss_kwargs)
            try:
                self.loss_fn_unreduced = getattr(torch.nn, loss)(reduction="none", **loss_kwargs)
            except TypeError:
                self.loss_fn_unreduced = None
        else:
            raise ValueError(f"Loss function torch.nn.{loss} not found")

        # Cross entropy loss wants dimension order (batch, classes, positions)
        self.positions_last = loss == "CrossEntropyLoss"

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        frame_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.positions_last:
            prediction = prediction.transpose(-2, -1)
            target = target.transpose(-2, -1)

        if frame_padding_mask is not None and self.loss_fn_unreduced is None:
            raise NotImplementedError(f"{self.loss_fn} does not support frame_padding_mask.")

        if frame_padding_mask is None:
            return self.loss_fn(prediction, target)

        # Masked-mean over valid positions. Per-element loss has shape `(B, n_positions, D)` after
        # canonicalization; build a `(B, n_positions, 1)` valid mask from the `(B, T)` frame mask.
        per_element = self.loss_fn_unreduced(prediction, target)
        b, n_pos, d = per_element.shape
        valid = self._expand_frame_mask_to_positions(frame_padding_mask, n_pos).to(per_element.dtype)
        valid = valid.to(per_element.device)[:, :, None]
        weighted = per_element * valid
        denom = valid.sum() * d
        return weighted.sum() / denom.clamp(min=1)


class MSELoss(TorchLoss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, loss="MSELoss", **kwargs)


class CrossEntropyLoss(TorchLoss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, loss="CrossEntropyLoss", **kwargs)


class Slot_Slot_Contrastive_Loss(Loss):
    def __init__(
        self,
        pred_key: str,
        target_key: str,
        temperature: float = 0.1,
        batch_contrast: bool = True,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.batch_contrast = batch_contrast

    def forward(self, slots, _, frame_padding_mask=None):
        if frame_padding_mask is not None and self.batch_contrast:
            # This case is not implemented yet.
            raise NotImplementedError(
                "Slot_Slot_Contrastive_Loss does not support full-episode padding when "
                "`batch_contrast=True`. Use `batch_contrast=False` for full-episode validation, "
                "or stay in subsequence mode."
            )
        slots = nn.functional.normalize(slots, p=2.0, dim=-1)
        if self.batch_contrast:
            slots = slots.split(1)  # [1xTxKxD]
            slots = torch.cat(slots, dim=-2)  # 1xTxK*BxD
        s1 = slots[:, :-1, :, :]
        s2 = slots[:, 1:, :, :]
        ss = torch.matmul(s1, s2.transpose(-2, -1)) / self.temperature
        B, T, S, _ = ss.shape
        ss = ss.reshape(B * T, S, S)
        target = torch.eye(S).expand(B * T, S, S).to(ss.device)
        if frame_padding_mask is None:
            return self.criterion(ss, target)

        # batch_contrast=False: each (video, pair) is its own contrastive item, so a per-video
        # validity mask is exactly the right granularity. Drop pairs whose ends are not both real.
        valid_per_frame = ~frame_padding_mask  # (B_orig, T_orig)
        pair_valid = valid_per_frame[:, :-1] & valid_per_frame[:, 1:]  # (B_orig, T_orig-1)
        pair_valid = pair_valid.reshape(-1).to(ss.device, dtype=ss.dtype)  # (B * T,)
        if pair_valid.sum() == 0:
            return ss.sum() * 0.0  # degenerate (e.g. T_i = 1): keep a zero connected to the graph

        # `cross_entropy(..., reduction='none')` with input (N, C, d1) and class-prob target of
        # the same shape returns (N, d1) — here (B*T, S). Match `reduction='mean'` semantics:
        # average over (valid_pairs * S) elements, not just valid_pairs.
        per_pair_loss = nn.functional.cross_entropy(ss, target, reduction="none")  # (B*T, S)
        s_dim = per_pair_loss.shape[-1]
        return (per_pair_loss * pair_valid[:, None]).sum() / (pair_valid.sum() * s_dim)


class DynamicsLoss(Loss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, **kwargs)
        self.criterion = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        rollout_length = prediction.shape[1]
        target = target[:, -rollout_length:]
        loss = self.criterion(prediction, target)
        return loss
