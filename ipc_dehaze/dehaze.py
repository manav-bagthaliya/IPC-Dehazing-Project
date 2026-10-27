"""IPC-Dehaze model wrapper: Predictor + Critic iterative code decoding."""

import numpy as np
import torch
import torch.nn.functional as F

from basicsr.archs.dehazeToken_arch import Critic, DehazeTokenNet
from basicsr.utils import img2tensor, parallel_decode, tensor2img
from ipc_dehaze.config import CRITIC_PATH, DEFAULT_MAX_SIZE, DEFAULT_NUM_ITERATIONS, PAD_MULTIPLE, PREDICTOR_PATH


def _resize(img, size=None, scale=None):
    """Bilinear resize of an NCHW tensor (same as nn.UpsamplingBilinear2d)."""
    return F.interpolate(img, size=size, scale_factor=scale, mode='bilinear', align_corners=True)


def mirror_pad(img, multiple=PAD_MULTIPLE):
    """Reflect-pad H and W of an NCHW tensor up to the next multiple of ``multiple``."""
    _, _, h, w = img.shape
    h_pad = (h // multiple + 1) * multiple - h
    w_pad = (w // multiple + 1) * multiple - w
    img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h + h_pad, :]
    img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w + w_pad]
    return img


class Dehazer:
    """Loads the networks once; ``dehaze`` then processes BGR uint8 images."""

    def __init__(self, predictor_path=PREDICTOR_PATH, critic_path=CRITIC_PATH, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.net_g = DehazeTokenNet(codebook_params=[64, 1024, 256], blk_depth=16, LQ_stage=True,
                                    predictor_name='swinLayer').to(self.device)
        self.net_g.load_state_dict(torch.load(predictor_path, map_location=self.device)['params'], strict=True)
        self.net_g.eval()

        self.net_critic = Critic().to(self.device)
        self.net_critic.load_state_dict(torch.load(critic_path, map_location=self.device)['params'], strict=True)
        self.net_critic.eval()

    def _tokens_to_logits(self, seq, h=0, w=0, critic=False):
        if critic:
            return self.net_critic(seq, h, w)
        return self.net_g.transformer(seq, critic)

    def _tokens_to_feats(self, seq):
        """seq: (b, 1, h, w) code indices -> codebook features."""
        return self.net_g.vqgan.quantize.get_codebook_entry(seq.long())

    def prepare(self, image_bgr, max_size=DEFAULT_MAX_SIZE, resize_to=0, force_size=None):
        """BGR uint8 -> NCHW float tensor at processing resolution (unpadded)."""
        img = img2tensor(image_bgr / 255.0).unsqueeze(0).to(self.device)
        _, _, h, w = img.shape

        if force_size:
            return _resize(img, size=force_size)
        if resize_to and max(h, w) < resize_to:
            img = _resize(img, scale=resize_to / max(h, w))
        if h * w >= max_size ** 2:
            img = _resize(img, scale=max_size / max(h, w))
        return img

    @torch.no_grad()
    def dehaze_tensor(self, img, num_iterations=DEFAULT_NUM_ITERATIONS):
        """Dehaze an NCHW tensor in [0, 1]; returns a tensor of the same size."""
        _, _, h_old, w_old = img.shape
        img = mirror_pad(img)

        # Encoder
        enc_feats = self.net_g.vqgan.multiscale_encoder(img.detach())[::-1]
        if not enc_feats:
            raise RuntimeError('Encoder produced no features')
        x = enc_feats[0]
        b, _, h, w = x.shape
        feat_to_quant = self.net_g.vqgan.before_quant(x)

        # Iterative predictor-critic code decoding, starting with every code masked
        mask_tokens = -1 * torch.ones(b, h * w, device=feat_to_quant.device).long()
        output_tokens, _ = parallel_decode.decode_critic(
            mask_tokens,
            feat_to_quant,
            self._tokens_to_logits,
            self._tokens_to_feats,
            num_iter=num_iterations,
        )
        quant_feats = self._tokens_to_feats(output_tokens[:, -1, :].reshape(b, 1, h, w))

        # Decoder, fusing encoder features back in at each scale
        x = self.net_g.vqgan.after_quant(quant_feats)
        for i in range(min(self.net_g.max_depth, len(enc_feats))):
            cur_res = self.net_g.gt_res // 2 ** self.net_g.max_depth * 2 ** i
            x = self.net_g.fuse_convs_dict[str(cur_res)](enc_feats[i].detach(), x, 1)
            x = self.net_g.vqgan.decoder_group[i](x)

        out = self.net_g.vqgan.out_conv(x)
        return out[..., :h_old, :w_old]

    def dehaze(self, image_bgr, num_iterations=DEFAULT_NUM_ITERATIONS, max_size=DEFAULT_MAX_SIZE,
               resize_to=0, force_size=None):
        """Dehaze a BGR uint8 image.

        Output size: ``force_size`` (h, w) if given; the upscaled size if ``resize_to``
        enlarged a small image; otherwise the original size.

        Returns:
            (dehazed_bgr, processed_input_bgr) - the second is the resized input the
            network actually saw, useful for showing users the processing size.
        """
        origin_h, origin_w = image_bgr.shape[:2]
        img = self.prepare(image_bgr, max_size, resize_to, force_size)
        out = self.dehaze_tensor(img, num_iterations)

        upscaled = bool(resize_to) and max(origin_h, origin_w) < resize_to
        if not force_size and not upscaled and out.shape[2:] != (origin_h, origin_w):
            out = _resize(out, size=(origin_h, origin_w))

        return tensor2img(out, rgb2bgr=True), tensor2img(img, rgb2bgr=True)


def looks_blank(image):
    """True if an output is near-uniform or almost all white/black (failed dehaze)."""
    return float(np.var(image)) < 1.0 or not 5 <= float(np.mean(image)) <= 250
