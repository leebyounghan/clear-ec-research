"""Run on Kubernetes: sparse SAM equivalence, masked leakage, distillation gradients."""
import copy
import sys
import unittest
import torch
from segment_anything.modeling.image_encoder import ImageEncoderViT
from cpsam_ssl_methods import token_forward, ema_update


class MaskedEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.encoder = ImageEncoderViT(img_size=32, patch_size=8, embed_dim=32,
            depth=2, num_heads=4, out_chans=16, use_rel_pos=True, window_size=0)
        # Nonzero learned position tables are essential for a meaningful test.
        for b in self.encoder.blocks:
            torch.nn.init.normal_(b.attn.rel_pos_h, std=.1)
            torch.nn.init.normal_(b.attn.rel_pos_w, std=.1)
        self.image = torch.randn(2, 3, 32, 32)

    def test_all_tokens_match_original_encoder(self):
        expected = self.encoder(self.image)
        tokens = token_forward(self.encoder, self.image)
        actual = self.encoder.neck(tokens.reshape(2, 4, 4, 32).permute(0, 3, 1, 2))
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_masked_pixels_cannot_affect_context_or_receive_gradients(self):
        visible = torch.tensor([[0, 1, 4, 5], [2, 3, 6, 7]])
        keep = torch.zeros(2, 16, dtype=torch.bool).scatter_(1, visible, True).reshape(2, 1, 4, 4)
        pixels = keep.repeat_interleave(8, 2).repeat_interleave(8, 3).expand(-1, 3, -1, -1)
        changed = torch.where(pixels, self.image, torch.randn_like(self.image)*100)
        a = token_forward(self.encoder, self.image, visible)
        b = token_forward(self.encoder, changed, visible)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        x = self.image.clone().requires_grad_()
        token_forward(self.encoder, x, visible, grad_checkpoint=True).square().mean().backward()
        assert (x.grad[~pixels] == 0).all()
        assert x.grad[pixels].abs().sum() > 0

    def test_ema_teacher_is_detached_and_updated(self):
        student = self.encoder
        teacher = copy.deepcopy(student)
        for p in teacher.parameters():
            p.requires_grad_(False)
        before = next(teacher.parameters()).clone()
        with torch.no_grad():
            next(student.parameters()).add_(1)
        ema_update(student, teacher, .9)
        torch.testing.assert_close(next(teacher.parameters()), before+.1)
        loss = (token_forward(student, self.image)-token_forward(teacher, self.image).detach()).square().mean()
        loss.backward()
        assert all(p.grad is None for p in teacher.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())

    def test_dino_head_can_be_strictly_copied_and_teacher_detached(self):
        sys.path.insert(0, 'experiments/cpsam_methods_20260914/upstream/dino')
        from vision_transformer import DINOHead
        from main_dino import DINOLoss
        student=DINOHead(32,64,hidden_dim=64,bottleneck_dim=16)
        teacher=DINOHead(32,64,hidden_dim=64,bottleneck_dim=16)
        teacher.load_state_dict(student.state_dict(),strict=True)
        criterion=DINOLoss(64,4,.04,.04,0,1)
        criterion.update_center=lambda values: None
        target=teacher(torch.randn(4,32));target.retain_grad()
        loss=criterion(student(torch.randn(8,32)),target,0)
        loss.backward()
        assert target.grad is None and torch.isfinite(loss)
        assert student.mlp[0].weight.grad.abs().sum()>0

    def test_official_predictor_mask_alignment_and_gradient(self):
        sys.path.insert(0,'experiments/cpsam_methods_20260914/upstream/ijepa')
        from src.models.vision_transformer import vit_predictor
        from src.masks.utils import apply_masks
        model=vit_predictor(num_patches=16,embed_dim=32,predictor_embed_dim=32,depth=2,num_heads=4)
        enc=[torch.tensor([[0,1,4,5],[2,3,6,7]])]
        masks=[torch.tensor([[10,11],[8,9]]),torch.tensor([[14,15],[12,13]])]
        context=token_forward(self.encoder,self.image,enc[0])
        target=apply_masks(token_forward(self.encoder,self.image).detach(),masks)
        pred=model(context,enc,masks)
        assert pred.shape==target.shape==(4,2,32)
        (pred-target).square().mean().backward()
        assert self.encoder.patch_embed.proj.weight.grad.abs().sum()>0


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
