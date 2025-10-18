import torch, cv2, numpy as np
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt

def gradcam_on_image(model, img_tensor, target_class=None, layer_name=None):
    # model: eval mode, img_tensor: [1,C,H,W]
    activations = []
    gradients = []

    if layer_name is None:
        # pick last conv layer automatically for efficientnet/resnet
        layer = None
        for n,m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d): layer = m
        layer_name = layer

    def fwd_hook(_, __, output): activations.append(output.detach())
    def bwd_hook(_, grad_in, grad_out): gradients.append(grad_out[0].detach())

    handle_f = layer_name.register_forward_hook(fwd_hook)
    handle_b = layer_name.register_backward_hook(bwd_hook)

    out = model(img_tensor)
    if target_class is None: target_class = out.argmax(1).item()
    score = out[:, target_class]
    model.zero_grad(); score.backward()

    acts = activations[0]          # [B, C, h, w]
    grads = gradients[0]           # [B, C, h, w]
    weights = grads.mean(dim=(2,3), keepdim=True)
    cam = (weights*acts).sum(dim=1)  # [B, h, w]
    cam = F.relu(cam)
    cam = (cam - cam.min())/(cam.max()-cam.min()+1e-6)
    handle_f.remove(); handle_b.remove()
    return cam[0].cpu().numpy()

def overlay_cam(rgb, cam):
    h,w,_ = rgb.shape
    cam_resized = cv2.resize(cam, (w,h))
    heatmap = cv2.applyColorMap((cam_resized*255).astype(np.uint8), cv2.COLORMAP_JET)
    over = (0.5*heatmap + 0.5*rgb[:,:,::-1]).astype(np.uint8)  # cv2 expects BGR
    return over[:,:,::-1]
