import torch
import mmrotate
from mmrotate.models.backbones import ViTAE_Window_NoShift_basic
from mmrotate.models.necks import  ReFPN

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 51, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'ViTAE_stages3_7': _cfg(),
}

# @register_model
def ViTAE_Window_NoShift_12_basic_stages4_14(**kwargs): # adopt performer for tokens to token
    # if pretrained:
        # kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = ViTAE_Window_NoShift_basic(RC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], NC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
                            NC_depth=[2, 2, 8, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], num_classes=51, **kwargs)
    model.default_cfg = default_cfgs['ViTAE_stages3_7']
    
    return model

def get_random_image():
    # Create a random tensor of size (3, 224, 224) with values between 0 and 1
    image_tensor = torch.rand(2, 3, 224, 224)

    # Alternatively, create a random tensor of size (224, 224, 3) with values between 0 and 1
    # image_tensor = torch.rand(224, 224, 3).permute(2, 0, 1)

    # Scale the tensor values to be between 0 and 255 (since most image libraries expect this range)
    image_tensor = image_tensor * 255

    # Convert the tensor to an unsigned 8-bit integer tensor (which is the standard format for images)
    image_tensor = image_tensor.float()

    return image_tensor


backbone = ViTAE_Window_NoShift_12_basic_stages4_14()
checkpoint_path = 'rsp-vitaev2-s-ckpt.pth'

checkpoint = torch.load(checkpoint_path)
# print(checkpoint.keys())

# for i in checkpoint.keys():
#     print(i)
# print(checkpoint['model'])
# load the model state dictionary from the checkpoint file
backbone.load_state_dict(checkpoint['model'], strict=False)

x = get_random_image()
x = backbone(x)

neck = ReFPN(in_channels=[512, 512], out_channels=256, num_outs=5)

x = neck(x)

print(len(x[0]))
