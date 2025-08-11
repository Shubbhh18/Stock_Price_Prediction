"""
Utility transforms and simple chart-cropping helpers.
If you want advanced chart-to-OHLC parsing, use OpenCV routines (not included here).
"""
from PIL import Image
from torchvision import transforms

def get_transforms(image_size=224, train=False):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

# placeholder chart crop function (basic center crop)
def center_crop_pil(img: Image.Image, crop_fraction=0.9):
    w, h = img.size
    nw, nh = int(w*crop_fraction), int(h*crop_fraction)
    left = (w - nw)//2
    top = (h - nh)//2
    return img.crop((left, top, left+nw, top+nh))
