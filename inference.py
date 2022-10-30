import torch

from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize

# load model
model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
model.eval();

# non-strict, because we only stored decoder weights (not CLIP weights)
model.load_state_dict(torch.load('/Users/nielsrogge/Documents/CLIPSeg/clipseg_weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False)

# for name, param in model.named_parameters():
#     print(name, param.shape)

image_transforms = Compose([
    ToTensor(),
    Resize((224, 224)),
])

image = Image.open("/Users/nielsrogge/Documents/cats.jpg").convert("RGB")
pixel_values = image_transforms(image).unsqueeze(0)

prompts = ['a glass', 'something to fill', 'wood', 'a jar']

# NOTE: we hardcode the input_ids inside the model, so we don't use the prompts

outputs = model(pixel_values.repeat(4,1,1,1), prompts)[0]

print("Shape of decoder outputs:", outputs.shape)
print("First values of decoder outputs:", outputs[0,0,:3,:3])

torch.save(model.state_dict(), '/Users/nielsrogge/Documents/CLIPSeg/test.pth')