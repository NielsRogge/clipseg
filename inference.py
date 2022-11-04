import torch

from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize

supported_models = ["rd64-uni-refined", "rd64-uni", "rd16-uni"]

model_name = "rd16-uni"
if model_name not in supported_models:
    raise ValueError(f"Model {model_name} not supported")

# load model and weights
reduce_dim = 16 if "rd16" in model_name else 64
model = CLIPDensePredT(version='ViT-B/16', reduce_dim=reduce_dim, complex_trans_conv=True if "refined" in model_name else False)
model.eval();

# non-strict, because we only stored decoder weights (not CLIP weights)
checkpoint_path = f"/Users/nielsrogge/Documents/CLIPSeg/clipseg_weights/{model_name}.pth"

model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')), strict=False)

# for name, param in model.named_parameters():
#     print(name, param.shape)

image_transforms = Compose([
    ToTensor(),
    # Resize((224, 224)),
    Resize((352,352)),
])

image = Image.open("/Users/nielsrogge/Documents/cats.jpg").convert("RGB")
pixel_values = image_transforms(image).unsqueeze(0)

prompts = ['a glass', 'something to fill', 'wood', 'a jar']

# NOTE: we hardcode the input_ids inside the model, so we don't use the prompts

outputs = model(pixel_values.repeat(4,1,1,1), prompts)[0]

print("Shape of decoder outputs:", outputs.shape)
print("First values of decoder outputs:", outputs[0,0,:3,:3])

torch.save(model.state_dict(), f'/Users/nielsrogge/Documents/CLIPSeg/clip_plus_{model_name}.pth')