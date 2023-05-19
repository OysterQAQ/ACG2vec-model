import torch
import clip
from torch import optim, nn
import time
import warnings
warnings.filterwarnings("ignore")
from DCLIP.data_generator import DanbooruIterableDataset
from PIL import Image
# Latest Update : 18 July 2022, 09:55 GMT+7
import gc
# TO ADD :
# Gradient Checkpointing
# Filter out bias from weight decay
# Decaying learning rate with cosine schedule
# Half-precision Adam statistics
# Half-precision stochastically rounded text encoder weights were used

# BATCH_SIZE must larger than 1

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-L/14", device=device, jit=False)  # Must set jit=False for training



# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()






checkpoint = torch.load("/Volumes/Data/oysterqaq/Desktop/dclip_7.pt",map_location=torch.device('cpu'))

model.load_state_dict(checkpoint['model_state_dict'])
if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16
del checkpoint
gc.collect()

image = preprocess(Image.open("/Volumes/Data/oysterqaq/Desktop/3.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["Azur Lane", "3 girl with sword", "8 ninja", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    print(text_features)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

torch.save(model, "/Volumes/Data/oysterqaq/Desktop/dclip_2023_05_18.pt")

#torch.save(model, "/Volumes/Data/oysterqaq/Desktop/dclip_2023_05_18.pt")
