import torch
import clip
from torch import optim, nn
import time
import warnings
warnings.filterwarnings("ignore")
from DCLIP.data_generator import DanbooruIterableDataset
from PIL import Image
# Latest Update : 18 July 2022, 09:55 GMT+7

# TO ADD :
# Gradient Checkpointing
# Filter out bias from weight decay
# Decaying learning rate with cosine schedule
# Half-precision Adam statistics
# Half-precision stochastically rounded text encoder weights were used

# BATCH_SIZE must larger than 1
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-L/14", device=device, jit=False)  # Must set jit=False for training





# use your own data

ds = DanbooruIterableDataset(start=0, end=2996459, offset=100, )
dataloader = torch.utils.data.DataLoader(ds, num_workers=10,batch_size=40)

# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6,
                       weight_decay=0.001)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
EPOCH=20
# add your own code to track the training progress.
for epoch in range(EPOCH):
    print("当前训练到 epoch: " + str(epoch))
    for batch in dataloader:
        optimizer.zero_grad()

        images, texts = batch

        images = images.to(device)
        texts = texts.to(device)
        since = time.time()

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        time_elapsed = time.time() - since

        #print('\r',"loss:   "+str(total_loss.item())+"    cost: "+str(time_elapsed* 1000)+"ms",end='')
        print('\r', "loss:{:>2.10f}  cost:{:>3.2f} ms".format(total_loss.item(), time_elapsed * 1000), end='')

        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
    print('')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
    }, "model_checkpoint/dclip_"+str(epoch)+".pt")  # just change to your preferred folder/filename
    print("Saved model to model_checkpoint/dclip_"+str(epoch)+".pt")
    print('')