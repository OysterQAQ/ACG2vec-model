import torch
import clip
from torch import optim, nn
import time
import warnings
warnings.filterwarnings("ignore")
from DCLIP.data_generator import DanbooruIterableDataset,PixivIterableDataset
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
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-L/14", device=device, jit=False)  # Must set jit=False for training





# use your own data

ds = DanbooruIterableDataset(start=0, end=2996459, offset=2000, )
ds_pixiv = DanbooruIterableDataset()
dataloader = torch.utils.data.DataLoader(ds, num_workers=10,batch_size=40)
dataloader_pixiv = torch.utils.data.DataLoader(ds, num_workers=10,batch_size=40)

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
optimizer = optim.Adam(model.parameters(), lr=1e-6, betas=(0.9, 0.98), eps=1e-6,
                       weight_decay=0.001)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
EPOCH=7
batchs=100079
batchs_pixiv=0
checkpoint = torch.load("model_checkpoint/dclip_7.pt")

# Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"]
#checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
#checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
#checkpoint['model_state_dict']["vocab_size"] = model.vocab_size


model.load_state_dict(checkpoint['model_state_dict'])
del checkpoint
gc.collect()
for epoch in range(0,EPOCH):
    print("当前训练到 epoch: " + str(epoch))
    batch_index = 1
    for batch in dataloader_pixiv:
        batch_index+=1
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
        print("\r{:>10d}/{:<10d} loss:{:>2.10f}  cost:{:<3.2f}ms    ".format(batch_index,batchs_pixiv if batchs_pixiv is not None else 0,total_loss.item(), time_elapsed * 1000), end='')

        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
    batchs_pixiv=batch_index
    print('')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
    }, "model_checkpoint/dclip_7_pixiv"+str(epoch)+".pt")  # just change to your preferred folder/filename
    print("Saved model to model_checkpoint/dclip_7_pixiv"+str(epoch)+".pt")
    print('')

#
# for epoch in range(4,EPOCH):
#     print("当前训练到 epoch: " + str(epoch))
#     batch_index = 1
#     for batch in dataloader:
#         batch_index+=1
#         optimizer.zero_grad()
#
#         images, texts = batch
#
#         images = images.to(device)
#         texts = texts.to(device)
#         since = time.time()
#
#         logits_per_image, logits_per_text = model(images, texts)
#
#         ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
#
#         total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
#         time_elapsed = time.time() - since
#
#         #print('\r',"loss:   "+str(total_loss.item())+"    cost: "+str(time_elapsed* 1000)+"ms",end='')
#         print("\r{:>10d}/{:<10d} loss:{:>2.10f}  cost:{:<3.2f}ms    ".format(batch_index,batchs if batchs is not None else 0,total_loss.item(), time_elapsed * 1000), end='')
#
#         total_loss.backward()
#         if device == "cpu":
#             optimizer.step()
#         else:
#             convert_models_to_fp32(model)
#             optimizer.step()
#             clip.model.convert_weights(model)
#     batchs=batch_index
#     print('')
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': total_loss,
#     }, "model_checkpoint/dclip_"+str(epoch)+".pt")  # just change to your preferred folder/filename
#     print("Saved model to model_checkpoint/dclip_"+str(epoch)+".pt")
#     print('')