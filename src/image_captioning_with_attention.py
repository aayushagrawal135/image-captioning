# %%
#imports
import torch
import numpy as np
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from data_loader import FlickrDataset, get_data_loader
from plots_and_save import *
from model import get_model
from utils import *

# Initiate the Dataset and Dataloader
# setting the constants
data_location =  "../input/flickr8k"
BATCH_SIZE = 256
NUM_WORKER = 0

#defining the transform to be applied
transforms = T.Compose([
    T.Resize(226),                     
    T.RandomCrop(224),                 
    T.ToTensor(),                               
    T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])

#testing the dataset class
dataset =  FlickrDataset(
    root_dir = data_location+"/Images",
    captions_file = data_location+"/captions.txt",
    transform=transforms
)

#writing the dataloader
data_loader = get_data_loader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKER,
    shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Hyperparams
embed_size=300
vocab_size = len(dataset.vocab)
attention_dim=256
encoder_dim=2048
decoder_dim=512
learning_rate = 3e-4
num_epochs = 60
print_every = 30

def wrapper(model_name):

    model = get_model(len(dataset.vocab), model_name=model_name)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train(epoch):
        saved_count = 0
        model.train()
        loss_values = list()
        for idx, (image, captions) in enumerate(data_loader):
            image,captions = image.to(device),captions.to(device)

            # Zero the gradients.
            optimizer.zero_grad()

            # Feed forward
            outputs, attentions = model(image, captions)

            # Calculate the batch loss.
            targets = captions[:,1:]
            loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
            
            # Backward pass.
            loss.backward()

            # Update the parameters in the optimizer.
            optimizer.step()

            if (idx+1)%print_every == 0:
                print("Epoch: {} loss: {:.5f}".format(epoch,loss.item()))

                #generate the caption
                # model.eval()
                with torch.no_grad():
                    dataiter = iter(data_loader)
                    img, _ = next(dataiter)
                    # features = model.encoder(img[0:1].to(device))
                    features = model.encode(img[0:1].to(device))
                    caps, alphas = model.decoder.generate_caption(features,vocab=dataset.vocab)
                    
                    # model.forward()
                    
                    caption = ' '.join(caps)
                    save_image(img[0], saved_count, model_name, title=caption, epoch = epoch)
                    caps, alphas = get_caps_from(img[0].unsqueeze(0), model, dataset.vocab)
                    plot_attention(img[0], caps, alphas, epoch, saved_count, model_name)
                saved_count = saved_count + 1
                loss_values.append(loss.item())
        model.save(model, epoch)
        return np.asarray(loss_values).mean()

    loss_values = list()
    for epoch in range(1, num_epochs+1):
        loss = train(epoch)
        loss_values.append(loss)
        print(f"Loss list: {loss_values}")
    plot_loss(loss_values, model_name=model_name)

model_names = ["resnet50", "vgg16"]
for model_name in model_names:
    wrapper(model_name)
