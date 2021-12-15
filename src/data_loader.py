# %% [markdown]
# # <h2>1) Exploring the dataset</h2>
# <p>Reading the image data and their corresponding captions from the flick dataset folder. Showing the image and captions to get the insighs of the data. Dowload link for the dataset used <a href="https://www.kaggle.com/adityajn105/flickr8k">here</a></p>

# %% [code] {"execution":{"iopub.status.busy":"2021-12-12T18:55:33.218631Z","iopub.execute_input":"2021-12-12T18:55:33.219005Z","iopub.status.idle":"2021-12-12T18:55:34.017275Z","shell.execute_reply.started":"2021-12-12T18:55:33.218974Z","shell.execute_reply":"2021-12-12T18:55:34.016065Z"}}
#location of the data 
data_location =  "../input/flickr8k"
# !ls $data_location

# %% [code] {"execution":{"iopub.status.busy":"2021-12-12T18:55:34.019437Z","iopub.execute_input":"2021-12-12T18:55:34.019757Z","iopub.status.idle":"2021-12-12T18:55:34.153283Z","shell.execute_reply.started":"2021-12-12T18:55:34.019720Z","shell.execute_reply":"2021-12-12T18:55:34.152372Z"}}
#reading the text data 
import pandas as pd
caption_file = data_location + '/captions.txt'
df = pd.read_csv(caption_file)
# print("There are {} image to captions".format(len(df)))
# df.head(7)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-12T18:55:34.154774Z","iopub.execute_input":"2021-12-12T18:55:34.155105Z","iopub.status.idle":"2021-12-12T18:55:34.463009Z","shell.execute_reply.started":"2021-12-12T18:55:34.155073Z","shell.execute_reply":"2021-12-12T18:55:34.460605Z"}}
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#select any index from the whole dataset 
#single image has 5 captions
#so, select indx as: 1,6,11,16...
data_idx = 11

#eg path to be plot: ../input/flickr8k/Images/1000268201_693b08cb0e.jpg
image_path = data_location+"/Images/"+df.iloc[data_idx,0]
# img=mpimg.imread(image_path)
# plt.imshow(img)
# plt.show()

#image consits of 5 captions,
#showing all 5 captions of the image of the given idx 
# for i in range(data_idx,data_idx+5):
#     print("Caption:",df.iloc[i,1])


# %% [markdown]
# <h2>2) Writing the custom dataset</h2>
# <p>Writing the custom torch dataset class so, that we can abastract out the dataloading steps during the training and validation process</p>
# <p>Here, dataloader is created which gives the batch of image and its captions with following processing done:</p>
# 
# <li>caption word tokenized to unique numbers</li>
# <li>vocab instance created to store all the relivent words in the datasets</li>
# <li>each batch, caption padded to have same sequence length</li>
# <li>image resized to the desired size and converted into captions</li>
# 
# <br><p>In this way the dataprocessing is done, and the dataloader is ready to be used with <b>Pytorch</b></p>

# %% [code] {"execution":{"iopub.status.busy":"2021-12-12T18:55:34.465223Z","iopub.execute_input":"2021-12-12T18:55:34.465687Z","iopub.status.idle":"2021-12-12T18:55:36.807554Z","shell.execute_reply.started":"2021-12-12T18:55:34.465639Z","shell.execute_reply":"2021-12-12T18:55:36.806554Z"}}
#imports 
import os
from collections import Counter
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T

from PIL import Image

# %% [code] {"execution":{"iopub.status.busy":"2021-12-12T18:55:36.809871Z","iopub.execute_input":"2021-12-12T18:55:36.810201Z","iopub.status.idle":"2021-12-12T18:55:37.731339Z","shell.execute_reply.started":"2021-12-12T18:55:36.810154Z","shell.execute_reply":"2021-12-12T18:55:37.730010Z"}}
#using spacy for the better text tokenization 
spacy_eng = spacy.load('en_core_web_sm')

#example
text = "This is a good place to find a city"
[token.text.lower() for token in spacy_eng.tokenizer(text)]

# %% [code] {"execution":{"iopub.status.busy":"2021-12-12T18:55:37.733223Z","iopub.execute_input":"2021-12-12T18:55:37.733524Z","iopub.status.idle":"2021-12-12T18:55:37.743621Z","shell.execute_reply.started":"2021-12-12T18:55:37.733494Z","shell.execute_reply":"2021-12-12T18:55:37.742748Z"}}
class Vocabulary:
    def __init__(self,freq_threshold):
        #setting the pre-reserved tokens int to string tokens
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        
        #string to int tokens
        #its reverse dict self.itos
        self.stoi = {v:k for k,v in self.itos.items()}
        
        self.freq_threshold = freq_threshold
        
    def __len__(self): return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]
    
    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                #add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self,text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ]    

# %% [code] {"execution":{"iopub.status.busy":"2021-12-12T18:55:37.744713Z","iopub.execute_input":"2021-12-12T18:55:37.745134Z","iopub.status.idle":"2021-12-12T18:55:37.760250Z","shell.execute_reply.started":"2021-12-12T18:55:37.745095Z","shell.execute_reply":"2021-12-12T18:55:37.759278Z"}}
#testing the vicab class 
v = Vocabulary(freq_threshold=1)

v.build_vocab(["This is a good place to find a city"])
print(v.stoi)
print(v.numericalize("This is a good place to find a city here!!"))

# %% [code] {"execution":{"iopub.status.busy":"2021-12-12T18:55:37.761815Z","iopub.execute_input":"2021-12-12T18:55:37.762115Z","iopub.status.idle":"2021-12-12T18:55:37.772639Z","shell.execute_reply.started":"2021-12-12T18:55:37.762086Z","shell.execute_reply":"2021-12-12T18:55:37.771604Z"}}
class FlickrDataset(Dataset):
    """
    FlickrDataset
    """
    def __init__(self,root_dir,captions_file,transform=None,freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file)
        self.transform = transform
        
        #Get image and caption colum from the dataframe
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        #Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())
        
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir,img_name)
        img = Image.open(img_location).convert("RGB")
        
        #apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)
        
        #numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]
        
        return img, torch.tensor(caption_vec)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-12T18:55:37.773762Z","iopub.execute_input":"2021-12-12T18:55:37.774196Z","iopub.status.idle":"2021-12-12T18:55:37.788850Z","shell.execute_reply.started":"2021-12-12T18:55:37.774142Z","shell.execute_reply":"2021-12-12T18:55:37.787695Z"}}
#defing the transform to be applied
transforms = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

# %% [code] {"execution":{"iopub.status.busy":"2021-12-12T18:55:37.790024Z","iopub.execute_input":"2021-12-12T18:55:37.790477Z","iopub.status.idle":"2021-12-12T18:55:37.800203Z","shell.execute_reply.started":"2021-12-12T18:55:37.790444Z","shell.execute_reply":"2021-12-12T18:55:37.799284Z"}}
# def show_image(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated

# %% [code] {"execution":{"iopub.status.busy":"2021-12-12T18:55:37.801502Z","iopub.execute_input":"2021-12-12T18:55:37.801812Z","iopub.status.idle":"2021-12-12T18:55:39.789092Z","shell.execute_reply.started":"2021-12-12T18:55:37.801783Z","shell.execute_reply":"2021-12-12T18:55:39.788130Z"}}
#testing the dataset class
dataset =  FlickrDataset(
    root_dir = data_location+"/Images",
    captions_file = data_location+"/captions.txt",
    transform=transforms
)



# img, caps = dataset[0]
# show_image(img,"Image")
# print("Token:",caps)
# print("Sentence:")
# print([dataset.vocab.itos[token] for token in caps.tolist()])

# %% [code] {"execution":{"iopub.status.busy":"2021-12-12T18:55:39.790357Z","iopub.execute_input":"2021-12-12T18:55:39.790627Z","iopub.status.idle":"2021-12-12T18:55:39.797719Z","shell.execute_reply.started":"2021-12-12T18:55:39.790592Z","shell.execute_reply":"2021-12-12T18:55:39.796959Z"}}
class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self,pad_idx,batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    
    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs,targets

# %% [code] {"execution":{"iopub.status.busy":"2021-12-12T18:55:39.799017Z","iopub.execute_input":"2021-12-12T18:55:39.799532Z","iopub.status.idle":"2021-12-12T18:55:39.813478Z","shell.execute_reply.started":"2021-12-12T18:55:39.799484Z","shell.execute_reply":"2021-12-12T18:55:39.812437Z"}}
#writing the dataloader
#setting the constants
BATCH_SIZE = 4
NUM_WORKER = 1

#token to represent the padding
pad_idx = dataset.vocab.stoi["<PAD>"]

# data_loader = DataLoader(
#     dataset=dataset,
#     batch_size=BATCH_SIZE,
#     num_workers=NUM_WORKER,
#     shuffle=True,
#     collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
# )

def get_data_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=shuffle,
    collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-12T18:55:39.826508Z","iopub.execute_input":"2021-12-12T18:55:39.827100Z","iopub.status.idle":"2021-12-12T18:55:40.934144Z","shell.execute_reply.started":"2021-12-12T18:55:39.827058Z","shell.execute_reply":"2021-12-12T18:55:40.932926Z"}}
#generating the iterator from the dataloader
# dataiter = iter(data_loader)

#getting the next batch
# batch = next(dataiter)

#unpacking the batch
# images, captions = batch

#showing info of image in single batch
# for i in range(BATCH_SIZE):
#     img,cap = images[i],captions[i]
#     caption_label = [dataset.vocab.itos[token] for token in cap.tolist()]
#     eos_index = caption_label.index('<EOS>')
#     caption_label = caption_label[1:eos_index]
#     caption_label = ' '.join(caption_label)                      
#     show_image(img,caption_label)
#     plt.show()