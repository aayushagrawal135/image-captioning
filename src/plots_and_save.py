import matplotlib.pyplot as plt

def show_image(img, title=None, epoch = None):
    """Imshow for Tensor."""
    #unnormalize 
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224 
    img[2] = img[2] * 0.225 
    img[0] += 0.485 
    img[1] += 0.456 
    img[2] += 0.406
    
    img = img.numpy().transpose((1, 2, 0))
    
    plt.imshow(img)
    if title is not None:
        if epoch is not None:
            plt.title(f"Epoch: {epoch} : Caption: {title}")
        else:
            plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def save_image(img, saved_count, model_name, title=None, epoch = None):
    """Imshow for Tensor."""
    #unnormalize 
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224 
    img[2] = img[2] * 0.225 
    img[0] += 0.485 
    img[1] += 0.456 
    img[2] += 0.406
    
    img = img.numpy().transpose((1, 2, 0))
    if title is not None:
        if epoch is not None:
            plt.title(f"{title}")
        else:
            plt.title(title)
    plt.imshow(img)
    if epoch is not None:
        plt.savefig(f"./data/{model_name}/results/{epoch}_{saved_count}.png")  # pause a bit so that plots are updated
    else:
        plt.savefig(f"./data/{model_name}/results/{saved_count}.png")  # pause a bit so that plots are updated
    plt.pause(0.01)
    
    print(f"saved_count: {saved_count}")

#Show attention
def plot_attention(img, result, attention_plot, epoch, saved_count, model_name):
    #untransform
    try:
        img[0] = img[0] * 0.229
        img[1] = img[1] * 0.224 
        img[2] = img[2] * 0.225 
        img[0] += 0.485 
        img[1] += 0.456 
        img[2] += 0.406
        
        img = img.numpy().transpose((1, 2, 0))
        temp_image = img
        fig = plt.figure(figsize=(15, 15))
        len_result = len(result)
        for l in range(len_result):
            temp_att = attention_plot[l].reshape(7,7)
            ax = fig.add_subplot(len_result//2,len_result//2, l+1)
            ax.set_title(result[l])
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=img.get_extent())
        plt.tight_layout()
        plt.savefig(f"./data/{model_name}/attention/{epoch}_{saved_count}.png")
        plt.pause(0.01)
        plt.show()
    except:
        pass     

def plot_loss(loss_values, model_name = None):
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(loss_values)
    plt.show()
    model_name = "" if None else model_name
    plt.savefig(f"./data/loss/loss_graph_{model_name}.png")
