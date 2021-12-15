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


def save_image(img, saved_count, title=None, epoch = None):
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
            plt.title(f"Epoch: {epoch} : Caption: {title}")
        else:
            plt.title(title)
    plt.imshow(img)
    if epoch is not None:
        plt.savefig(f"../data/{epoch}_{saved_count}.png")  # pause a bit so that plots are updated
    else:
        plt.savefig(f"../data/{saved_count}.png")  # pause a bit so that plots are updated
    plt.pause(0.001)
    
    print(f"saved_count: {saved_count}")

#Show attention
def plot_attention(img, result, attention_plot):
    #untransform
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
    plt.show()