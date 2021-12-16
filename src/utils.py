import torch
# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#generate caption
def get_caps_from(features_tensors, model, vocab):
    #generate the caption
    model.eval()
    with torch.no_grad():
        # features = model.encoder(features_tensors.to(device))
        features = model.encode(features_tensors.to(device))
        caps,alphas = model.decoder.generate_caption(features,vocab=vocab)
        caption = ' '.join(caps)
        # show_image(features_tensors[0],title=caption)
    return caps, alphas