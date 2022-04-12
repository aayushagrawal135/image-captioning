# %%

# imports
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate import bleu_score
from wordmap import WordMap
from dataset import FlickrDataset
from encoder import Encoder
from decoder import Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_metrics():
    pass


def train(args: argparse,
          train_loader: DataLoader,
          encoder: Encoder,
          decoder: Decoder,
          encoder_optimizer: torch.optim,
          decoder_optimizer: torch.optim,
          criterion: nn.modules.loss,
          epoch: int):
    encoder.train()
    decoder.train()

    for idx, (imgs, captions, caption_lens) in enumerate(train_loader):
        imgs = imgs.to(device)  # (batch_size, 3, 224, 224)
        captions = captions.to(device)  # (batch_size, x)
        caption_lens = caption_lens.to(device)

        image_features = encoder(imgs)  # (batch_size, 14, 14, 512)
        predictions, encoded_captions, decode_lengths, alphas, _ = decoder(image_features, captions, caption_lens)

        # all samples (rows), skip the 0th column because it corresponds to <start>
        targets = encoded_captions[:, 1:]

        predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Extract tensors from packed sequence
        predictions = predictions.data
        targets = targets.data

        # Get loss
        loss = criterion(predictions, targets)

        # Back propagation
        decoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients

        # Update weights
        decoder_optimizer.step()

        log_metrics()

        if args.dry_run:
            break


def validate(val_loader: DataLoader,
             encoder: Encoder,
             decoder: Decoder,
             criterion: nn.modules.loss,
             word_map: WordMap):
    decoder.eval()
    encoder.eval()

    with torch.no_grad():
        for i, (images, captions, caption_lens, all_captions) in enumerate(val_loader):
            # Move to device
            images = images.to(device)
            captions = captions.to(device)
            caption_lens = caption_lens.to(device)

            image_features = encoder(images)
            predictions, encoded_captions, decode_lengths, alphas, indices = decoder(image_features, captions,
                                                                                     caption_lens)

            targets = encoded_captions[:, 1:]

            # Cloned so that we can use the original prediction for finding BLEU scores later
            predictions_copy = predictions.clone()
            predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            predictions = predictions.data
            targets = targets.data

            loss = criterion(predictions, targets)

            log_metrics()

            break

        bleu4 = get_bleu4_scores(encoded_captions, indices, word_map, predictions_copy, caption_lens)
    return bleu4


def get_bleu4_scores(encoded_captions, indices, word_map: WordMap, predictions, caption_lens):
    reference = list()
    encoded_captions = encoded_captions[indices]
    for caption in encoded_captions:
        caption_tokens = caption.tolist()
        reference_caption = list()
        for token in caption_tokens:
            if token not in {word_map['<start>'], word_map['<pad>']}:
                reference_caption.append(token)
        reference.append(reference_caption)

    # Pick the word (index) with the maximum probability for each (batch, location in sentence)

    hypotheses = list()
    _, predicted_token_indices = torch.max(predictions, dim=2)  # values, indices (batch_size, number of tokens)
    predicted_token_indices = predicted_token_indices.tolist()
    redacted_predicted_token_indices = list()
    for batch_num, p in enumerate(predicted_token_indices):
        redacted_predicted_token_indices.append(
            predicted_token_indices[batch_num][:caption_lens[batch_num]])  # remove pads

    hypotheses.extend(redacted_predicted_token_indices)
    return 1


def save_checkpoint():
    pass


def setup_encoder_decoder(args: argparse, wordmap: WordMap):
    # Read word map
    # Set up checkpoints - if ... else ...
    vocab_size = len(wordmap)

    decoder = Decoder(embed_dim=args.embed_dim,
                      vocab_size=vocab_size,
                      attention_dim=args.attention_dim,
                      encoder_dim=args.encoder_dim,
                      decoder_dim=args.decoder_dim)
    decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                   lr=args.decoder_lr)
    # fine tune
    encoder = Encoder()
    # encoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, encoder.parameters()),
    #                                lr = args.encoder_lr)
    encoder_optimizer = None

    # move encoder, decoder to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    return encoder, decoder, encoder_optimizer, decoder_optimizer, criterion


def setup_dataloaders(args: argparse):
    transforms = T.Compose([
        T.Resize(226),
        T.RandomCrop(224),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_dataset = FlickrDataset(args.data_folder, args.data_name, "TRAIN", transforms)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=False)

    val_dataset = FlickrDataset(args.data_folder, args.data_name, "VAL", transforms)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)

    return train_loader, val_loader


def setup(args: argparse):
    word_map = WordMap(args)
    encoder, decoder, encoder_optimizer, decoder_optimizer, criterion = setup_encoder_decoder(args, word_map)
    train_loader, val_loader = setup_dataloaders(args)

    # epochs in range
    for epoch in range(args.epochs):
        # terminate if no improvement

        train(args, train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, epoch)

        bleu_score = validate(val_loader, encoder, decoder, criterion, word_map)

        if args.dry_run:
            break

        # update best bleu score
        save_checkpoint()


def main():
    parser = argparse.ArgumentParser(description="Image captioning with Attention")

    parser.add_argument("--batch-size", type=int, default=32, help="input batch size for images")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs to train")
    parser.add_argument("--encoder", type=str, default="resnet50", help="encoder model for captioning")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--encoder-dim", type=int, default=512)
    parser.add_argument("--encoder-lr", type=float, default=3e-4)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--embed-dim", type=int, default=300)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-folder", type=str, default="../data/out")
    parser.add_argument("--data-name", type=str, default="flickr8k_5_cap_per_img_5_min_word_freq")
    parser.add_argument("--dry-run", type=bool, default=True)
    # TODO: Add more parser args

    args = parser.parse_args()
    setup(args)


if __name__ == '__main__':
    main()
