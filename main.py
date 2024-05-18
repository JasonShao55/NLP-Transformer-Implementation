import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import torch.nn as nn

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

from transformer import TransformerDecoder, create_mask
from utilities import Utilities


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 501 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100, tokenizer=None):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    with torch.no_grad():
        for  i, (X, Y) in enumerate(data_loader):
            X, Y = X.to(device), Y.to(device)
            mask = create_mask(X.size(0)).to(device)  # Create the mask
            outputs,_ = decoderLMmodel(X, mask)
            #print("----output logits---",outputs)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs.view(-1, tokenizer.vocab_size), Y.view(-1))
            #loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
            losses.append(loss.item())
            #total_loss += loss.item()
            if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

  
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    '''Perplexity test data'''
    #inputfile = "speechesdataset/test_LM_hbush.txt"
    inputfile = "speechesdataset/test_LM_obama.txt"
    #inputfile = "speechesdataset/test_LM_wbush.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    test_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    test_LM_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=True)
    #perplexity_data_loader=train_LM_loader #train perplexity 
    perplexity_data_loader=test_LM_loader #other perplexity 

    ''' Model initialization here'''
    # Initialize the TransformerDecoder model
    model = TransformerDecoder(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=block_size,
        embed_dim=n_embd,
        num_heads=n_head,
        ff_hidden_dim=100,  # Hidden dimension for feed-forward network in the transformer decoder
        num_layers=n_layer,
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()



     # for the classification  task, you will train for a fixed number of epochs like this:
    for epoch in range(epochs_CLS):
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            # CLS training code here


    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        # LM training code here

         # Create the mask
        mask = create_mask(xb.size(0)).to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs,_ = model(xb, mask)

        # Calculate the loss
        loss = criterion(outputs.view(-1, tokenizer.vocab_size), yb.view(-1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print iteration every eval_interval
        if i % 100 == 0:
            # print perplexity:
            perplexity = compute_perplexity(model, perplexity_data_loader, eval_iters, tokenizer) # Train perplexity
            print(f"Iteration {i}, Loss: {loss.item()}, Perplexity: {perplexity}") 
    
    print('Training complete.')
    # sanity check
    sentence = "It is costly and politically difficult to continue this conflict."
    utils = Utilities(tokenizer, model)
    utils.sanity_check_decoder(sentence, block_size)



    



if __name__ == "__main__":
    main()
