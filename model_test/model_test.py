import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import numpy as np
import pickle
import pandas as pd
import io
import yaml
import os
import argparse

# from .squence import pad_sequences


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.
    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.
    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the beginning or the end
    if padding='post.
    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.
    Pre-padding is the default.
    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.
    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`
    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

# % matplotlib inline

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def main(args):
    with open(args.config) as f:
        config = yaml.load(f)
        for key, value in config.items():
            setattr(args, key, value)    

    args.output_dir = os.path.join('results', args.config.split('/')[-1].split('.')[0].upper())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)

    print(f'Torch Version: {torch.__version__}')
    print('-----------------------------------------------')
    print(f'Cuda Is Available: {torch.cuda.is_available()}')
    print('-----------------------------------------------')
    print(f'Cuda number: {torch.cuda.device_count()}')
    print('-----------------------------------------------')
    # print(f'GPU Version: {torch.cuda.get_device_name()}')
    print('-----------------------------------------------')

    df = pd.read_csv(args.train_dir, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    # Create sentence and label lists
    sentences = df.sentence.values
    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    labels = df.label.values
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    print ("Tokenize the first sentence:")
    print (tokenized_texts[0])
    # Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway. 
    # In the original paper, the authors used a length of 512.
    MAX_LEN = 128
    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    # Use train_test_split to split our data into train and validation sets for training
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                                random_state=2018, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                random_state=2018, test_size=0.1)

    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
    batch_size = 32

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': args.weight_decay_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]

    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = BertAdam(optimizer_grouped_parameters,
                        lr=args.lr,
                        warmup=.1)


    # Function to calculate the accuracy of our predictions vs labels
    # Store our loss and accuracy for plotting
    train_loss_set = []
    test_acc_set = []
    # Number of training epochs (authors recommend between 2 and 4)
    epochs = args.epochs
    
    print("Training begin!!!")
    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):
        # Training
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
    
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            train_loss_set.append(loss.item())
            
            eval_accuracy=0
            nb_test_steps = 0
            # Evaluate data for one epoch
            for step, batch in enumerate(validation_dataloader):
            # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():
                    # loss_val = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                    logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                nb_test_steps += 1    
                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                te_acc = flat_accuracy(logits, label_ids)            
                eval_accuracy += te_acc
            test_acc_set.append(eval_accuracy/nb_test_steps)

            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # print("Train loss: {}".format(tr_loss/nb_tr_steps))

    # with open(os.path.join(args.output_dir,args.saver_train),'wb') as f:
    #     pickle.dump(train_loss_set,f)
    with open(os.path.join(args.output_dir,args.saver_test),'wb') as f:
        pickle.dump(test_acc_set,f)

    print("End of procedure!!!!!")  
  # # Validation
  # # Put model in evaluation mode to evaluate loss on the validation set
  # model.eval()

  # # Tracking v`ariables 
  # eval_loss, eval_accuracy = 0, 0
#   nb_eval_steps, nb_eval_examples = 0, 0

  # # Evaluate data for one epoch
#   for batch in validation_dataloader:
    # Add batch to GPU
    # batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    # b_input_ids, b_input_mask, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up validation
    # with torch.no_grad():
    #   # Forward pass, calculate logit predictions
    #   logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
    # Move logits and labels to CPU
    # logits = logits.detach().cpu().numpy()
    # label_ids = b_labels.to('cpu').numpy()

    # tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    
    # eval_accuracy += tmp_eval_accuracy
    # nb_eval_steps += 1
#   print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
                                            

# plt.figure(figsize=(15,8))
# plt.title("Training loss")
# plt.xlabel("Batch")
# plt.ylabel("Loss")
# plt.plot(train_loss_set)
# plt.plot(test_loss_set)
# plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'End-to-End Human Object Interaction Detection with HOI Transformer')
    parser.add_argument('--config', default='')
    args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
