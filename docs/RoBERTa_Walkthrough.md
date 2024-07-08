# **A Step-by-Step Case Study using RoBERTa**

Similart to what we have done above, we need to follow the following steps when applying a RoBERTa model.

* RoBERTa Initialization: Initializes RoBERTa tokenizer and model
* Data Preparation: Loads and preprocesses the dataset
* Batch Tokenization: Tokenizes abstracts in batches
* Embedding Generation: Generates embeddings using RoBERTa, and save it
* Topic Modeling: Applies BERTopic with RoBERTa embeddings
* Improve and fine-tune
* Visualization

This section focuses on integrating RoBERTa into the topic modeling pipeline, enhancing its analytical capabilities.

## **Dataset**

We will be using the same dataset, "Web_of_Science_Query May 07 2024_1-5000.csv".

```markdown
import pandas as pd

# Load dataset
df = pd.read_csv('Web_of_Science_Query May 07 2024_1-5000.csv', encoding='utf-8')
abstracts = df['Abstract'].dropna().tolist()  # Ensure no NaN values

# Ensure all elements are strings
abstracts = [str(abstract) for abstract in abstracts]

# Debug: Print the first few elements to check
print(abstracts[:5])
```



## **Tokenize the Data**

Convert the abstracts into tokens that the RoBERTa model can process.

```markdown
# Function to tokenize in batches
def batch_tokenize(texts, batch_size=32):
    all_inputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        all_inputs.append(inputs)
    return all_inputs

# Tokenize abstracts in batches
batched_inputs = batch_tokenize(abstracts)
```

## **Embedding Generation**
This following part is responsible for generating embeddings for each batch of tokenized inputs. More specifically:

   - `inputs`: This parameter represents a list of tokenized inputs. Each element in the list corresponds to a batch of tokenized input data.
   - `embeddings = []`: This initializes an empty list to store the embeddings generated for each batch.
   - Batch Processing: The function iterates through each batch of tokenized inputs provided in the `inputs` list. Within each iteration, a `with torch.no_grad():` block ensures that no gradients are calculated during the forward pass, reducing memory consumption and speeding up computations.
   - `outputs = model(**input)`: This line feeds the current batch of tokenized inputs (`input`) to the RoBERTa model (`model`) to obtain the model outputs.
   - `outputs.last_hidden_state`: The `outputs` object contains various attributes, including the last hidden states of all tokens in the input sequence. Here, `last_hidden_state` retrieves these hidden states.
   - `batch_embeddings = outputs.last_hidden_state.mean(dim=1)`: This computes the mean of the last hidden states along the sequence dimension (dimension 1), resulting in a single vector representation (embedding) for each input sequence in the batch.
   - `torch.cat(embeddings)`: Finally, all the embeddings generated for different batches are concatenated along the batch dimension (dimension 0) using PyTorch's `torch.cat()` function, resulting in a tensor containing embeddings for all input sequences.

Please note that executing this step may take **a substantial amount of time** due to its computational complexity.

```markdown
import torch

# Function to generate embeddings for each batch
def batch_embed(inputs):
    embeddings = []
    for input in inputs:
        with torch.no_grad():
            outputs = model(**input)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings)
    return torch.cat(embeddings)

# Generate embeddings
embeddings = batch_embed(batched_inputs)
import csv

# Define the file path to save the embeddings
output_file = "embeddings_roberta.csv"

# Convert embeddings tensor to a numpy array
embeddings_array = embeddings.numpy()

# Write the embeddings to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for embedding_row in embeddings_array:
        writer.writerow(embedding_row
```

## **Topic Modeling**
```markdown
import pandas as pd
import numpy as np

# Load the embeddings from the CSV file
df = pd.read_csv("embeddings_roberta.csv", header=None)
embeddings = df.values

# Create a BERTopic instance without specifying an embedding model
topic_model = BERTopic()

# Fit the topic model and get topics and probabilities
topics, probabilities = topic_model.fit_transform(abstracts, embeddings)
```

## **Visualizing, Analyzing and Comparing Results**
Similar to what we have produced above, we are first looking at the Intertopic Distance Map.
```markdown
topic_model.visualize_topics()
topic_model.visualize_hierarchy()
topic_model.get_topic_info()
topic_model.get_topics()
```

## **Improve and Fine-Tune**
Clearly, we can see the performance of this model is not ideal, as there are many stop words that influence the quality of our output. Stop words such as "we," "the," "of," and "and" are common and do not carry significant meaning, which can dilute the meaningful patterns in our data and negatively impact the performance of our NLP model. To improve the performance, we can pre-process the textual dataset as follows:  

```markdown
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# Load the dataset again
df = pd.read_csv('Web_of_Science_Query May 07 2024_1-5000.csv')
abstracts = df['Abstract'].dropna().tolist()

# Define a pre-processing function
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]  # Remove stop words
    return ' '.join(words)

# Preprocess the abstracts
abstracts = [preprocess(abstract) for abstract in abstracts]
```

Then we repeat the analysis again:

```markdown
from transformers import RobertaTokenizer, RobertaModel
import torch

# Again, load RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

# Function to tokenize text in batches
def batch_tokenize(texts, batch_size=32):
    all_inputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        all_inputs.append(inputs)
    return all_inputs

# Function to generate embeddings for each batch
def batch_embed(inputs):
    embeddings = []
    for input in inputs:
        with torch.no_grad():
            outputs = model(**input)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings)
    return torch.cat(embeddings)

# Generate embeddings
batched_inputs = batch_tokenize(abstracts)
embeddings = batch_embed(batched_inputs)

# Save the updated embeddings
output_file = "embeddings_roberta_updated.csv"

# Convert this embeddings tensor to a numpy array
embeddings_array = embeddings.numpy()

# Write the new embeddings to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for embedding_row in embeddings_array:
        writer.writerow(embedding_row)
df = pd.read_csv("embeddings_roberta_updated.csv", header=None)
embeddings = df.values

# Create a BERTopic instance without specifying an embedding model
topic_model = BERTopic()

# Fit the topic model and get topics and probabilities
topics, probabilities = topic_model.fit_transform(abstracts, embeddings)
topic_model.visualize_topics() # Visualize the topics
topic_info = topic_model.get_topic_info()
print("Optimized Topic Information:")
print(topic_info.head(10))  # Print the top 10 topics
topic_model.visualize_barchart(top_n_topics=15)
topic_model.visualize_hierarchy()
topic_model.visualize_heatmap()
```
