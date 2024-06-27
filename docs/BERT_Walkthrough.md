# **A Step-by-Step Case Study using BERTopic to Analyze One web of Science Dataset**
In this step-by-step case study, we will focus on the application of BERTopic, to analyze a sample dataset sourced from Web of Science. Through this tutorial, we aim to guide you through the process:

 * Installation and setup of BERTopic
 * Collecting the raw data and preprocessing the dataset
 * Implementing BERTopic for topic modeling
 * Visualizing the inferred topics and interpreting the results
 * Fine-tuning topic representations
 * Additional readings about the wider application of BERTopic

By following along, you will gain practical insights into leveraging BERTopic for insightful analysis of scholarly literature from Web of Science.

## **Data Preparation**
### Load the dataset & Preview the data
```markdown
```python
df = pd.read_csv("Web_of_Science_Query May 07 2024_1-5000.csv", encoding='utf-8')
print(df.head())

## **Data Analysis**
### **Applying BERT and Validation**
The provided code snippet employs the BERTopic library to conduct topic
modeling on the given dataset "Web_of_Science_Query May 07 2024_1-5000.csv". Initially, a BERTopic instance is created, enabling the implementation of BERT-based topic modeling. Subsequently, the model is trained on the dataset by fitting it with the abstracts extracted from the DataFrame 'df'. The 'Abstract' column is accessed and converted into a list of documents ('docs'), which serves as the input data for the topic modeling process.
As a result, the code generates topics and corresponding probabilities for each document, facilitating the extraction of meaningful themes and insights from the dataset.

from bertopic import BERTopic
import pandas as pd

### Load your DataFrame with the abstracts
df = pd.read_csv("Web_of_Science_Query May 07 2024_1-5000.csv", encoding='utf-8')

### Preprocess the data to handle null values
df['Abstract'] = df['Abstract'].fillna('')  # Replace null values with empty strings

### Create a BERTopic instance
topic_model = BERTopic(verbose=True)

### Fit the model on your dataset
docs = df['Abstract'].tolist()
topics, probs = topic_model.fit_transform(docs)
Before we start visualizing the results, a good approach to double-check whether the topics have been successfully inferred by the BERTopic model is listed below. The code is checking whether the topics_ attribute exists and is not None in the topic_model object. If the attribute exists and is not None, it prints the inferred topics; otherwise, it prints a message indicating that the topics_ attribute has not been populated yet.

By doing so, we can prevent errors and ensure that we have valid topic data to work with.
### Check if the topics_ attribute exists and is not None
if hasattr(topic_model, 'topics_') and topic_model.topics_ is not None:
    # Print the inferred topics
    print("Inferred Topics:")
    print(topic_model.topics_)
else:
    print("The topics_ attribute has not been populated yet.")

### **Number of documents**
In many cases, especially in text analysis tasks like topic modeling, each record or row in the dataset corresponds to a single document, especially when the dataset is highly structured. For example, in this DataFrame where each row represents a different research paper, then the 'Abstract' column in each row contains the abstract of that paper - each record in the DataFrame contains unique textual content. In this case, there is a one-to-one mapping between documents and records, where each document corresponds to a single record in the DataFrame.

To determine the number of documents in this demo dataset, you can use the len() function in Python, which returns the length of a list or the number of elements in an object.
num_documents = len(docs)
print("Number of documents:", num_documents)
If the dataset is a plain text file with huge chunks of text, determining the number of documents can be more challenging as there may not be a clear separation between individual documents. However, you can use various techniques to identify and count the number of documents in the text file.

Try to identify a pattern or delimiter that separates individual documents in the text file. This could be a specific string, a sequence of characters, or a blank line. For example, if each document starts with a line like "Document #123", you can use that as the delimiter, and use the split() method with the defined delimiter to split the text into a list of individual documents.

### **Embedding**
In the following code snippet, we are generating and preserving embeddings — a numerical representation of textual data (see in "key concepts" section).

Utilizing the `sentence_transformers` library, we begin by initializing a SentenceTransformer model called 'all-MiniLM-L6-v2'. This model serves as our guide in converting text into dense numerical vectors known as embeddings. With the model ready, we proceed to create a BERTopic model, which leverages the embeddings for topic modeling. We then apply our SentenceTransformer model to encode a list of documents into embeddings.

These embeddings are saved in various formats—a NumPy array and a pickle file—for future analysis or reuse. Additionally, we convert the embeddings into a pandas DataFrame and export them to a CSV file, facilitating easy access and manipulation of the data.

However, **why do we need to save the embeddings?** Because it allows for efficient storage and retrieval of numerical representations of textual data, facilitating reproducibility and consistency in downstream tasks. It also serves as a backup or checkpointing mechanism for data processing pipelines, enhancing the efficiency and robustness of analyses and experiments.
### Import the SentenceTransformer library to access pre-trained models for generating sentence embeddings
from sentence_transformers import SentenceTransformer

### Initialize a SentenceTransformer model with the 'all-MiniLM-L6-v2' variant for generating embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

### Initialize a BERTopic model with the specified SentenceTransformer embedding model and enable verbose mode for logging
topic_model = BERTopic(embedding_model=embedding_model, verbose=True)

### Encode the list of documents into embeddings using the initialized SentenceTransformer model, showing a progress bar during the encoding process
embeddings = embedding_model.encode(docs, show_progress_bar=True)

### Save the embeddings to a NumPy array file (.npy)
import numpy as np
np.save('embeddings.npy', embeddings)  # Save to .npy file

### Save the embeddings to a pickle file for serialization (.pkl)

Serialization refers to the process of converting an object into a format that can be easily stored, transmitted, or reconstructed later. In Python, serialization is commonly used for saving objects to files or transferring them between different systems. The .pkl extension here denotes a pickle file, which is a binary file format used for serializing and deserializing objects. Pickle files can store various Python objects, such as lists, dictionaries, and even custom classes, in a compact and efficient binary format.

import pickle
with open('embeddings.pkl', 'wb') as file:
    pickle.dump(embeddings, file)

### Convert the embeddings into a pandas DataFrame for further analysis and export it to a CSV file without indexing
import pandas as pd
embeddings_df = pd.DataFrame(embeddings)
embeddings_df.to_csv('embeddings.csv', index=False)

### **Visulizations and Interpretation**
It is difficult to (mathematically) define **interpretability** (Molnar 2022). A (non-mathematical) definition of interpretability that I like by Miller (2017) is: Interpretability is the degree to which a human can understand the cause of a decision. Another one by Kim et al. (2016) is: Interpretability is the degree to which a human can consistently predict the model’s result.

Through visualizations and explanations of these outputs, I aim to guide you through a process that you can gain insights into how the model makes decisions and understand the factors influencing its predictions.

### **Intertopic Distance Map**

The `visualize_topics()` method in BERTopic is used to generate visualizations of the inferred topics. It provides a graphical representation that allows you to explore the topics and their associated keywords in an intuitive way.

When you call `topic_model.visualize_topics()`, it generates a visualization that typically includes:

1. **Topic Clusters**: Topics are often represented as clusters, with each cluster containing multiple related topics. These clusters can help you identify overarching themes or categories within the dataset.

2. **Keyword Distribution**: For each topic cluster, the visualization typically displays the keywords that are most strongly associated with each topic. These keywords give you insights into the main concepts and ideas represented by each topic.

3. **Topic Distribution**: The visualization may also include information about the distribution of documents across topics. This can help you understand how prevalent each topic is in the dataset and how topics relate to one another.
topic_model.visualize_topics()
#### **Topic Word Scores**

The `visualize_barchart()` method in BERTopic generates a bar chart visualization of the most prominent topics based on their prevalence in the document corpus.

When you call `topic_model.visualize_barchart(top_n_topics=15)`, it generates a bar chart that typically includes:

1. **Topic Distribution**: The bar chart displays the distribution of documents across the top N topics, where N is specified by the `top_n_topics` parameter. Each bar represents a topic, and the height of the bar indicates the proportion of documents assigned to that topic.

2. **Topic Labels**: The topics are usually labeled along the x-axis of the bar chart, allowing you to identify each topic.

3. **Document Counts**: The y-axis of the bar chart typically represents the number of documents assigned to each topic, providing insight into the prevalence of each topic in the document corpus.

**Deciding the value of `n` for topic modeling** involves considering domain knowledge, conducting exploratory analysis, and evaluating model performance. It is essential to balance granularity and interpretability, aiming for a value that produces meaningful topics. Experimenting with different `n` values and assessing the coherence and relevance of the generated topics can help in making an informed decision.
topic_model.visualize_barchart(top_n_topics=15)

### **Hierarchical Clustering**

The visualize_hierarchy method in topic_model is used to create a hierarchical visualization of topics, with the top 100 topics being displayed. This visualization helps in understanding the relationships and hierarchical structure between topics, providing insights into how topics are grouped and nested within the document corpus.
topic_model.visualize_hierarchy(top_n_topics=100)
### **Similarity Matrix**

The `visualize_heatmap` method in `topic_model` generates a heatmap visualization of the topic-document matrix, highlighting the distribution of topics across documents. With `top_n_topics` set to `100`, the heatmap displays the top 100 topics and their prevalence within the document corpus. This visualization aids in understanding the relative importance and coverage of different topics, offering insights into the thematic composition of the dataset and identifying potential patterns or trends.
topic_model.visualize_heatmap(top_n_topics=100)


## **Fine-tune Topic Representations**

To fine-tune topic representations, you can explore several strategies:

1. **Adjusting Model Hyperparameters**: Experiment with different hyperparameters of the topic modeling algorithm, such as the number of topics (`n_topics`), the vectorization method, or the dimensionality reduction technique. Tuning these parameters can affect the quality and granularity of the inferred topics.

2. **Optimizing Text Preprocessing**: Refine the preprocessing steps applied to the text data before topic modeling. This may involve techniques such as tokenization, stemming, lemmatization, or removing stop words. Fine-tuning preprocessing can enhance the quality of topic representations by reducing noise and improving semantic coherence.

3. **Incorporating Domain Knowledge**: Incorporate domain-specific knowledge or constraints into the topic modeling process. This can be achieved by providing seed words or phrases related to specific topics of interest or by constraining the model to generate topics within predefined thematic boundaries.

4. **Ensemble Modeling**: Explore ensemble modeling techniques, where multiple topic modeling algorithms or variations of the same algorithm are combined to improve topic representations. Ensemble methods can mitigate the limitations of individual models and enhance the robustness of topic inference.

5. **Evaluation and Iteration**: Continuously evaluate the quality of topic representations using domain-specific metrics or qualitative assessments. Iterate on the fine-tuning process based on feedback and insights gained from analyzing the topics generated by the model.
## **Additional Outcomes to Explore**

In addition to the outcomes we introduced above, you could also consider exploring the following outcomes if time and resources allow:

* **Topic Evolution Over Time:** Analyze how topics evolve over time. This
can be particularly useful if your dataset spans several years. You can also identify trends and shifts in research focus.

* **Institutional or Geographic Distribution:** Examine how topics vary across different institutions or geographic regions.

* **Sentiment Analysis:** Apply sentiment analysis within each topic to understand the sentiment trends related to different research themes.

* **Topic Diversity and Distribution:** Calculate metrics such as topic diversity or entropy to understand how spread out the topics are across the papers.

* **Topic Coherence and Perplexity:** Evaluate the coherence and perplexity of the topics to quantitatively measure their quality.

## **Wider applications of BERT**

### **Example 1 - Embedding backend for the BERTopic library**

The following command installs BERTopic along with additional dependencies for different embedding backends. Here's a brief explanation of each backend:

* Flair: Flair is a natural language processing library that provides contextual string embeddings. It offers state-of-the-art pre-trained models for various NLP tasks.

* Gensim: Gensim is a Python library for topic modeling, document similarity analysis, and other natural language processing tasks. It includes algorithms for word embedding models like Word2Vec and Doc2Vec.

* Spacy: Spacy is a powerful and efficient library for natural language processing in Python. It provides pre-trained word vectors and other NLP functionalities.

* USE (Universal Sentence Encoder): USE is a deep learning model developed by Google for generating universal sentence embeddings. It encodes sentences into fixed-length dense vectors that capture semantic information.

By installing BERTopic with these additional dependencies, you gain access to multiple embedding options, allowing you to choose the one that best suits your needs for topic modeling and related tasks.
pip install bertopic[flair, gensim, spacy, use]

### **Example 2 - BEiT: Topic modelling with images**

BEiT (BERT Pre-Training of Image Transformers) is a novel approach to pre-training image transformers, inspired by the success of BERT (Bidirectional Encoder Representations from Transformers) in natural language processing tasks. Unlike traditional convolutional neural networks (CNNs) commonly used for image processing, BEiT leverages transformer architectures to learn representations of images. It represents a promising direction in the field of computer vision, offering a new perspective on image representation learning and paving the way for advancements in various image-related tasks.

If you are interested in reading more about this, please visit this link: https://arxiv.org/pdf/2106.08254
