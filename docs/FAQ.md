**1. Specify a preferred hardware accelerator on Colab**

CPU (Central Processing Unit), GPU (Graphics Processing Unit), and TPU (Tensor Processing Unit) are different types of processors designed for different computing tasks.

CPUs are general-purpose processors suitable for a wide range of computing tasks, GPUs are specialized for parallel processing and are widely used for accelerating machine learning and scientific computations, while TPUs are custom-designed by Google specifically for machine learning workloads and are primarily used in data centers and cloud environments.

In this tutorial, I have set CPU as a default.


**What types of GPU/TPUs are available in Colab?**

The types of GPUs and TPUs that are available in Colab vary over time. This is necessary for Colab to be able to provide access to these resources free of charge.

We have two TPU types: "TPU (deprecated)" and "TPU v2". "TPU (deprecated)" is backed by a TPU Node architecture system. On TPU Node architecture systems, the TPU is hosted on a remote machine. All TPU operations are sent over the network. This can introduce performance problems and debugging difficulties. "TPU v2" has a TPU VM architecture, where the TPU is attached to the local VM. The "TPU v2" backend has a 4-chip v2 TPU.

You can access premium GPUs subject to availability by purchasing one of our paid plans here.

If you would like access to specific dedicated hardware, explore using GCP Marketplace Colab.


**To specify the preferred hardware accelerator on Google Colab, follow these steps:**

1. Open Google Colab: Go to Google Colab.

2. Create a new notebook or open an existing one.

3. Go to the "Runtime" menu at the top of the Colab interface.

4. Select "Change runtime type" from the dropdown menu. In the dialog that appears, you can choose the hardware accelerator you prefer:
   * For a GPU, select "GPU" from the "Hardware accelerator" dropdown.
   * For a TPU, select "TPU" from the "Hardware accelerator" dropdown.
Click "Save" to apply the changes.

****


**2. Besides Web of Science, where else can I find datasets, and how can I import literature datasets in bulk across platforms?**

There are many different sources. Here are some examples:

1. **ArXiv Dataset - ArXiv dataset and metadata of 1.7M+ scholarly papers across STEM**: For nearly 30 years, ArXiv has served the public and research communities by providing open access to scholarly articles, from the vast branches of physics to the many subdisciplines of computer science to everything in between, including math, statistics, electrical engineering, quantitative biology, and economics. This rich corpus of information offers significant, but sometimes overwhelming depth. It is a collaboratively funded, community-supported resource founded by Paul Ginsparg in 1991 and maintained and operated by Cornell University. In these times of unique global challenges, efficient extraction of insights from data is essential.

2. **Kaggle**: To help make the ArXiv more accessible, a free, open pipeline has been presented on Kaggle to the machine-readable ArXiv dataset: a repository of 1.7 million articles, with relevant features such as article titles, authors, categories, abstracts, full text PDFs, and more. The hope is to empower new use cases that can lead to the exploration of richer machine learning techniques that combine multi-modal features towards applications like trend analysis, paper recommender engines, category prediction, co-citation networks, knowledge graph construction and semantic search interfaces. Therefore, it can be a good data source. There are other datasets on Kaggle too.

3. **Semantic Scholar**: The ArXiv dataset is rather large (1.1TB and growing). It is an AI-powered academic search engine that helps researchers discover and access scientific literature more efficiently. It offers advanced search capabilities, personalized recommendations, and tools for analyzing research papers. To retrieve papers with specific search criteria like "Environmental Science", date range from "2004-2024", "Has PDF", and from Journals & Conference as "arXiv.org" without specific keywords, and you can use API to retrieve datasets.






**3. Can I publish the textual dataset I pre-processed, and where？**

Publishing your curated text dataset for future NLP model training is a great way to contribute to the research community and support the development of new models. There are several platforms where you can publish and share your dataset, for instance:

1. **Kaggle**: Kaggle is a well-known platform for data science competitions and datasets. It has a large community of data scientists and researchers. **Link**: [Kaggle Datasets](https://www.kaggle.com/datasets)

2. **Zenodo**: Zenodo is an open-access repository developed by CERN. It allows researchers to share datasets, papers, and other research outputs. Zenodo also assigns a DOI (Digital Object Identifier) to your dataset for easy citation. **Link**: [Zenodo](https://zenodo.org/)

3. **GitHub**: GitHub is a popular platform for hosting code repositories. It can also be used to share datasets, especially if they are relatively small or if you want to include code for data processing. When creating a new repository and upload your dataset files. Remember to add a README file with details about the dataset. **Link**: [GitHub](https://github.com/)





**4. Why do we need to explore literature and compared to manual exploration, what are the advantages of applying models like BERT?**

Exploring literature allows researchers to uncover new insights and understand trends within a field. By systematically reviewing what has been written, you can identify gaps in knowledge, emerging themes, and potential areas for further research. This process is crucial for advancing any scientific discipline. The rapid pace of academic publishing means that staying current with the latest developments is a daunting task. Literature exploration helps keep researchers up-to-date with the newest findings and theories, ensuring that their work remains relevant and informed by the latest evidence.

Using models like BERT offers significant advantages, such as efficiency and speed, where BERT acts like a highly skilled assistant, automating the processing of large text volumes and allowing researchers to focus on interpretation rather than data handling. Its depth and precision enable a nuanced understanding of language, accurately identifying and categorizing topics beyond simple keyword methods. BERT ensures consistency and reduces biases inherent in human analysis, providing a more objective and reliable outcome. It also scales efficiently to handle extensive datasets, offering timely insights regardless of size. Additionally, BERT can uncover hidden patterns and trends in literature, aiding strategic planning, and supports multilingual analysis, making it invaluable in global research contexts by breaking down language barriers for a comprehensive review.

****

**5. Factors helping you decide whether to apply RoBERTa (if you have used a general BERToic model):**

Quality of Topics: Evaluate the coherence and relevance of the topics generated by BERTopic. If the topics are not very coherent or do not align well with your domain knowledge, applying RoBERTa might improve the results as it can capture more nuanced language features.

Comparison and Validation: If you want to validate and compare the robustness of the topics, applying RoBERTa can provide a benchmark. Comparing the results can highlight strengths and weaknesses of both models.

Computational Resources: Applying RoBERTa is computationally intensive. Ensure you have the necessary resources to run and compare the models.

Specific Research Questions: If your research questions require highly refined topics or involve understanding subtleties in language that BERTopic might miss, then using RoBERTa could be beneficial.






**6. Why is preprocessing necessary when using RoBERTa but not always required when using BERTopic？**

BERTopic is a topic modeling technique that often combines pre-trained embedding models (like BERT, RoBERTa, etc.) with clustering algorithms. BERTopic can take raw text inputs directly and generate topics. The default embedding models used by BERTopic, such as those from sentence-transformers, are generally robust to common noise (like stop words) and can still produce meaningful topics even without extensive preprocessing. This built-in robustness means that BERTopic can sometimes handle raw text better without extensive cleaning.

RoBERTa, on the other hand, is a transformer-based language model that processes text at a more granular level. When using RoBERTa directly for tasks like generating embeddings, the presence of noise, such as stop words, punctuation, and irrelevant characters, can impact the quality of the embeddings. RoBERTa does not inherently ignore or downplay stop words and other noise. Therefore, preprocessing steps such as removing stop words, converting text to lowercase, and eliminating punctuation can help improve the quality of the embeddings produced by RoBERTa, leading to better downstream analysis. So, preprocessing text before using RoBERTa can significantly improve the quality of your topic modeling results by reducing the impact of irrelevant words and noise.






**7. Why is it necessary to pretrain a RoBERTa model?**

Pretraining a RoBERTa model is necessary because it allows the model to learn rich, general-purpose language representations from large amounts of text data. During pretraining, RoBERTa is exposed to vast corpora of text, where it learns to understand the nuances of language, capture contextual information, and encode semantic meaning into its embeddings. This pretraining process enables RoBERTa to capture a wide range of linguistic patterns and structures, making it a powerful tool for various natural language processing tasks.






**8. What happens next after pretraining?**

After pretraining, the next step is fine-tuning the RoBERTa model on your specific downstream task. Fine-tuning involves taking the pretrained RoBERTa model and adapting it to perform well on your specific dataset and task. During fine-tuning, you feed your task-specific data (e.g., labeled text for sentiment analysis, question-answer pairs for question answering) into the pretrained RoBERTa model and update its parameters through backpropagation. This process allows RoBERTa to adjust its learned representations to better suit the intricacies of your particular task, ultimately leading to improved performance and better results. Fine-tuning typically involves training the model on a smaller dataset specific to your task, which helps it learn task-specific patterns and nuances. Once fine-tuning is complete, you can use the fine-tuned RoBERTa model for inference and predictions on new, unseen data in your specific domain or application.

