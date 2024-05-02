<!-- Your Project title, make it sound catchy! -->

# ReCoDE - Analysis of environmental literature with BERTopic and RoBERTa

<!-- Provide a short description to your project -->

## Explosive litearture in Environmental and Sustainability Studies

The field of **environmental and sustainability** studies has witnessed an explosive growth in literature over the past few decades, driven by the increasing global awareness and urgency surrounding environmental issues, climate change, and the need for sustainable practices.

This rapidly expanding body of literature is characterized by its **interdisciplinary nature**, encompassing a wide range of disciplines such as ecology, climate science, energy, economics, policy, sociology, and more. With a global focus and contributions from countries around the world, the literature base reflects **diverse cultural, socio-economic, and geographical contexts**, often in multiple languages. **Novel research areas and emerging topics**, such as circular economy, sustainable urban planning, environmental justice, biodiversity conservation, renewable energy technologies, and ecosystem services, continue to arise as environmental challenges evolve and our understanding deepens. The **development of environmental policies**, regulations, and international agreements, as well as increased public interest and awareness, have further fueled research and the demand for literature aimed at informing and engaging various stakeholders. **Technological advancements** in areas like remote sensing, environmental monitoring, and computational modeling have enabled new avenues of research and data-driven studies, contributing to the proliferation of literature. **The rise of open access publishing and digital platforms** has facilitated the dissemination and accessibility of this constantly evolving and interdisciplinary body of knowledge.

So, in summary, the explosive growth of the literature across multiple disciplines, geographic regions, languages, and emerging topics poses significant challenges in terms of effectively organizing, synthesizing, and extracting insights from this vast and rapidly expanding body of knowledge. This is where **Natural Language Processing (NLP)** techniques like **topic modeling** with BERTopic and advanced language models like RoBERTa can play a crucial role. Their ability to process large volumes of text data, identify semantic topics and patterns, cluster related documents, and handle multiple languages can help researchers, policymakers, and stakeholders navigate this extensive literature more effectively.


Also, as a STEMM PhD student at Imperial, who is going to step into a new field like Sustainability, it is helpful to learn how to take advantage of the NLP tools to accelerate your literature exploration and review process, and achieve a more smooth interdisciplinary research.

**Furthermore, as a STEMM PhD student at Imperial stepping into a new field such as Sustainability, taking advantage of the NLP tools can significantly enhance the efficiency of literature exploration and review. This skill facilitates a seamless transition into interdisciplinary research, empowering you to navigate diverse datasets and extract valuable insights with greater ease and precision.**

## The Potential of Topic Modeling

Topic modeling is a technique in NLP and machine learning used to discover abstract "topics" that occur in a collection of documents. The key idea is that documents are made up of mixtures of topics, and that each topic is a probability distribution over words.

More specifically, topic modeling algorithms like Latent Dirichlet Allocation (LDA) work by:

1. Taking a set of text documents as input.
2. Learning the topics contained in those documents in an unsupervised way. Each topic is represented as a distribution over the words that describe that topic.
3. Assigning each document a mixture of topics with different weights/proportions.

For example, if you ran topic modeling on a set of news articles, it may discover topics like "politics", "sports", "technology", etc. The "politics" topic would be made up of words like "government", "election", "policy" with high probabilities. Each document would then be characterized as a mixture of different proportions of these topics.

The key benefits of topic modeling include:

1. Automatically discovering topics without need for labeled data
2. Understanding the themes/concepts contained in large document collections
3. Organizing, searching, and navigating over a document corpus by topics
4. Providing low-dimensional representations of documents based on their topics

Topic modeling has found applications in areas like **information retrieval, exploratory data analysis, document clustering and classification, recommendation systems**, and more. Popular implementations include Latent Dirichlet Allocation (LDA), Biterm Topic Model (BTM), and techniques leveraging neural embeddings like BERTopic.

## Learning Outcomes

By the end of this tutorial, students will be able to acquire the following learning outcomes:

1. Proficiency in Text Data Preprocessing: Participants will gain hands-on experience in preprocessing environmental literature datasets, including cleaning, tokenisation, and normalisation techniques, essential for preparing data for NLP analysis.

2. Understanding the principle of embedding-matrix-based NLP techniques: Through the application of BERTopic for topic modelling and RoBERTa for sentiment analysis, students will develop a deep understanding of advanced NLP methods and their practical implementation in dissecting environmental and sustainability texts and beyond.

3. Critical Analysis Skills: Participants will learn to critically analyse and interpret the results of NLP analyses, including identifying dominant themes, sentiment shifts, and trends in environmental literature, fostering a nuanced understanding of environmental discourse.

4. Interpretation and Application: Relying on a real-world example, this project demonstrates how to generate visualisations and reports to present the results of the topic modelling and sentiment analysis, facilitating interpretation and discussion.

## Requirements

It would help a lot if you went through the following Graduate School courses before going through this exemplar:
* Introduction to Python
* Data Exploration and Visualisation
* Data Processing with Python Pandas
* Plotting in Python with Matplotlib
* Binary Classification of Patent Text Using Natural Language Processing (another ReCoDE project)

### Academic

* Access to Google Colaboratory
* Basic Math (matrices, averages)
* Programming skills (python, pandas, numpy, tensorflow)
* Machine learning theory (at level of intro to machine learning course)

### System

Windows


<!-- Instructions on how the student should start going through the exemplar.

Structure this section as you see fit but try to be clear, concise and accurate
when writing your instructions.

For example:
Start by watching the introduction video,
then study Jupyter notebooks 1-3 in the `intro` folder
and attempt to complete exercise 1a and 1b.

Once done, start going through through the PDF in the `main` folder.
By the end of it you should be able to solve exercises 2 to 4.

A final exercise can be found in the `final` folder.

Solutions to the above can be found in `solutions`.
-->

## Getting Started

Please visit this Colab page to access the detailed content of this tutorial: [https://colab.research.google.com/drive/1vJzmFTFurlK-NGDw_fhJgxSmcKSZooLn#scrollTo=ETJeJC-rMzsf](https://colab.research.google.com/drive/1vJzmFTFurlK-NGDw_fhJgxSmcKSZooLn?usp=sharing)


<a href=""> <img src="https://colab.research.google.com/drive/1vJzmFTFurlK-NGDw_fhJgxSmcKSZooLn?usp=sharing" alt="Open In Colab"/></a>


<!-- An overview of the files and folder in the exemplar.
Not all files and directories need to be listed, just the important
sections of your project, like the learning material, the code, the tests, etc.

A good starting point is using the command `tree` in a terminal(Unix),
copying its output and then removing the unimportant parts.

You can use ellipsis (...) to suggest that there are more files or folders
in a tree node.

-->

## Project Structure

```log
.
├── examples
│   ├── ex1
│   └── ex2
├── src
|   ├── file1.py
|   ├── file2.cpp
|   ├── ...
│   └── data
├── app
├── docs
├── main
└── test
```

<!-- Change this to your License. Make sure you have added the file on GitHub -->

## License

This project is licensed under the [BSD-3-Clause license](LICENSE.md)
