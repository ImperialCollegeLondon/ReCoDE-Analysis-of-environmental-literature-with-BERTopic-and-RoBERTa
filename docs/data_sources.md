### **Web of Science**

Web of Science is a widely used research platform and citation database that provides access to a vast collection of scholarly literature and scientific information across various disciplines. One of the key features of Web of Science is its citation indexing, which allows users to track and analyze citation metrics, such as citation counts, h-index, and citation networks. In addition to citation data, Web of Science offers **powerful search and discovery tools**, advanced filtering options, and analytical capabilities to facilitate literature review, bibliometric analysis, and knowledge exploration. It provides access to high-quality, peer-reviewed content from reputable publishers and scholarly organizations, making it a valuable resource for academic research, scientific publishing, and decision-making in various fields.

In this demonstration, I will elucidate a step-by-step process for gathering and refining original data for subsequent analysis.

Commencing with a predefined query, accessible via this link: https://www.webofscience.com/wos/woscc/summary/a09550d4-2eb9-4b8c-9f9e-711acfe751a0-e65d872d/relevance/1, I have delineated the search parameters to encompass the topics of "environment" and "nature," incorporating keywords like "sustainability". Furthermore, I have delineated the search category as "Environmental Sciences," specified English as the language, and selected "Article" as the document type. Notably, the search results are filtered to exclusively include "Open Access" documents.

Web of Science permits the export of up to 1000 records per search iteration. As such, I've compiled the top 5000 records into five Excel files, each sharing a uniform structure: "Web_of_Science_Search_1-1000 results.xls," "Web_of_Science_Search_1001-2000 results.xls," and so forth. Now, the objective is to amalgamate these files into a singular CSV file named "Web_of_Science_Query May 07 2024_1-5000.csv," retaining select columns, namely "Publication Type," "Authors," "Article Title," "Source Title," "Abstract," "Publication Year," and "DOI."

**NB**

To use Web of Science and export datasets, you need to register.

If you wish to use a pre-processed dataset in this demonstration, all datasets must be uploaded to the Colab Files in advance; otherwise, they will not be automatically loaded. All files are available on the [GitHub repository](https://github.com/ImperialCollegeLondon/ReCoDE-Analysis-of-environmental-literature-with-BERTopic-and-RoBERTa/tree/dev).
