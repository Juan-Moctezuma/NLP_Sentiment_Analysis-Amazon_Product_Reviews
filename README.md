# NLP Sentiment Analysis - Amazon Product Reviews

### Technical Description in Non-Technical Terms
This project’s goal is to analyze comments provided by customers by determining if the given review is associated with the following sentiment: ‘positive’, ‘negative’ or ‘neutral’; and to apply Ai-powered methods for sentiment prediction based on actual rated reviews provided by customers. In this data science project, we are exclusively analyzing reviews of every whey protein product available (as of February 2023) for sell at Amazon. Whey Protein powder is a sporting supplement (based on dairy milk) consumed commonly by athletes, body-builders and anyone who wants or needs additional protein on their diet. How do we obtain our data? By scraping (automatically extracting) pages one through ten from the comments’ section for every single product available when searching for the terms ‘Whey Protein’ at Amazon’s (E-Commerce website) searchbar. What is the main tool for this project? We used Python 3 – programming language for both web scraping and applying both multiple Natural Language Processing (NLP) techniques and statistical methods / data science prediction models.

### Presentation GIF
<img src="GIF/ppt.gif" width="80%">

<b>Click the following link to see PowerPoint Presentation (PDF format):</b> https://github.com/Juan-Moctezuma/NLP_Sentiment_Analysis-Amazon_Product_Reviews/blob/main/Amazon_Product_Review_Sentiment_Analysis.pdf

### Steps
Multiple steps or processes were required for the completion of this project. The following bullet points lists the name of process and the Jupyter Notebook containing the Python code with splitted steps and comments:
1. <b>Web Scraper</b> - Data extraction phase - https://github.com/Juan-Moctezuma/NLP_Sentiment_Analysis-Amazon_Product_Reviews/blob/main/Part1-Whey_Protein_Amazon_Reviews_Scraper.ipynb
2. <b>Data Preprocessing</b> - Data manipulation phase - https://github.com/Juan-Moctezuma/NLP_Sentiment_Analysis-Amazon_Product_Reviews/blob/main/Part2-Amazon_Reviews_Data_Preprocessing.ipynb
3. <b>Exploratory Data Analysis</b> - Process of making graphs and charts to visualize data and verify if our data is appropriate - https://github.com/Juan-Moctezuma/NLP_Sentiment_Analysis-Amazon_Product_Reviews/blob/main/Part3-Amazon_Reviews_Data_Exploration.ipynb
4. Application of Machine Learning techniques and statistical methods for sentiment analysis and prediction:
    * <b>Vader model vs. Roberta model</b>  - https://github.com/Juan-Moctezuma/NLP_Sentiment_Analysis-Amazon_Product_Reviews/blob/main/Part4-Whey_Protein_Sentiment_Analysis_Pre-trained_Models-Vader_vs_Roberta.ipynb
    * <b>Bert neural network / linear regression model</b> - https://github.com/Juan-Moctezuma/NLP_Sentiment_Analysis-Amazon_Product_Reviews/blob/main/Part5-Whey_Protein_Sentiment_Analysis_Pre-trained_BERT_Neural_Network.ipynb
    * <b>TextBlob</b> - Python Library - https://github.com/Juan-Moctezuma/NLP_Sentiment_Analysis-Amazon_Product_Reviews/blob/main/Part6-Whey_Protein_Sentiment_Analysis_Pre-trained_TextBlob.ipynb
    * <b>Naive Bayes Classifier</b> - based on Bayes' Probability Theorem - https://github.com/Juan-Moctezuma/NLP_Sentiment_Analysis-Amazon_Product_Reviews/blob/main/Part7-Whey_Protein_Sentiment_Analysis_Naive_Bayes_Classifier.ipynb
5. <b>Business Analysis: Understanding Customer's Negative Sentiments</b> - https://github.com/Juan-Moctezuma/NLP_Sentiment_Analysis-Amazon_Product_Reviews/blob/main/Part8-Business_Analysis-Root_Cause_of_Negative_Sentiments.ipynb

### Technologies
1. Microsoft Office:
   * Excel
   * PowerPoint
2. Jupyter Notebook (Python 3)
3. Python 3 Libraries:
   * AutoModelForSequenceClassification (for Roberta & Bert models)
   * AutoTokenizer (for Roberta model)
   * BeautifulSoup (for webscraping)
   * datetime
   * decimal
   * math
   * matplotlib
   * nltk (for TextBlob library)
   * numpy
   * pandas
   * re
   * requests
   * seaborn
   * SentimentIntensityAnalyzer (for Vader model)
   * sklearn (for Bert model & Naive Bayes algorithm)
   * softmax
   * statsmodels (for Bert model)
   * scipy
   * textblob (library or model itself)
   * torch (for Bert model)
   * tqdm (for Vader model)

### Other Knowledge required for the completion of this project 
1. Applied Mathematics & Statistics
2. Machine Learning & Data Science methods
3. Programming (Python 3)
4. Data Analytics
5. Business Acumen
