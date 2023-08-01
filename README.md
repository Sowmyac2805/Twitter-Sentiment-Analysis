# Twitter-Sentiment-Analysis

### Introduction
Twitter is a social media platform where it has around 450 million monthly active users as of 2023 and users can tweet, up to 280 characters. Twitter is extensively used for exchanging thoughts and ideas, networking, marketing, and promoting organizations and products. It is also utilized for real-time news updates. Celebrities, public figures, and politicians frequently use it as a forum to interact with their fans and express their ideas. It enables users to subscribe to accounts and take part in discussions with hashtags, which can help users find and interact with other people who share their interests.

The goal of sentiment analysis, a Natural Language Processing (NLP) technique, is to recognize and extract subjective information from text sources. Twitter sentiment analysis involves analyzing the sentiments expressed in tweets to gain insights into public opinion. It is also helpful for political analysis, public opinion research, and other fields where it's important to comprehend people's thoughts and emotions. This project outlines a comprehensive approach for tackling sentiment analysis problems. It begins with data preprocessing and exploration, followed by the extraction of features from the cleaned text. Ultimately, the article demonstrates the successful construction of multiple models using the feature sets, which were then utilized to classify the tweets.

The goal of this project is to develop a reliable and efficient classifier that can accurately classify the sentiment of a stream of tweets, enabling a better understanding of public sentiment.

### Related work
The research listed below are just a few that have recently been done on Twitter sentiment analysis.

The research article, Twitter Sentiment Classification using Distant Supervision by (Alec Go, Richa Bhayani, Lei Huang) performed their research on sentiment analysis by using machine learning classifiers, Naive Bayes, Maximum Entropy (MaxEnt), and Support Vector Machines (SVM). The feature extractors are unigrams, bigrams, unigrams and bigrams, and unigrams with part of speech tags. The concept of using tweets with emoticons for distant supervised learning is the key contribution of this research.

The research article, A reliable sentiment analysis for classification of tweets in social networks by(Masoud AminiMotlagh, HadiShahriar Shahhoseini, Nina Fatehi) performed their research on sentiment analysis by using K-nearest neighbor, decision tree, support vector machine, and naive Bayes and two bagging ensemble methods.

### Dataset Description

The project will utilize a dataset called Twitter Sentiment Analysis, which was sourced from Kaggle. This dataset contains a total of 27480 tweets.

### Data Attributes Description:
Twitter Sentiment Analysis dataset consists of the following attributes.
• TextID – A unique identifier for each tweet in the dataset.
• Text – The actual text of the tweet that the user posted.
• Selected_text - A subset of the original tweet text that represents the sentiment of the tweet.
• Sentiment - The sentiment of the tweet, which can be either positive, negative, or neutral.
• Time of Tweet - The time of day that the tweet was posted, categorized as morning, noon, or night.
• Age of User - The age group of the user who posted the tweet.
• Country - The country associated with the user who posted the tweet.
• Population -2020 - Population of the country associated with the user who posted the tweet.
• Land Area (Km2) - Area of the Country.
• Density (P/Km2) - Density of the Country.

![image](https://github.com/Sowmyac2805/Twitter-Sentiment-Analysis/assets/120443811/c816b344-7057-4d65-a871-b394e8794ebc)


### Data Preprocessing

An important phase of any machine learning research is data preprocessing. It involves cleaning and transforming raw data into a format that is suitable for training a model. Initially, the process involves verifying if there are any duplicate values or missing values within the dataset, and subsequently eliminating them if they do not have any significant impact on the model. To prepare text data for analysis, common preprocessing techniques include removing punctuation, special characters, digits, white spaces, and stop words. Additionally, it is important to convert all text to lowercase to maintain consistency during the stemming process. Overall, these preprocessing steps help ensure the data is clean and organized, setting a strong foundation for accurate modeling and analysis.

### Handling Missing Values:

One of the essential steps in dataset analysis is handling missing values. As shown in the figure, there was only a single missing value in the ‘text’ and ‘selected_text’ fields. The variables ‘Population -2020’, ‘Land Area (Km2)’, ‘Density (P/Km2)’ which are not necessary for this project, were also removed, along with the missing data.


![image](https://github.com/Sowmyac2805/Twitter-Sentiment-Analysis/assets/120443811/57d5bb08-b9c7-4319-aeb2-e323999d8b8f)


The following figure shows the columns without any missing values.


![image](https://github.com/Sowmyac2805/Twitter-Sentiment-Analysis/assets/120443811/c3c2141d-c4df-4ea5-8d60-52ef4c373041)

### Text Preprocessing:

In this project, the text is being preprocessed by tokenizing it, converting all text to lower case, removing stop words, lemmatizing words, and removing punctuation and special characters. Also used Universal POS tag which is important to correctly lemmatize words. The goal is to transform the text into numerical feature vectors using the Bag of Words strategy and TF-IDF vectorization. This approach involves using CountVectorizer and TF-IDFVectorizer to create a matrix of token counts.

Steps involved in text preprocessing:
Tokenization: Breaking down the text into individual words or tokens.
Lowercasing: Converting all text to lowercase to standardize the text.
Stop word removal: Removing commonly used words like "the" or "and" that don't add much meaning to the text.
Lemmatization: Reducing words to their root form to capture their base meaning.
Punctuation and special character removal: Removing non-alphanumeric characters like commas or parentheses.
Removing numbers or digits: Removing numerical characters, unless they add important context to the text.
Part-of-speech tagging (POS tagging): Labeling each word with its grammatical category to improve lemmatization accuracy.


### Exploratory Data Analysis:

Exploratory data analysis involves using visualizations and statistical methods to analyze data and summarize their main characteristics. By performing exploratory data analysis, one can gain insight into the underlying patterns and trends in the data, which can guide the development of machine learning models.


![image](https://github.com/Sowmyac2805/Twitter-Sentiment-Analysis/assets/120443811/322cdacf-9e63-4231-a60b-3fd14aeaec12)


The above figure shows the distribution of the sentiment column with the positive, negative, and neutral labels.


![image](https://github.com/Sowmyac2805/Twitter-Sentiment-Analysis/assets/120443811/b58f3d99-a889-449c-bcb9-d6484c67eb82)


The above figure displays a list of the most frequently occurring words in the tweets.


### Text Analysis:

One technique for text analysis is the use of n-gram, which are contiguous sequences of n items from a given sample of text. In sentiment analysis, n-grams can be used to identify key phrases or combinations of words that are strongly associated with a particular sentiment or emotion. The following figures shows the most frequent bigram in tweets based on positive, negative, and neutral sentiment.



![image](https://github.com/Sowmyac2805/Twitter-Sentiment-Analysis/assets/120443811/32e462dd-d160-437f-bef5-3711b7b44dce)


![image](https://github.com/Sowmyac2805/Twitter-Sentiment-Analysis/assets/120443811/0af70b74-6d50-404f-9a61-b34378980d18)


![image](https://github.com/Sowmyac2805/Twitter-Sentiment-Analysis/assets/120443811/00562f1f-e405-4d67-8051-4f7ef63f10e7)


### Word Cloud:

Word clouds are a visualization technique used to represent textual data. This technique is used to group words together in a random arrangement, with the size of each word indicating its frequency of occurrence in the text.

The following figures shows the word cloud for Positive, negative, and neutral sentiment tweets.


![image](https://github.com/Sowmyac2805/Twitter-Sentiment-Analysis/assets/120443811/66d59c0c-f0dd-4f5f-b587-3dd78d782212)


![image](https://github.com/Sowmyac2805/Twitter-Sentiment-Analysis/assets/120443811/37451544-7d3a-471f-93fc-1e1286f37415)


![image](https://github.com/Sowmyac2805/Twitter-Sentiment-Analysis/assets/120443811/ea7429df-ed8f-41f6-b79d-61f3541f76b6)


### TF-IDF Vectorization:

The TF-IDF Vectorization converts a collection of raw documents into a matrix of TF-IDF features. The term frequency-inverse document frequency (TF-IDF) is a numerical statistic that reflects how important a word is to a document in a collection or corpus. The resulting feature matrices can be used as input to a machine learning model for sentiment analysis.


![image](https://github.com/Sowmyac2805/Twitter-Sentiment-Analysis/assets/120443811/d780014b-7a1f-4055-bd46-a7bc0a60f2d8)


### Bag Of Words(BoW):

Bag of Words is a text representation technique used to convert a text document into a numerical vector. The BoW model treats a text document as a "bag" or collection of words, ignoring the order in which they appear and focusing only on their frequency of occurrence. The process of creating a BoW representation typically involves two steps: tokenization, which involves splitting the text into individual words or tokens, and counting, which involves tallying the frequency of each word in the document. It is a simple and effective approach for many text- based tasks.

### Model Development:

Once the preprocessing, exploratory analysis, feature extraction, and text analysis steps were completed, the next step was to develop the model. To do this, we used a 75/25 train-test split approach. The sentiment attribute represents the sentiment of the tweet, which can be positive, negative, or neutral. It is the target variable for the sentiment analysis, and it is based on the selected text of the tweet. Sentiment classification can be accomplished using machine learning techniques. This project uses several machine learning models including K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), Random Forest, and Decision Tree Classifiers to accomplish this analysis. The performance of each model is evaluated using three metrics: Test Accuracy, Test Precision, and Test Recall.

Test Accuracy measures the proportion of correctly classified samples out of the total number of samples in the testing dataset. Test Precision measures the proportion of true positive samples (i.e., samples correctly identified as positive) out of the total number of positive samples identified by the model. Test Recall measures the proportion of true positive samples out of the total number of actual positive samples in the testing dataset.


![image](https://github.com/Sowmyac2805/Twitter-Sentiment-Analysis/assets/120443811/63926a69-6d9c-4cd4-ad5b-4307b58cd34d)


The above table shows the performance of machine learning models for sentiment classification using the Bag-of-words technique. We can see that RandomForestClassifier achieved the highest Test Accuracy.


![image](https://github.com/Sowmyac2805/Twitter-Sentiment-Analysis/assets/120443811/f8440667-5f19-4112-b3a3-2bd5d7b958db)


The above table shows the performance of models for sentiment classification using the TF-IDF (Term Frequency-Inverse Document Frequency) technique. We can see that RandomForestClassifier achieved the highest Test Accuracy. However, the best model should be selected based on a balance between high accuracy, precision, and recall, and not just based on a single metric.


![image](https://github.com/Sowmyac2805/Twitter-Sentiment-Analysis/assets/120443811/bd99a2f5-22a5-4228-a062-82dfa73644ad)


This plot shows the Test Accuracy of different machine learning models for sentiment classification using both Bag-of-Words (BOW) and TF-IDF approaches. The x-axis represents the different machine learning models being evaluated, and the y-axis represents the Test Accuracy metric.
The results show that the RandomForestClassifier consistently achieves the highest accuracy with both Bag of Words and TF-IDF vectorization. Therefore, we can conclude that the RandomForestClassifier is the most suitable model for classifying tweets based on sentiment in our dataset.


### Bag of Words Models

The above table shows the performance of machine learning models for sentiment classification using the Bag-of-words technique. We can see that RandomForestClassifier achieved the highest Test Accuracy.

The above table shows the performance of models for sentiment classification using the TF-IDF (Term Frequency-Inverse Document Frequency) technique. We can see that RandomForestClassifier achieved the highest Test Accuracy. However, the best model should be selected based on a balance between high accuracy, precision, and recall, and not just based on a single metric.
  
This plot shows the Test Accuracy of different machine learning models for sentiment classification using both Bag-of-Words (BOW) and TF-IDF approaches. The x-axis represents the different machine learning models being evaluated, and the y-axis represents the Test Accuracy metric.
The results show that the RandomForestClassifier consistently achieves the highest accuracy with both Bag of Words and TF-IDF vectorization. Therefore, we can conclude that the RandomForestClassifier is the most suitable model for classifying tweets based on sentiment in our dataset.

### Conclusion:

In conclusion, this Twitter sentiment analysis project demonstrated the process of collecting, preprocessing, analyzing, and classifying tweets based on their sentiment. This analysis has many potential applications, including brand monitoring, market research, and political analysis.

We used TF-IDF and BoW approach in our analysis, While both approaches have their strengths and limitations, the decision ultimately depends on factors such as the size of the dataset, the desired level of accuracy, and the available resources. For example, BoW may be a more suitable choice for smaller datasets or environments with limited resources, while TF-IDF may be better for larger datasets or more nuanced analysis. As the field of sentiment analysis continues to evolve, it is important to carefully evaluate and choose the most appropriate method for each project.

Future work in Twitter sentiment analysis could explore multi-lingual sentiment analysis, fine- grained sentiment analysis, contextual analysis, sentiment analysis for specific domains, and real-time sentiment analysis. These areas of research could help to improve the accuracy and relevance of sentiment analysis in the context of Twitter data and could have applications in fields such as brand monitoring, market research, and political analysis.
Overall, Twitter sentiment analysis is a powerful tool for gaining insights into public opinion and sentiment on a variety of topics.

### References:
[1] Sentiment Analysis: Machine Learning Approach. (n.d.). Retrieved from
https://www.kaggle.com/code/poojag718/sentiment-analysis-machine-learning- approach#Twitter-Sentiment-Analysis

[2] NLTK :: Natural Language Toolkit. (n.d.). Retrieved from https://www.nltk.org/

[3] AminiMotlagh, M., Shahhoseini, H., & Fatehi, N. (2022, December 12). A reliable sentiment analysis for classification of tweets in social networks. https://doi.org/10.1007/s13278-022- 00998-2

[4] Matplotlib — Visualization with Python. (2023, February 14). Retrieved from https://matplotlib.org/

[5] scikit-learn: machine learning in Python &mdash; scikit-learn 1.2.2 documentation. (n.d.). Retrieved from https://scikit-learn.org/stable/
       
