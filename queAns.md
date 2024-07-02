# What is the difference between list, tuple, and set in Python?

List: Ordered, mutable collection of elements. Allows duplicate elements.
my_list = [1, 2, 2, 3]

Tuple: Ordered, immutable collection of elements. Allows duplicate elements.
my_tuple = (1, 2, 2, 3)

Set: Unordered, mutable collection of unique elements. Does not allow duplicates.
my_set = {1, 2, 3}


# Q. Explain the difference between deepcopy and shallow copy in Python.

## Shallow Copy: Creates a new object but inserts references into it to the objects found in the original.
import copy
original_list = [[1, 2, 3], [4, 5, 6]]
shallow_copy = copy.copy(original_list)

## Deep Copy: Creates a new object and recursively copies all objects found in the original.
import copy
original_list = [[1, 2, 3], [4, 5, 6]]
deep_copy = copy.deepcopy(original_list)


# Que: How does Python handle memory management?
Answer:
Python uses automatic memory management via garbage collection. It has a private heap containing all Python objects and data structures. The Python memory manager handles allocation and deallocation of memory, keeping track of which objects are in use and which are not. When an object's reference count drops to zero, it's deallocated from memory. Python's gc module provides tools for garbage collection and debugging memory management.


# Q6: How do you import a module in Python and what are the different ways to import?
A6:
In Python, modules are imported using the import statement. There are several ways to import a module:

Import the entire module:
import math
print(math.sqrt(16))  # Output: 4.0
Import specific functions or variables from a module:
 
from math import sqrt, pi
print(sqrt(16))  # Output: 4.0
print(pi)        # Output: 3.141592653589793
Import a module with an alias:

import numpy as np
print(np.array([1, 2, 3]))  # Output: [1 2 3]
Import all names from a module (not recommended for large modules due to namespace conflicts):
 
from math import *
print(sqrt(16))  # Output: 4.0
print(pi)        # Output: 3.141592653589793



# Q9: What is a dictionary and how do you manipulate it?

A9:
A dictionary in Python is an unordered collection of key-value pairs, defined using curly braces {}. Keys must be unique and immutable (e.g., strings, numbers), while values can be of any data type.

Example:
 
my_dict = {"name": "Alice", "age": 25, "city": "New York"}

# Accessing value by key
print(my_dict["name"])  # Output: Alice

# Adding a new key-value pair
my_dict["email"] = "alice@example.com"

# Updating an existing key
my_dict["age"] = 26

# Removing a key-value pair
del my_dict["city"]

print(my_dict)  # Output: {'name': 'Alice', 'age': 26, 'email': 'alice@example.com'}


# Q10: How to handle exceptions in Python? Provide an example using try, except, and finally.

A10:
Exceptions in Python are handled using try and except blocks. The try block contains the code that might raise an exception, while the except block handles the exception. The finally block contains code that will execute regardless of whether an exception occurred or not, often used for cleanup actions.

Example:

try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error occurred: {e}")
finally:
    print("This will always execute.")



# Que: Explain the concept of closures in Python.
Answer:
A closure in Python is a nested function that remembers and has access to variables in the local scope in which it was created, even after the outer function has finished executing. Closures are often used to create function factories or to create functions with persistent state across multiple calls.

Example:

python
Copy code
def outer_function(x):
    def inner_function(y):
        return x + y
    return inner_function

closure = outer_function(5)
print(closure(3))  # Output: 8


# -------------------------------------------------------------------------------------------------------------------------------------

 # What are the types of AI?
Artificial intelligence can be divided into different types on the basis of capabilities and functionalities.

Based on Capabilities:
Weak AI or Narrow AI: Weak AI is capable of performing some dedicated tasks with intelligence. Siri is an example of Weak AI.
General AI: The intelligent machines that can perform any intellectual task with efficiency as a human.
Strong AI: It is the hypothetical concept that involves the machine that will be better than humans and will surpass human intelligence.


# What is Strong AI, and how is it different from the Weak AI?
Strong AI: Strong AI is about creating real intelligence artificially, which means a human-made intelligence that has sentiments, self-awareness, and emotions similar to humans. It is still an assumption that has a concept of building AI agents with thinking, reasoning, and decision-making capabilities similar to humans.

Weak AI: Weak AI is the current development stage of artificial intelligence that deals with the creation of intelligent agents and machines that can help humans and solve real-world complex problems. Siri and Alexa are examples of Weak AI programs.


# 1. Pandas (import pandas as pd)
Q1: What is Pandas and why is it widely used in data analysis?

A1:
Pandas is an open-source data manipulation and analysis library for Python. It provides data structures like Series (one-dimensional) and DataFrame (two-dimensional) that make it easy to work with structured data. Pandas is widely used because it offers powerful and flexible data manipulation capabilities, supports operations on data in various formats (like CSV, Excel, SQL databases), and integrates well with other libraries in the Python data science ecosystem. It simplifies tasks like data cleaning, transformation, and exploration, which are essential in data analysis.

Q2: Explain the concept of a DataFrame in Pandas.

A2:
A DataFrame in Pandas is a two-dimensional, size-mutable, and heterogeneous tabular data structure with labeled axes (rows and columns). It can be thought of as a spreadsheet or a SQL table. DataFrames allow you to store and manipulate data in a structured way, with the ability to perform operations like filtering, grouping, merging, and statistical analysis efficiently. They are versatile and can hold data of different types (e.g., integers, floats, strings) in each column.

# 2. Regular Expressions (import re)
Q3: What are regular expressions and how are they used in Python?

A3:
Regular expressions (regex) are sequences of characters that define search patterns, primarily for string matching and manipulation. They are used to find, match, and manage text in strings. In Python, the re module provides support for regular expressions. You can use regex for tasks like validating inputs (e.g., email addresses, phone numbers), searching for patterns in text, replacing substrings, and splitting strings based on patterns.

Q4: Give an example of a simple regex pattern and explain what it does.

A4:
A simple regex pattern is \d{3}-\d{2}-\d{4}. This pattern matches strings in the format of a U.S. Social Security number (SSN), which consists of three digits, a hyphen, two digits, another hyphen, and four digits (e.g., "123-45-6789"). Here, \d represents any digit, and {n} specifies that the preceding element must occur exactly n times.

 
# Q5: What is NLTK and what are its typical applications in NLP?
A5:
NLTK (Natural Language Toolkit) is a comprehensive Python library for working with human language data (text). It provides easy-to-use interfaces for over 50 corpora and lexical resources, text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, as well as wrappers for industrial-strength NLP libraries. NLTK is typically used for tasks such as text classification, tokenization, stemming and lemmatization, part-of-speech tagging, and named entity recognition.

Q6: Explain the role of stopwords in text processing.

A6:
Stopwords are commonly used words (such as "the", "is", "in", "and") that are often removed in text processing because they carry minimal meaningful information and can clutter the analysis. Removing stopwords helps in focusing on the words that are more significant for tasks like text classification, information retrieval, and topic modeling. NLTK provides a predefined list of stopwords for several languages which can be customized as per the application requirements.

# 4. Lemmatization (from nltk.stem import WordNetLemmatizer)
Q7: What is lemmatization and how does it differ from stemming?

A7:
Lemmatization is the process of reducing words to their base or root form (lemma), considering the context and part of speech of the word. For example, the words "running" and "ran" would both be lemmatized to "run". Lemmatization ensures that the base form of the word is a valid word in the language.

Stemming, on the other hand, is a simpler and more heuristic-based process that removes suffixes from words to reduce them to their root form. For example, "running" might be stemmed to "run", but "ran" could be incorrectly stemmed to "ra". Stemming often results in non-words or incorrectly truncated forms.

Q8: How does the WordNetLemmatizer work in NLTK?

A8:
The WordNetLemmatizer in NLTK uses the WordNet lexical database to map words to their base or dictionary forms (lemmas). It considers the word and its part of speech to perform accurate lemmatization. For example, given the word "better" and its part of speech as an adjective, WordNetLemmatizer would return "good". If no part of speech is provided, it defaults to a noun.

# # 5. TF-IDF Vectorization (from sklearn.feature_extraction.text import TfidfVectorizer)
Q9: What is TF-IDF and why is it important in text analysis?

A9:
TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus). TF-IDF combines two metrics: Term Frequency (TF), which measures how frequently a term appears in a document, and Inverse Document Frequency (IDF), which measures how important or rare a term is across the corpus. The TF-IDF score helps in identifying words that are significant in a document while reducing the weight of commonly used words. It's important in text analysis because it improves the ability to capture meaningful and distinguishing terms in text classification and information retrieval.

Q10: How does the TfidfVectorizer in scikit-learn work?

A10:
The TfidfVectorizer in scikit-learn transforms a collection of raw documents into a matrix of TF-IDF features. It converts text into numerical features by computing the TF-IDF score for each term in each document. It allows for various preprocessing options like tokenization, lowercasing, removing stopwords, and handling n-grams. This matrix can then be used as input for machine learning models for tasks like document classification or clustering.

# 6. Cosine Similarity (from sklearn.metrics.pairwise import cosine_similarity)
Q11: What is cosine similarity and how is it used in text analysis?

A11:
Cosine similarity is a measure of similarity between two non-zero vectors that calculates the cosine of the angle between them. In text analysis, cosine similarity is used to measure how similar two documents are, based on the words they contain. It ranges from -1 (completely dissimilar) to 1 (completely similar), with 0 indicating no similarity. This metric is particularly useful for comparing text vectors in high-dimensional spaces, such as those generated by TF-IDF or word embeddings.

Q12: Describe a scenario where cosine similarity can be applied.

A12:
Cosine similarity can be applied in document clustering or recommendation systems. For example, in a news recommendation system, cosine similarity can be used to find articles that are similar to a user’s previously read articles. By comparing the TF-IDF vectors of the articles, the system can recommend articles that are closely related in terms of content, helping users find relevant and interesting content based on their reading history.



# What is a neural network? Explain its basic components.
A neural network is a series of algorithms that attempt to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.
Neurons (nodes): The fundamental units that receive input, process it, and pass it on to the next layer.

Layers:
Input layer: Receives the input signals.
Hidden layers: Intermediate layers that process the inputs received from the input layer.
Output layer: Produces the final output.
Weights and Biases: Parameters that are learned during training to map inputs to outputs.
Activation Function: Introduces non-linearity into the network, allowing it to learn more complex patterns.


# Explain the difference between classification and regression.
Classification: Predicts discrete labels or categories. Example: Spam detection (spam or not spam).
Regression: Predicts continuous values. Example: Predicting house prices.

# What pre-processing techniques are you most familiar with in Python?
Pre-processing techniques are used to prepare data in Python, and there are many different techniques you can use. Some common ones you might talk about include:

Normalization - In Python, normalization is done by adjusting the values in the feature vector.
Dummy variables - Dummy variables is a pandas technique in which an indicator variable (0 or 1) indicates whether a categorical variable can take a specific value or not.
Checking for outliers - There are many methods for checking for outliers, but some of the most common are univariate, multivariate, and Minkowski errors.


# How do you split training and testing datasets in Python?
In Python, you can do this with the Scikit-learn module, using the train_test_split function. This is used to split arrays or matrices into random training and testing datasets.

Generally, about 75% of the data will go to the training dataset; however you will likely test different iterations.

Here’s a code example:

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.4)






# Chat Models
Chat Models are a core component of LangChain.

A chat model is a language model that uses chat messages as inputs and returns chat messages as outputs (as opposed to using plain text).

LangChain has integrations with many model providers (OpenAI, Cohere, Hugging Face, etc.) and exposes a standard interface to interact with all of these models.

LangChain allows you to use models in sync, async, batching and streaming modes and provides other features (e.g., caching) and more.


# https://python.langchain.com/v0.1/docs/modules/model_io/chat/function_calling/

# https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/

# https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/structured/

