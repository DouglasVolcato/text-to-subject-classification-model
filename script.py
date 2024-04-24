from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config
import praw
import re

subjects = ["datascience", "machinelearning", "physics", "astrology", "conspiracy"]

def load_data():
    api_reddit = praw.Reddit(
        client_id="hNJoqF_KAate-k_sMSCHYA",
        client_secret="iM8CK9mgBhDJ59uq5lxSDX6cWF5N5Q",
        user_agent="DouglasVolcato"
    )

    chat_count = lambda post: len(re.sub(r"\W|\d", '', post.selftext))
    mask = lambda post: chat_count(post) >= 100


    data = []
    labels = []

    for i, subject in enumerate(subjects):
        subreddit_data = api_reddit.subreddit(subject).new(limit=1000)
        posts = [post.selftext for post in filter(mask, subreddit_data)]
        data.extend(posts)
        labels.extend([i] * len(posts))

        print(r"Posts number for subject {subject}: {n}".format(subject=subject, n=len(posts)));
        print(r"Some extracted post: {posts[0][:600]}")

    return data, labels

TEST_SIZE = .2
RANDOM_STATE = 0

def split_data(data, labels):
    print(f"Split {100 * TEST_SIZE}% of the data to test and evaluate the model...")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print(f"{len(y_test)} test samples.")
    return x_train, x_test, y_train, y_test

MIN_DOC_FREQ = 2
N_COMPONENTS = 1000
N_EW = 1000

def process__pipeline():
    pattern = r'\W|\d|http.*\s+|www.*\s+'   
    preprocessor = lambda text: re.sub(pattern, ' ', text)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

    vectorizer = TfidfVectorizer(preprocessor=preprocessor, stop_words='english', min_df=MIN_DOC_FREQ)
    decomposition = TruncatedSVD(n_components=N_COMPONENTS, n_iter=N_EW)

    pipelie = [('tfidf', vectorizer), ('svd', decomposition)]
    return pipelie

N_NEIGHBORS = 4
CV = 3

def create_models():
    models = [
        ('KNN', KNeighborsClassifier(n_neighbors=N_NEIGHBORS)),
        ('RandomForest', RandomForestClassifier(random_state=RANDOM_STATE)),
        ('LogReg', LogisticRegressionCV(cv=CV, random_state=RANDOM_STATE))
    ]
    return models

def train_and_evaluate(models, pipeline, X_train, X_test, y_train, y_test):
    
    results = []
    
    for name, model in models:
        pipe = Pipeline(pipeline + [(name, model)])

        # training
        print(f"Training the model {name} with data from training set...")
        pipe.fit(X_train, y_train)

        # predictions with data from test set
        y_pred = pipe.predict(X_test)

        # calculation of metrics
        report = classification_report(y_test, y_pred)
        print("Classification Report\n", report)

        results.append([model, {'model': name, 'predictions': y_pred, 'report': report,}])           

    return results

if __name__ == "__main__":
    # Loading data from Reddit
    data, labels = load_data()

    # Splitting the data into training and test sets
    x_train, x_test, y_train, y_test = split_data(data, labels)

    # Creating the preprocessing pipeline
    pipeline = process__pipeline()

    # Creating the models
    models = create_models()
 
    # Training and evaluating the models
    results = train_and_evaluate(models, pipeline, x_train, x_test, y_train, y_test)

print("Trainment done!")



def plot_distribution():
    _, counts = np.unique(labels, return_counts=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 6), dpi=120)
    plt.title("Number of Posts by Subject")
    sns.barplot(x=subjects, y=counts)
    plt.legend([f'{f.title()} - {c} posts' for f, c in zip(subjects, counts)])
    plt.show()


def plot_confusion(result):
    print("Classification Report\n", result[-1]['report'])
    y_pred = result[-1]['predictions']
    conf_matrix = confusion_matrix(y_test, y_pred)
    _, test_counts = np.unique(y_test, return_counts=True)
    conf_matrix_percent = conf_matrix / test_counts.transpose() * 100
    plt.figure(figsize=(9, 8), dpi=120)
    plt.title(result[-1]['model'].upper() + " results")
    plt.xlabel("True label")
    plt.ylabel("Model prediction")
    ticklabels = [f"{sub}" for sub in subjects]
    sns.heatmap(data=conf_matrix_percent, xticklabels=ticklabels, yticklabels=ticklabels, annot=True, fmt='.2f')
    plt.show()

# Evaluation plot
plot_distribution()

# KNN result
plot_confusion(results[0])

# RandomForest result
plot_confusion(results[1])

# Logistic Regression result
plot_confusion(results[2])
