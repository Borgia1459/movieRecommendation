import numpy as np
import pandas as pd
from config import RATINGS
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score

class RecommendationSystem:
    def load_data(self):
        """Load the ratings data from the CSV file.
        -save the ratings file to the ratings variable.
        return the number of ratings each movie_id index
        mapping.
        data_path= path of the rating data file
        n_users= number of users
        n_movies= number of movies that have ratings
        return: rating data in the numpy array of [user,movie];
        movie_n_rating, {movie_id: # of ratings};
        movie_id_mapping,{movie_id: column index in rating data}
        """
        ratings = pd.read_csv(RATINGS)
        # Assuming the CSV has columns 'user_id', 'movie_id', 'rating'
        data = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0).values
        movie_n_rating = ratings.groupby('movie_id').size().to_dict()
        movie_id_mapping = {movie_id: idx for idx, movie_id in enumerate(ratings['movie_id'].unique())}
        return data, movie_n_rating, movie_id_mapping

    def highly_rated(self, movie_n_rating):
        """Find the movie with the highest number of ratings."""
        return max(movie_n_rating, key=movie_n_rating.get)

    def target_movie(self, movie_id, movie_id_mapping, data):
        """Target a specific movie based on movie_id."""
        if movie_id in movie_id_mapping:
            column_index = movie_id_mapping[movie_id]
            X = data[:, column_index].reshape(-1, 1)
            Y = data[:, column_index]
            return X, Y
        else:
            print(f'Movie ID : {movie_id} not found in mapping.')
            return None, None

    def split_data(self, X, Y):
        """Split the data into training and testing sets."""
        return train_test_split(X, Y, test_size=0.2, random_state=42)

    def train(self, x_train, Y_train, x_test, Y_test):
        """Train the model and evaluate its performance."""
        clf = MultinomialNB(alpha=1.0, fit_prior=True)
        clf.fit(x_train, Y_train)
        prediction = clf.predict(x_test)
        prediction_prob = clf.predict_proba(x_test)[:, 1]

        print('Prediction: ', prediction[:10])
        print(f'Accuracy: {np.mean(prediction == Y_test) * 100:.2f}%')
        print(f'Confusion matrix:\n{confusion_matrix(Y_test, prediction)}')
        print(f'Classification report:\n{classification_report(Y_test, prediction)}')
        print(f'Precision: {precision_score(Y_test, prediction):.2f}')
        print(f'Recall: {recall_score(Y_test, prediction):.2f}')
        print(f'F1 score: {f1_score(Y_test, prediction):.2f}')
        print(f'ROC AUC: {roc_auc_score(Y_test, prediction_prob):.2f}')
        return prediction[:10]

    def cross_validation(self, X, Y):
        """Perform cross-validation to find the best hyperparameters."""
        k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_record = {}

        for alpha in [1, 2, 3, 4, 5, 6]:
            for fit_prior in [True, False]:
                auc_scores = []
                for train_indices, test_indices in k_fold.split(X, Y):
                    x_train, x_test = X[train_indices], X[test_indices]
                    Y_train, Y_test = Y[train_indices], Y[test_indices]

                    clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
                    clf.fit(x_train, Y_train)
                    prediction_prob = clf.predict_proba(x_test)[:, 1]
                    auc_scores.append(roc_auc_score(Y_test, prediction_prob))

                auc_record[(alpha, fit_prior)] = np.mean(auc_scores)
                print(f'Alpha: {alpha}, Fit Prior: {fit_prior}, Mean AUC: {np.mean(auc_scores):.2f}')

        best_params = max(auc_record, key=auc_record.get)
        print(f'Best parameters: Alpha: {best_params[0]}, Fit Prior: {best_params[1]}, AUC: {auc_record[best_params]:.2f}')

if __name__ == '__main__':
    rs = RecommendationSystem()
    data, movie_n_rating, movie_id_mapping = rs.load_data()
    highest_rated_movie_id = rs.highly_rated(movie_n_rating)
    X, y = rs.target_movie(movie_id=highest_rated_movie_id, movie_id_mapping=movie_id_mapping, data=data)

    if X is not None and y is not None:
        y_binary = (y > 3).astype(int)
        x_train, x_test, Y_train, Y_test = rs.split_data(X, y_binary)
        rs.train(x_train, Y_train, x_test, Y_test)
        rs.cross_validation(X, y_binary)