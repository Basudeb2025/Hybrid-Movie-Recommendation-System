# Hybrid-Movie-Recommendation-System
Movie Recommendation System part 3

# Hybrid Recommendation System

## Overview

This project implements a hybrid recommendation system that combines content-based and collaborative filtering techniques. It recommends movies based on both the content of movies and user preferences.

## How It Works

1. **Import Libraries**: 
   - **pandas**: For handling and analyzing data.
   - **scikit-learn**: For calculating similarities between movies.
   - **numpy**: For numerical operations.

2. **Load Data**: 
   - Read three datasets: `movies.csv`, `tags.csv`, and `ratings.csv`. 
     - `movies.csv` contains movie details.
     - `tags.csv` contains movie tags.
     - `ratings.csv` contains user ratings for movies.

3. **Prepare Content-Based Filtering**:
   - **Combine Tags and Genres**: Merge movie tags with movie genres to create a description for each movie.
   - **Compute Content Similarity**: Use TF-IDF (Term Frequency-Inverse Document Frequency) to convert movie descriptions into numerical vectors and calculate how similar each movie is to every other movie based on these descriptions.

4. **Prepare Collaborative Filtering**:
   - **Create User-Movie Matrix**: Make a matrix where rows represent users, columns represent movies, and cells contain ratings given by users.
   - **Filter Movies**: Ensure that only movies present in both the user-movie matrix and the content similarity matrix are used.
   - **Compute Collaborative Similarity**: Calculate how similar movies are based on user ratings.

5. **Hybrid Recommendation**:
   - **Combine Scores**: Use both content-based and collaborative filtering scores to recommend movies. The `alpha` parameter controls the weight given to each type of score.
   - **Get Recommendations**: Sort movies by their combined score and return the top recommendations.

6. **Example Usage**:
   - Call the function with a movie title and a user ID to get a list of recommended movies.

## Code

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load datasets
movies = pd.read_csv('movies.csv')
tags = pd.read_csv('tags.csv')
ratings = pd.read_csv('ratings.csv')

# Prepare Content-Based Filtering
movies = movies.merge(tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index(), on='movieId', how='left')
movies['tag'] = movies['tag'].fillna('')
movies['content'] = movies['genres'] + ' ' + movies['tag']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
content_indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Prepare Collaborative Filtering
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
# Filter movies in the user-item matrix to match those in the content matrix
user_item_matrix = user_item_matrix.loc[:, user_item_matrix.columns.isin(movies['movieId'])]
movies = movies[movies['movieId'].isin(user_item_matrix.columns)]

# Recalculate content similarity matrix with the filtered movies
tfidf_matrix = tfidf.fit_transform(movies['content'])
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
content_indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Compute collaborative filtering similarity
item_similarity = cosine_similarity(user_item_matrix.T)
collab_sim_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Hybrid Recommendation Function
def get_hybrid_recommendations(movie_title, user_id, alpha=0.5):
    """
    Get hybrid recommendations combining content-based and collaborative filtering.
    
    Args:
    - movie_title (str): The title of the movie for content-based recommendations.
    - user_id (int): The ID of the user for collaborative filtering recommendations.
    - alpha (float): Weight for combining content-based and collaborative scores.
    
    Returns:
    - list: List of recommended movie titles.
    """
    if movie_title not in content_indices:
        return "Movie not found in the dataset."
    
    movie_idx = content_indices[movie_title]
    content_scores = content_sim[movie_idx]
    
    if user_id not in user_item_matrix.index:
        return "User not found in the dataset."
    user_ratings = user_item_matrix.loc[user_id]
    collab_scores = collab_sim_df.dot(user_ratings).div(collab_sim_df.sum(axis=1))
    
    # Combine content and collaborative scores
    combined_scores = alpha * content_scores + (1 - alpha) * collab_scores.values
    top_indices = combined_scores.argsort()[-11:][::-1]
    
    return movies['title'].iloc[top_indices[1:]].tolist()

# Example usage
hybrid_recommendations = get_hybrid_recommendations('Die Hard (1988)', user_id=1)
print(hybrid_recommendations)
```
## Advantages
-**Comprehensive Recommendations:** Combines content-based and collaborative filtering methods, making the recommendations more accurate and diverse.
-**Flexibility:** Adjusts the importance of content-based vs. collaborative scores with the alpha parameter.
-**Improved Accuracy:** Addresses the weaknesses of using either method alone, leading to better overall recommendations.

## Disadvantages
-**Complexity:** More complicated than using just one recommendation method.
-**Cold Start Problem:** New movies without enough tags or ratings may not be recommended effectively.
-**Resource Intensive:** Requires computation of similarities based on both content and user ratings, which can be resource-heavy.

## Usage
-**Ensure you have movies.csv, tags.csv, and ratings.csv in your working directory.**
-**Run the script to load data, compute similarities, and get recommendations.**
-**Modify the example movie title and user ID to test different recommendations.**

 ## License

This project is licensed under the NIT Sikkim License. For more details, see the [LICENSE](LICENSE) file.
