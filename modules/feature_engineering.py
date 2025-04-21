from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_soup(x):
    return x['name'] + ' ' + x['cuisine'] + ' ' + x['course'] + ' ' + x['ingredients_name'] + ' ' + x['diet']

def build_count_matrix(df, new_features):
    food_data = df[new_features].copy()

    from .cleaning import clean_data
    for feature in new_features:
        food_data[feature] = food_data[feature].apply(clean_data)

    food_data['soup'] = food_data.apply(create_soup, axis=1)
    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(food_data['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    return cosine_sim, food_data
