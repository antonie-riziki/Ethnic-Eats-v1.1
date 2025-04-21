
import streamlit as st
import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, sigmoid_kernel

sys.path.insert(1, './modules')

from cleaning import clean_data
from feature_engineering import create_soup
from recommend_general import recommend_food_general
from recommend_narrative import recommend_food_narrative


from dotenv import load_dotenv

load_dotenv()

df = pd.read_csv(r"./src/food recipe/food_recipe.csv")
df = df.fillna('')


new_features = ['name', 'cuisine', 'course', 'ingredients_name', 'diet']
food_data = df[new_features].copy()
for i in new_features:
    food_data[i] = df[i].apply(clean_data)

food_data['soup'] = food_data.apply(create_soup, axis=1)


cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(food_data['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

food_data = food_data.reset_index()
indices = pd.Series(food_data.index, index=food_data['name'])


st.title("🍲 Food Recommendation System")

select_food = st.selectbox("Select a Food Item", df['name'].unique())

if select_food:
    result = recommend_food_general(select_food, cosine_sim, df)

    tf_vect = TfidfVectorizer(min_df=3, ngram_range=(1,3), stop_words='english', strip_accents='unicode')
    result['description'] = result['description'].fillna('')
    desc_matrix = tf_vect.fit_transform(result['description'])
    sig = sigmoid_kernel(desc_matrix, desc_matrix)

    indices_narrative = pd.Series(result.index, result['name']).drop_duplicates()
    recommend_food_narrative(result['name'].iloc[0], result, sig, indices_narrative)
