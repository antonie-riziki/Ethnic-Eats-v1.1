
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, sigmoid_kernel

# from utils.cleaners import clean_data, create_soup
# from utils.recommender import recommend_food_general, recommend_food_narrative_api

sys.path.insert(1, './modules')

from cleaning import clean_data
from feature_engineering import create_soup
from recommend_general import recommend_food_general
from recommend_narrative import recommend_food_narrative


from dotenv import load_dotenv

load_dotenv()


app = Flask(__name__)

# Load dataset
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

@app.route('/recommend', methods=['GET'])
def recommend():
    food_name = request.args.get('food')
    if not food_name:
        return jsonify({"error": "Please provide a food name"}), 400
    
    try:
        result = recommend_food_general(food_name, cosine_sim, df, indices)
        tf_vect = TfidfVectorizer(min_df=3, ngram_range=(1,3), stop_words='english', strip_accents='unicode')
        result['description'] = result['description'].fillna('')
        desc_matrix = tf_vect.fit_transform(result['description'])
        sig = sigmoid_kernel(desc_matrix, desc_matrix)
        indices_narrative = pd.Series(result.index, result['name']).drop_duplicates()

        recs = recommend_food_narrative_api(food_name, result, sig, indices_narrative)
        return jsonify(recs)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
