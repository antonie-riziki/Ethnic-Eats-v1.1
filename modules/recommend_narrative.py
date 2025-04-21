import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

sys.path.insert(1, './modules')


from image_fetcher import fetch_food_image

def generate_sigmoid_kernel(result):
    tf_vect = TfidfVectorizer(min_df=3, max_features=None, ngram_range=(1, 3), 
                              stop_words='english', strip_accents='unicode')

    result['description'] = result['description'].fillna('')
    desc_matrix = tf_vect.fit_transform(result['description'])
    return sigmoid_kernel(desc_matrix, desc_matrix)

def recommend_food_narrative(meal, result, sig, st):
    indices = pd.Series(result.index, result['name']).drop_duplicates()
    food_list = [x for x in result['name']]

    if meal in food_list:
        idx = indices[meal]
        food_list = list(enumerate(sig[idx]))
        sort_food = sorted(food_list, key=lambda x: x[1], reverse=True)
        top_ten = sort_food[1:21]
        food_rec = [x[0] for x in top_ten]

        st.text("")
        st.text('Food For Thought')
        st.dataframe(result[['name', 'cuisine', 'course', 'diet', 'prep_time (in mins)', 'cook_time (in mins)', 'ingredients_name']].iloc[food_rec])

        for i in range(0, len(food_rec), 3):
            col1, col2, col3 = st.columns(3)
            row = food_rec[i:i + 3]

            with col1:
                if len(row) >= 1:
                    st.image(fetch_food_image(result['name'][row[0]]), caption=result['name'][row[0]], use_container_width=True)
            with col2:
                if len(row) >= 2:
                    st.image(fetch_food_image(result['name'][row[1]]), caption=result['name'][row[1]], use_container_width=True)
            with col3:
                if len(row) >= 3:
                    st.image(fetch_food_image(result['name'][row[2]]), caption=result['name'][row[2]], use_container_width=True)

    else:
        st.text('__DATABASE ERROR___ InvalidIndexError(key)')
        st.text('FOOD NOT FOUND')
