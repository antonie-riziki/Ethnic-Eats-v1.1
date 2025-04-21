def recommend_food_general(df, name, cosine_sim):
    name = name.replace(' ', '').lower()
    food_data = df.reset_index()
    indices = pd.Series(food_data.index, index=food_data['name'])
    
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:300]
    food_indices = [i[0] for i in sim_scores]
    
    result = df[['name', 'diet', 'cuisine', 'course', 'ingredients_name', 'prep_time (in mins)', 'cook_time (in mins)', 'description']].iloc[food_indices].reset_index(drop=True)
    return result
