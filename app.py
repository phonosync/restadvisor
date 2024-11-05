import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from surprise.dump import dump, load
from sklearn.metrics.pairwise import cosine_similarity

model = load('algo.pickle')[1]

df_osm = pd.read_csv('Mappe3.csv', sep=';')

# Fix latitude and longitude values
df_osm['@lat'] = df_osm['@lat'] / 1e7
df_osm['@lon'] = df_osm['@lon'] / 1e7

# Rename columns
df_osm = df_osm.rename(columns={'@lat': 'lat', '@lon': 'lon'})

# st.write(df_osm)

df_ratings = pd.read_csv('Ratings.csv', sep=';')
# st.write(df_ratings)

# Function to rate a restaurant
def rate_restaurant(i, col, restaurant_list):
    restaurant_name = col.selectbox(f'Wähle ein Restaurant {i+1}', restaurant_list)
    rating = col.number_input(f'Bewerte das Restaurant {i+1}', min_value=1, max_value=5, step=1)
    return restaurant_name, rating

def get_rests_chart(df):
        # Calculate the bounding box of the coordinates
    min_lat = df['lat'].min()
    max_lat = df['lat'].max()
    min_lon = df['lon'].min()
    max_lon = df['lon'].max()

    # Calculate the center of the bounding box
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Calculate the zoom level based on the extent of the coordinates
    lat_diff = max_lat - min_lat
    lon_diff = max_lon - min_lon
    zoom = 13 - max(lat_diff, lon_diff)  # Adjust the zoom level calculation as needed

    # Display the restaurants on a map with tooltips
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=df,
        get_position='[lon, lat]',
        get_radius=25,
        get_color=[255, 0, 0],
        pickable=True,
        auto_highlight=True,
        tooltip=True,
    )

    tooltip = {
        "html": "<b>Name:</b> {name}<br/><b>Küche:</b> {cuisine}",
        "style": {"color": "white"}
    }

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,
    )

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style='mapbox://styles/mapbox/streets-v11'  # Set the map style to light
    )
    return r

st.title('Restadvisor')
st.write('Restaurant-Empfehlungsdienst für Studierende in Winterthur')
st.write('Die Auswahl:')
# Display the restaurants on a map
# st.map(df_osm[['lat', 'lon']])


st.pydeck_chart(get_rests_chart(df_osm))

# provide new ratings
st.write('Bewerte drei Restaurants:')

# Get unique restaurant names
restaurant_names = df_osm['name'].unique()

# Ensure the user rates 3 different restaurants
rated_restaurants = []
for i, col in enumerate(st.columns(3)):
    restaurant_name, rating = rate_restaurant(i, col, restaurant_names)
    osm_id = df_osm[df_osm['name'] == restaurant_name]['@id'].values[0]
    rated_restaurants.append(('xuser', osm_id, rating))
    # Remove the selected restaurant from the list to avoid duplicate selection
    restaurant_names = restaurant_names[restaurant_names != restaurant_name]

# Display the selected restaurants and ratings
# for i, (restaurant_name, rating) in enumerate(rated_restaurants):
#    st.write(f'Sie haben {restaurant_name} mit {rating} bewertet.')


# Convert the new user's ratings to a DataFrame
new_user_ratings = pd.DataFrame(rated_restaurants, columns=['USER_ID', 'ITEM_ID', 'RATING'])
# Append the new user's ratings to df_ratings using pd.concat
df_ratings_xuser = pd.concat([df_ratings, new_user_ratings], ignore_index=True)

# st.write(df_ratings_xuser[-10:])

# st.write(df_osm[df_osm['@id'].isin([e[1] for e in rated_restaurants])])

# Remove duplicate entries by taking the mean of the ratings
df_ratings_xuser = df_ratings_xuser.groupby(['USER_ID', 'ITEM_ID'], as_index=False).mean()

# Create a user-item matrix
user_item_matrix = df_ratings_xuser.pivot(index='USER_ID', columns='ITEM_ID', values='RATING').fillna(0)

# st.write(user_item_matrix)

# Ensure the new user is the last row in the user-item matrix
new_user_vector = user_item_matrix.loc['xuser'].values.reshape(1, -1)
other_users_matrix = user_item_matrix.drop(index='xuser').values

# Calculate cosine similarity
cosine_sim = cosine_similarity(new_user_vector, other_users_matrix)
similar_users_indices = np.argsort(cosine_sim[0])[-3:][::-1]  # Get indices of 3 most similar users

# Get the user IDs of the most similar users
similar_user_ids = user_item_matrix.drop(index='xuser').index[similar_users_indices]

# st.write(similar_user_ids)

# if model.method == 'kNN':

#     # Get recommendations for the 3 most similar users
#     recommendations = []
#     for idx in similar_users_indices:
#         user_id = user_item_matrix.index[idx]
#         user_recommendations = model.get_neighbors(user_id, k=3)
#         recommendations.extend(user_recommendations)

#     # Aggregate and return the top 3 recommendations
#     recommendations = sorted(recommendations, key=lambda x: x.est, reverse=True)[:3]
#     st.write('Top 3 Empfehlungen für Sie:')
#     for rec in recommendations:
#         st.write(f'Item ID: {rec.iid}, Predicted Rating: {rec.est}')

# elif model.method == 'SVD':
# Get recommendations for the new user
all_items = df_osm['@id'].unique()
rated_items = new_user_ratings['ITEM_ID'].unique()
unrated_items = [item for item in all_items if item not in rated_items]

combined_top_predictions = []
for u_id in similar_user_ids:
    p = [model.predict(u_id, item) for item in unrated_items]
    for pred in sorted(p, key=lambda x: x.est, reverse=True)[:3]:
        combined_top_predictions.append(pred)

recs = [pred for pred in sorted(combined_top_predictions, key=lambda x: x.est, reverse=True)[:3]]

# for rec in recs:
#    st.write(f'Item ID: {rec.iid}, Predicted Rating: {rec.est}')

# Get the rows from df_osm with @id equal to the item_id for the three recommendations
recommended_restaurants = df_osm[df_osm['@id'].isin([pred.iid for pred in recs])]

# Display the top 3 recommendations
st.write('Restaurant-Empfehlungen für Dich:')
st.write(recommended_restaurants)

st.pydeck_chart(get_rests_chart(recommended_restaurants))