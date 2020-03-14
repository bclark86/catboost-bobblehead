import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from src.data import load_bobbleheads_data
from src.utils import parse_yaml


settings = parse_yaml('config.yaml')

st.title('Bobblehead Promotion Impact')
st.markdown("""
The app allows you to predict the increase in the number of tickets sold
for a specified MLB game based on whether or not a bobblehead promotional giveaway
is activated on the day. 
"""
)


@st.cache(allow_output_mutation=True)
def load_data():
    data = load_bobbleheads_data(settings['data_dir'])
    return data


@st.cache
def load_model():
    model_path = os.path.join(settings['model_dir'], 'estimator.pkl')
    model = joblib.load(model_path)
    return model


def load_train():
    data = pd.read_csv(os.path.join(settings['data_dir'], 'training', 'train.csv'))
    return data


def load_test():
    data = pd.read_csv(os.path.join(settings['data_dir'], 'training', 'test.csv'))
    return data


def make_prediction(model, new_data):
    prediction = model.predict(new_data)
    return np.round(prediction)


data = load_data()

st.sidebar.text('Prediction Inputs')

home_team = st.sidebar.selectbox(
    'Home Team',
    data['home_team'].unique()
)

opponent = st.sidebar.selectbox(
    'Opponent',
    [team for team in data['opponent'].unique()
     if team != home_team]
)

month = st.sidebar.selectbox(
    'Month',
    data['month'].unique()
)

day_of_week = st.sidebar.selectbox(
    'Day of Week',
    data['day_of_week'].unique()
)

day_night = st.sidebar.selectbox(
    'Day/Night',
    data['day_night'].unique()
)

skies = st.sidebar.selectbox(
    'Skies',
    data[data.home_team == home_team]['skies'].unique()
)

temp = st.sidebar.slider(
    'Temperature',
    0, 110, 70
)

cap = st.sidebar.checkbox(
    'Cap'
)

shirt = st.sidebar.checkbox(
    'Shirt'
)

fireworks = st.sidebar.checkbox(
    'Fireworks'
)

promo_options = ['cap', 'shirt', 'fireworks']
promo_selected = [cap, shirt, fireworks]

promos = [idx for idx, val in zip(promo_options, promo_selected) if val]
promos = ', '.join(promos)

if len(promos) < 1:
    promos = 'None'

bobble_df = pd.DataFrame({
    'year': '2012',
    'home_team': home_team,
    'month': month,
    'day': 1,
    'attend': 0,
    'day_of_week': day_of_week,
    'opponent': opponent,
    'temp': temp,
    'skies': skies,
    'day_night': day_night,
    'cap': np.where(cap, 'YES', 'NO'),
    'shirt': np.where(shirt, 'YES', 'NO'),
    'fireworks': np.where(fireworks, 'YES', 'NO'),
    'bobblehead': 'YES'
}, index=['Game'])


model = load_model()

bobble_pred = make_prediction(model, bobble_df)[0]

no_bobble_df = bobble_df.copy()
no_bobble_df['bobblehead'] = 'NO'
no_bobble_pred = make_prediction(model, no_bobble_df)[0]

bobble_effect = bobble_pred - no_bobble_pred
bobble_lift = bobble_effect / no_bobble_pred

train_data = load_train()
test_data = load_test()
train_data['predicted_attend'] = make_prediction(model, train_data)
test_data['predicted_attend'] = make_prediction(model, test_data)

test_rmse = np.sqrt(mean_squared_error(test_data.attend, test_data.predicted_attend))
test_r2 = r2_score(test_data.attend, test_data.predicted_attend)

bobble_samples = np.random.normal(bobble_pred, test_rmse, 10000)
no_bobble_samples = np.random.normal(no_bobble_pred, test_rmse, 10000)

confidence = np.mean(bobble_samples > no_bobble_samples)

st.subheader('Game Info:')
st.text(f'{opponent} at {home_team} ({day_of_week}, {month})')
st.text(f'Weather: {temp} degrees and {skies} ({day_night})')
st.text(f'Other promotions: {promos}')
st.text('')
st.subheader('Prediction:')
st.text(f'Attendance Prediction (bobblehead): {int(bobble_pred):,}')
st.text(f'Attendance Prediction (no bobblehead): {int(no_bobble_pred):,}')
st.text(f'# of Ticket Sales Increase: {int(bobble_effect):,}')
st.text(f'# of Ticket Sales Lift (%): {bobble_lift * 100:.1F}%')
st.text(f'Confidence: {confidence * 100:.1F}%')
st.text('')
st.subheader('Model:')
st.text(f'Test R2: {np.round(test_r2, 2)}')
st.text(f'Test RMSE: {np.round(test_rmse)}')
full_data = pd.concat([train_data, test_data], keys=['train', 'test'])
full_data.index = full_data.index.get_level_values(0)
full_data = full_data.reset_index().rename(columns={'index': 'data'})

scatter = alt.Chart(full_data).mark_circle(size=60).encode(
    x=alt.X('attend', axis=alt.Axis(format="~s", title='Actual Attendance')),
    y=alt.X('predicted_attend', axis=alt.Axis(format="~s", title='Predicted Attendance')),
    color=alt.Color('data', scale=alt.Scale(range=['#d62728', '#7f7f7f'])),
    tooltip=['home_team', 'opponent', 'month', 'day',
             'attend', 'predicted_attend',
             'day_of_week', 'temp', 'skies', 'day_night',
             'cap', 'shirt', 'fireworks', 'bobblehead']
).properties(
    title='Predictive Performance'
)

overlay_df = pd.DataFrame({
    'x': np.round(np.linspace(0, 60_000, 100))
})

overlay_df['y'] = overlay_df.x

line = alt.Chart(overlay_df).mark_line(color='black', strokeWidth=0.25).encode(
    x='x', y='y'
)

st.altair_chart(scatter + line)

