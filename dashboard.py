import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=Space+Mono&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
        background-color: #080810;
        color: #e8e8f0;
    }

    .stApp {
        background: #080810;
    }

    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f0f1a;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    [data-testid="stSidebar"] * {
        color: #e8e8f0 !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 0px;
        padding: 20px 24px;
        transition: border-color 0.3s;
    }

    [data-testid="metric-container"]:hover {
        border-color: rgba(0,245,196,0.4);
    }

    [data-testid="metric-container"] label {
        font-family: 'Space Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: #6b6b80 !important;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 36px !important;
        font-weight: 800 !important;
        color: #00f5c4 !important;
    }

    /* Buttons */
    .stButton > button {
        background: #00f5c4;
        color: #080810;
        font-family: 'Space Mono', monospace;
        font-size: 13px;
        letter-spacing: 2px;
        text-transform: uppercase;
        border: none;
        padding: 14px 32px;
        border-radius: 0px;
        width: 100%;
        font-weight: 700;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        background: #7b61ff;
        color: white;
        transform: translateY(-2px);
    }

    /* Selectbox & Slider */
    .stSelectbox > div > div,
    .stSlider > div {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 0px !important;
        color: #e8e8f0 !important;
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.06) !important;
    }

    /* Multiselect */
    .stMultiSelect > div {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 0px !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #080810; }
    ::-webkit-scrollbar-thumb { background: #00f5c4; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── PLOTLY THEME ─────────────────────────────────────────────────────────────
PLOT_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(255,255,255,0.02)',
    font=dict(family='Syne', color='#e8e8f0'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True),
    margin=dict(l=20, r=20, t=40, b=20),
)

COLORS = {
    'survived': '#00f5c4',
    'died': '#ff4d6d',
    'accent': '#7b61ff',
    'muted': '#6b6b80'
}

# ── DATA ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df.drop(columns=['Cabin'], inplace=True)
    return df

df = load_data()

# ── ML MODEL ──────────────────────────────────────────────────────────────────
@st.cache_resource(hash_funcs={pd.DataFrame: lambda x: None})
def train_model(df):
    ml_df = df.copy()
    le = LabelEncoder()
    ml_df['Sex'] = le.fit_transform(ml_df['Sex'])
    X = ml_df[['Pclass', 'Sex', 'Age']]
    y = ml_df['Survived']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 0 30px 0;'>
        <div style='font-family: Space Mono; font-size: 11px; color: #00f5c4; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 8px;'>// Dashboard</div>
        <div style='font-size: 28px; font-weight: 800; line-height: 1.1;'>Titanic<br/>Analysis</div>
    </div>
    <hr style='border-color: rgba(255,255,255,0.06); margin-bottom: 24px;'/>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-family: Space Mono; font-size: 10px; color: #6b6b80; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 12px;'>Filters</div>", unsafe_allow_html=True)

    gender = st.multiselect(
        "Gender",
        options=df['Sex'].unique(),
        default=df['Sex'].unique()
    )

    pclass = st.multiselect(
        "Passenger Class",
        options=sorted(df['Pclass'].unique()),
        default=sorted(df['Pclass'].unique()),
        format_func=lambda x: f"Class {x}"
    )

    age_range = st.slider(
        "Age Range",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(int(df['Age'].min()), int(df['Age'].max()))
    )

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family: Space Mono; font-size: 10px; color: #6b6b80; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px;'>About</div>
    <div style='font-size: 13px; color: #6b6b80; line-height: 1.7;'>
    Built by <span style='color: #00f5c4;'>Mohammed Masood</span><br/>
    CSE-DS · BIT Bangalore<br/>
    <a href='https://github.com/masood-mashu' style='color: #7b61ff;'>github.com/masood-mashu</a>
    </div>
    """, unsafe_allow_html=True)

# ── APPLY FILTERS ─────────────────────────────────────────────────────────────
filtered_df = df[
    (df['Sex'].isin(gender)) &
    (df['Pclass'].isin(pclass)) &
    (df['Age'] >= age_range[0]) &
    (df['Age'] <= age_range[1])
]

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 40px 0 32px 0;'>
    <div style='font-family: Space Mono; font-size: 11px; color: #00f5c4; letter-spacing: 4px; text-transform: uppercase; margin-bottom: 12px;'>// Data Analysis Project</div>
    <div style='font-size: 52px; font-weight: 800; line-height: 1.0; margin-bottom: 12px;'>Titanic Survival<br/><span style='background: linear-gradient(135deg, #00f5c4, #7b61ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Dashboard</span></div>
    <div style='font-family: Space Mono; font-size: 13px; color: #6b6b80; letter-spacing: 1px;'>April 15, 1912 · North Atlantic Ocean · 2,224 Passengers</div>
</div>
<hr style='border-color: rgba(255,255,255,0.06); margin-bottom: 32px;'/>
""", unsafe_allow_html=True)

# ── METRICS ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
survival_rate = round(filtered_df['Survived'].mean() * 100, 1)
survived = filtered_df['Survived'].sum()
died = len(filtered_df) - survived

c1.metric("Total Passengers", f"{len(filtered_df):,}")
c2.metric("Survived", f"{survived:,}")
c3.metric("Perished", f"{died:,}")
c4.metric("Survival Rate", f"{survival_rate}%")
c5.metric("Avg Age", f"{round(filtered_df['Age'].mean(), 1)} yrs")

st.markdown("<br/>", unsafe_allow_html=True)

# ── ROW 1: SURVIVAL + GENDER ──────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div style='font-family: Space Mono; font-size: 10px; color: #6b6b80; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 16px;'>01 — Survival Overview</div>", unsafe_allow_html=True)
    survival_data = filtered_df['Survived'].value_counts().reset_index()
    survival_data.columns = ['Status', 'Count']
    survival_data['Status'] = survival_data['Status'].map({1: 'Survived', 0: 'Perished'})

    fig = go.Figure(data=[go.Pie(
        labels=survival_data['Status'],
        values=survival_data['Count'],
        hole=0.6,
        marker_colors=[COLORS['survived'], COLORS['died']],
        textinfo='percent',
        textfont=dict(family='Space Mono', size=13),
    )])
    fig.update_layout(**PLOT_THEME, height=320,
        annotations=[dict(text=f'<b>{survival_rate}%</b><br>Survived', x=0.5, y=0.5,
                         font=dict(size=16, color='#00f5c4', family='Syne'), showarrow=False)])
    fig.update_traces(showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("<div style='font-family: Space Mono; font-size: 10px; color: #6b6b80; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 16px;'>02 — Survival by Gender</div>", unsafe_allow_html=True)
    gender_df = filtered_df.groupby(['Sex', 'Survived']).size().reset_index(name='Count')
    gender_df['Status'] = gender_df['Survived'].map({1: 'Survived', 0: 'Perished'})

    fig = px.bar(gender_df, x='Sex', y='Count', color='Status',
                 color_discrete_map={'Survived': COLORS['survived'], 'Perished': COLORS['died']},
                 barmode='group', text='Count')
    fig.update_layout(**PLOT_THEME, height=320, showlegend=True,
                      xaxis_title='', yaxis_title='Passengers')
    fig.update_traces(textposition='outside', textfont=dict(family='Space Mono', size=11))
    st.plotly_chart(fig, use_container_width=True)

# ── ROW 2: CLASS + AGE ────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div style='font-family: Space Mono; font-size: 10px; color: #6b6b80; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 16px;'>03 — Survival by Class</div>", unsafe_allow_html=True)
    class_df = filtered_df.groupby(['Pclass', 'Survived']).size().reset_index(name='Count')
    class_df['Status'] = class_df['Survived'].map({1: 'Survived', 0: 'Perished'})
    class_df['Class'] = class_df['Pclass'].map({1: 'First', 2: 'Second', 3: 'Third'})

    fig = px.bar(class_df, x='Class', y='Count', color='Status',
                 color_discrete_map={'Survived': COLORS['survived'], 'Perished': COLORS['died']},
                 barmode='group', text='Count')
    fig.update_layout(**PLOT_THEME, height=320, xaxis_title='', yaxis_title='Passengers')
    fig.update_traces(textposition='outside', textfont=dict(family='Space Mono', size=11))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("<div style='font-family: Space Mono; font-size: 10px; color: #6b6b80; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 16px;'>04 — Age Distribution</div>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=filtered_df[filtered_df['Survived']==1]['Age'],
        name='Survived', nbinsx=25,
        marker_color=COLORS['survived'], opacity=0.75
    ))
    fig.add_trace(go.Histogram(
        x=filtered_df[filtered_df['Survived']==0]['Age'],
        name='Perished', nbinsx=25,
        marker_color=COLORS['died'], opacity=0.75
    ))
    fig.update_layout(**PLOT_THEME, height=320, barmode='overlay',
                      xaxis_title='Age', yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)

# ── ROW 3: FARE + EMBARKED ────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div style='font-family: Space Mono; font-size: 10px; color: #6b6b80; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 16px;'>05 — Fare vs Age (by Survival)</div>", unsafe_allow_html=True)

    fig = px.scatter(filtered_df, x='Age', y='Fare',
                     color=filtered_df['Survived'].map({1: 'Survived', 0: 'Perished'}),
                     color_discrete_map={'Survived': COLORS['survived'], 'Perished': COLORS['died']},
                     opacity=0.6, size_max=8)
    fig.update_layout(**PLOT_THEME, height=320, xaxis_title='Age', yaxis_title='Fare (£)')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("<div style='font-family: Space Mono; font-size: 10px; color: #6b6b80; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 16px;'>06 — Embarkation Port</div>", unsafe_allow_html=True)

    port_map = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
    port_df = filtered_df.copy()
    port_df['Port'] = port_df['Embarked'].map(port_map)
    port_survival = port_df.groupby(['Port', 'Survived']).size().reset_index(name='Count')
    port_survival['Status'] = port_survival['Survived'].map({1: 'Survived', 0: 'Perished'})

    fig = px.bar(port_survival, x='Port', y='Count', color='Status',
                 color_discrete_map={'Survived': COLORS['survived'], 'Perished': COLORS['died']},
                 barmode='stack')
    fig.update_layout(**PLOT_THEME, height=320, xaxis_title='', yaxis_title='Passengers')
    st.plotly_chart(fig, use_container_width=True)

# ── INSIGHTS ──────────────────────────────────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("<div style='font-family: Space Mono; font-size: 10px; color: #6b6b80; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 20px;'>// Key Insights</div>", unsafe_allow_html=True)

i1, i2, i3, i4 = st.columns(4)

with i1:
    st.markdown("""
    <div style='background: rgba(0,245,196,0.06); border: 1px solid rgba(0,245,196,0.2); padding: 20px; height: 100px;'>
        <div style='font-size: 20px; margin-bottom: 8px;'>👩</div>
        <div style='font-size: 13px; color: #e8e8f0; line-height: 1.6;'>Women had a <span style='color:#00f5c4; font-weight:700;'>74% survival rate</span> vs 19% for men</div>
    </div>
    """, unsafe_allow_html=True)

with i2:
    st.markdown("""
    <div style='background: rgba(123,97,255,0.06); border: 1px solid rgba(123,97,255,0.2); padding: 20px; height: 100px;'>
        <div style='font-size: 20px; margin-bottom: 8px;'>🎩</div>
        <div style='font-size: 13px; color: #e8e8f0; line-height: 1.6;'>1st class passengers had <span style='color:#7b61ff; font-weight:700;'>63% survival</span> vs 24% in 3rd</div>
    </div>
    """, unsafe_allow_html=True)

with i3:
    st.markdown("""
    <div style='background: rgba(255,77,109,0.06); border: 1px solid rgba(255,77,109,0.2); padding: 20px; height: 100px;'>
        <div style='font-size: 20px; margin-bottom: 8px;'>👶</div>
        <div style='font-size: 13px; color: #e8e8f0; line-height: 1.6;'>Children under 10 had a <span style='color:#ff4d6d; font-weight:700;'>higher survival rate</span> across all classes</div>
    </div>
    """, unsafe_allow_html=True)

with i4:
    st.markdown("""
    <div style='background: rgba(0,245,196,0.06); border: 1px solid rgba(0,245,196,0.2); padding: 20px; height: 100px;'>
        <div style='font-size: 20px; margin-bottom: 8px;'>⚓</div>
        <div style='font-size: 13px; color: #e8e8f0; line-height: 1.6;'>Cherbourg passengers had the <span style='color:#00f5c4; font-weight:700;'>highest survival rate</span> of all ports</div>
    </div>
    """, unsafe_allow_html=True)

# ── ML PREDICTOR ──────────────────────────────────────────────────────────────
st.markdown("<br/><br/>", unsafe_allow_html=True)
st.markdown("""
<hr style='border-color: rgba(255,255,255,0.06);'/>
<div style='padding: 32px 0 24px 0;'>
    <div style='font-family: Space Mono; font-size: 11px; color: #00f5c4; letter-spacing: 4px; text-transform: uppercase; margin-bottom: 12px;'>// ML Predictor</div>
    <div style='font-size: 36px; font-weight: 800;'>Would <span style='background: linear-gradient(135deg, #00f5c4, #7b61ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>YOU</span> have survived?</div>
    <div style='color: #6b6b80; font-size: 15px; margin-top: 8px;'>Powered by Random Forest · Enter your details below</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    user_pclass = st.selectbox("Passenger Class", [1, 2, 3], format_func=lambda x: f"Class {x} {'(Upper)' if x==1 else '(Middle)' if x==2 else '(Lower)'}")

with col2:
    user_sex = st.selectbox("Gender", ["female", "male"], format_func=lambda x: x.capitalize())

with col3:
    user_age = st.slider("Age", 1, 80, 25)

st.markdown("<br/>", unsafe_allow_html=True)

if st.button("🔮  Predict My Survival"):
    user_sex_encoded = 0 if user_sex == "female" else 1
    prediction = model.predict([[user_pclass, user_sex_encoded, user_age]])
    probability = model.predict_proba([[user_pclass, user_sex_encoded, user_age]])
    survival_prob = round(probability[0][1] * 100, 1)
    death_prob = round(probability[0][0] * 100, 1)

    st.markdown("<br/>", unsafe_allow_html=True)

    if prediction[0] == 1:
        st.markdown(f"""
        <div style='background: rgba(0,245,196,0.08); border: 1px solid rgba(0,245,196,0.3); padding: 32px; text-align: center;'>
            <div style='font-size: 48px; margin-bottom: 12px;'>🎉</div>
            <div style='font-size: 28px; font-weight: 800; color: #00f5c4; margin-bottom: 8px;'>You Would Have SURVIVED!</div>
            <div style='font-family: Space Mono; font-size: 14px; color: #6b6b80;'>Survival Probability: <span style='color: #00f5c4; font-size: 20px; font-weight: 700;'>{survival_prob}%</span></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background: rgba(255,77,109,0.08); border: 1px solid rgba(255,77,109,0.3); padding: 32px; text-align: center;'>
            <div style='font-size: 48px; margin-bottom: 12px;'>💀</div>
            <div style='font-size: 28px; font-weight: 800; color: #ff4d6d; margin-bottom: 8px;'>You Would NOT Have Survived.</div>
            <div style='font-family: Space Mono; font-size: 14px; color: #6b6b80;'>Survival Probability: <span style='color: #ff4d6d; font-size: 20px; font-weight: 700;'>{survival_prob}%</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Probability bar
    st.markdown("<br/>", unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        x=[survival_prob, death_prob],
        y=['Survived', 'Perished'],
        orientation='h',
        marker_color=[COLORS['survived'], COLORS['died']],
        text=[f'{survival_prob}%', f'{death_prob}%'],
        textposition='inside',
        textfont=dict(family='Space Mono', size=13, color='#080810')
    ))
    fig.update_layout(**PLOT_THEME, height=160, showlegend=False)
    fig.update_xaxes(range=[0, 100], showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

# ── RAW DATA ──────────────────────────────────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)
with st.expander("📁 View Raw Data"):
    st.dataframe(
        filtered_df.style.applymap(
            lambda v: 'color: #00f5c4' if v == 1 else 'color: #ff4d6d' if v == 0 else '',
            subset=['Survived']
        ),
        use_container_width=True
    )

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<br/><hr style='border-color: rgba(255,255,255,0.06);'/>
<div style='display: flex; justify-content: space-between; padding: 20px 0; font-family: Space Mono; font-size: 11px; color: #6b6b80; letter-spacing: 1px;'>
    <div>© 2026 <span style='color: #00f5c4;'>Mohammed Masood</span> · BIT Bangalore</div>
    <div>Built with Python · Streamlit · Plotly</div>
</div>
""", unsafe_allow_html=True)