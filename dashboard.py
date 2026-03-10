import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Titanic Dashboard", page_icon="🚢", layout="wide")

# Title
st.title("🚢 Titanic Survival Dashboard")
st.write("An interactive analysis of the Titanic passenger dataset")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df.drop(columns=['Cabin'], inplace=True)
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filter Data")
gender = st.sidebar.multiselect("Select Gender:", options=df['Sex'].unique(), default=df['Sex'].unique())
pclass = st.sidebar.multiselect("Select Passenger Class:", options=sorted(df['Pclass'].unique()), default=sorted(df['Pclass'].unique()))

# Apply filters
filtered_df = df[(df['Sex'].isin(gender)) & (df['Pclass'].isin(pclass))]

# Key metrics
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Passengers", len(filtered_df))
col2.metric("Survived", filtered_df['Survived'].sum())
col3.metric("Died", len(filtered_df) - filtered_df['Survived'].sum())
col4.metric("Survival Rate", f"{round(filtered_df['Survived'].mean() * 100, 1)}%")

st.markdown("---")

# Charts row 1
col1, col2 = st.columns(2)

with col1:
    st.subheader("Survival Count")
    fig, ax = plt.subplots()
    filtered_df['Survived'].value_counts().plot(kind='bar', color=['red', 'green'], ax=ax)
    ax.set_xticklabels(['Died', 'Survived'], rotation=0)
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.subheader("Survival by Gender")
    fig, ax = plt.subplots()
    sns.countplot(x='Sex', hue='Survived', data=filtered_df, palette='Set1', ax=ax)
    ax.set_ylabel("Count")
    st.pyplot(fig)

st.markdown("---")

# Charts row 2
col1, col2 = st.columns(2)

with col1:
    st.subheader("Survival by Passenger Class")
    fig, ax = plt.subplots()
    sns.countplot(x='Pclass', hue='Survived', data=filtered_df, palette='Set2', ax=ax)
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    filtered_df[filtered_df['Survived']==1]['Age'].plot(kind='hist', alpha=0.5, label='Survived', color='green', bins=20, ax=ax)
    filtered_df[filtered_df['Survived']==0]['Age'].plot(kind='hist', alpha=0.5, label='Died', color='red', bins=20, ax=ax)
    ax.set_xlabel("Age")
    ax.legend()
    st.pyplot(fig)

st.markdown("---")

# Insights
st.subheader("🔍 Key Insights")
st.success("✅ Women had a much higher survival rate than men — 'Women and children first!'")
st.error("❌ Class 3 passengers had the highest death rate — lower decks had less access to lifeboats")
st.info("ℹ️ Class 1 passengers had the best survival rate due to their proximity to lifeboats")
st.warning("⚠️ Overall survival rate was only 38.4% — more than half of passengers died")

# Raw data
st.markdown("---")
st.subheader("📁 Raw Data")
st.dataframe(filtered_df)
