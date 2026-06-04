import streamlit as st
from data_loader import load_data
from ai_engine import AIEngine

st.set_page_config(page_title="AI Library System", layout="wide")

df = load_data()
ai = AIEngine(df)

st.title("📚 AI-Powered Smart Library System")

st.sidebar.title("Navigation")

menu = st.sidebar.selectbox(
    "Choose Feature",
    [
        "🔎 Semantic Search",
        "📖 Book Recommendation",
        "💬 Ask My Book AI",
        "📝 Book Summarizer",
        "🎯 Study Path Generator",
        "📊 Demo Mode"
    ]
)

if menu == "🔎 Semantic Search":
    st.header("Semantic Search 🔎")
    query = st.text_input("Enter your search query")
    if st.button("Search"):
        st.write(ai.semantic_search(query))

elif menu == "📖 Book Recommendation":
    st.header("Book Recommendation 📖")
    book = st.text_input("Enter a book title")
    if st.button("Recommend"):
        st.write(ai.recommend(book))

elif menu == "💬 Ask My Book AI":
    st.header("Ask My Book AI 💬")
    query = st.text_input("Ask anything")
    if st.button("Ask"):
        st.write(ai.ask_my_book(query))

elif menu == "📝 Book Summarizer":
    st.header("Book Summarizer 📝")
    book = st.text_input("Enter book title")
    if st.button("Summarize"):
        st.write(ai.summarize(book))

elif menu == "🎯 Study Path Generator":
    st.header("Study Path Generator 🎯")
    goal = st.text_input("Enter your goal (e.g. AI Engineer)")
    if st.button("Generate"):
        st.write(ai.study_path(goal))

elif menu == "📊 Demo Mode":
    st.header("System Demo 🚀")
    st.write("Semantic Search Example:")
    st.write(ai.semantic_search("machine learning"))

    st.write("Recommendation Example:")
    st.write(ai.recommend(df['title'].iloc[0]))

    st.write("AI Chat Example:")
    st.write(ai.ask_my_book("python"))

    st.success("Demo Completed Successfully")
