import streamlit as st
from qa_utils import run_query
st.set_page_config(page_title="Restaurent QA Assistant", layout="centered")
from without_db_utils import run_query_without_db

def main():
    st.title("Ask QUE About Restaurent(Review)")
    query = st.text_input("Ask a question:")
    if st.button("Get Answer"):
        if query.strip():
            with st.spinner("Fetching answer..."):
                try:
                    answer = run_query_without_db(query)
                    st.success(answer)
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()