import pandas as pd
import numpy as np
from dotenv import load_dotenv


load_dotenv()

books = pd.read_csv("books_with_emotions.csv")


books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpeg",
    books["large_thumbnail"]
)



from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

# ================================
# ✅ Step 1: Load file (correct parsing)
# ================================
documents = []

with open("tagged_description.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        # Remove quotes if present
        line = line.strip('"')

        # Split using colon
        parts = line.split(":", 1)
        isbn = parts[0].strip()
        description = parts[1].strip()

        # Ensure valid ISBN
        if not isbn.isdigit():
            continue

        # ✅ ADD ISBN INTO CONTENT ALSO
        full_text = f"{isbn} {description}"
        documents.append(
            Document(
                page_content=full_text,
                metadata={"isbn": isbn}
            )
        )

print(f"Total documents loaded: {len(documents)}")

# ================================
# ✅ Step 2: Limit for performance
# ================================
documents = documents[:500]
# ================================
# ✅ Step 3: Embeddings
# ================================
embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ================================
# ✅ Step 4: Create vector DB
# ================================
db_books = Chroma.from_documents(
    documents,
    embedding=embedding,
)

print("Vector DB ready ✅")

# ================================
# ✅ Step 5: LLM
# ================================
llm = OllamaLLM(model="llama3")


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:

    recs = db_books.similarity_search(
        query,
        k=initial_top_k,
        filter={"simple_categories": category} if category != "All" else None
    )

    books_list = [int(rec.metadata["isbn"]) for rec in recs]

    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    # Tone sorting
    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs.head(final_top_k)

def recommend_books(query:str,category:str,tone: str ):
    recommendations = retrieve_semantic_recommendations(query,category, tone)
    results =[]
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30])+ "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{','.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"]+ ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
import  gradio as gr

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please Enter a Description of a book:", placeholder="eg: A Story about Forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a Category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended Books", columns= 8, rows = 2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output,
                        )
if __name__ == "__main__":
    dashboard.launch()
