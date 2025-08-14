# chatbot/rag_chatbot.py

import os
import csv
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- Config ---
DATA_PATH = './data/realistic_restaurant_reviews.csv'
INDEX_PATH = './embeddings/faiss_index.bin'
N_RESULTS = 3

# Charger les variables d'environnement
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

if not OPENAI_API_KEY:
    raise ValueError("La clé API OpenAI n'est pas définie dans le fichier .env.")


# Initialisation OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Charger les avis
def load_texts(path):
    texts = []
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            texts.append(row['Review'])
    return texts

# Recherche dans FAISS
def search_faiss(query, index, model, texts, k=N_RESULTS):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [texts[i] for i in indices[0]]

# Génération de réponse avec GPT-4
def generate_answer(query, context):
    context_text = "\n".join(context)
    prompt = f"""
Tu es un assistant qui répond aux questions en te basant UNIQUEMENT sur les avis clients fournis.
Avis clients :
{context_text}

Question :
{query}

Réponse :
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # tu peux utiliser gpt-4o ou gpt-4-turbo si dispo
        messages=[
            {"role": "system", "content": "Tu es un expert en analyse d'avis clients."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=300
    )
    return response.choices[0].message.content

def main():
    print("Chargement des données...")
    texts = load_texts(DATA_PATH)

    print("Chargement du modèle et de l'index FAISS...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index(INDEX_PATH)

    while True:
        query = input("\nVotre question (ou 'exit' pour quitter) : ")
        if query.lower() == "exit":
            break

        context = search_faiss(query, index, model, texts)
        print("\n--- Avis pertinents ---")
        for c in context:
            print(f"- {c}")
            print("\n")

        try :
            answer = generate_answer(query, context)
            print("\n--- Réponse générée ---")
            print(answer)
        except Exception as e:
            print(f"\n\nErreur lors de la génération de la réponse : {e}")

if __name__ == "__main__":
    main()
