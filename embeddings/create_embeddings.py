# embeddings/create_embeddings.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import csv
import os


DATA_PATH = './data/realistic_restaurant_reviews.csv'
INDEX_PATH = 'faiss_index.bin'

def load_texts(path):
    texts = []

    if not os.path.exists(DATA_PATH):
        print(f"Erreur : le fichier {DATA_PATH} n'existe pas.")
        exit(1)

    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            texts.append(row['Review'])
    return texts

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_faiss_index(index, path):
    faiss.write_index(index, path)
    print(f"Index FAISS sauvegardé dans {path}")

def main():
    print("Chargement des avis clients...")
    texts = load_texts(DATA_PATH)
    print(f"{len(texts)} avis chargés.")

    print("Chargement du modèle d'embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Calcul des embeddings...")
    embeddings = model.encode(texts, convert_to_numpy=True)

    print("Création de l'index FAISS...")
    index = create_faiss_index(embeddings)

    print("Sauvegarde de l'index...")
    save_faiss_index(index, INDEX_PATH)

if __name__ == "__main__":
    main()
