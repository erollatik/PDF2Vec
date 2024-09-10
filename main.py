import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# Extract text

def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    return text

pdf_file = '/Users/erolatik/Desktop/kitaplar/Adam Phillips - Akıl Sağlığı Üzerine.pdf'
text = extract_text_from_pdf(pdf_file)

# Metni vektörlere dönüştürme
# Bert modeli
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

words = text.split()

word_embeddings = model.encode(words)

# for word, embedding in zip(words, word_embeddings):
#     print(f"Kelime: {word}\nVektör: {embedding}\n")

# Embedding'leri FAISS'e eklemek
embedding_dim = word_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)

index.add(np.array(word_embeddings))

print(f"Indekste {index.ntotal} vektör var.")

query_word = "verebilecek"
query_vector = model.encode([query_word])

k = 5
distances, indices = index.search(query_vector, k)

print(f"\n '{query_word}' kelimesine en yakın {k} kelime:")
for i in range(k):
    print(f"Kelime: {words[indices[0][i]]}, Distance: {distances[0][i]}")
