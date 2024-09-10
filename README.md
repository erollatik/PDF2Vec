# VectorPDF

VectorPDF extracts text from PDF documents, converts words into embedding vectors, and stores them in a vector database. This enables advanced search and similarity queries, allowing the discovery of semantic relationships between texts. Ideal for Natural Language Processing (NLP) and information retrieval projects.

## Features

- Extracts text from PDF files
- Converts words into embedding vectors using pre-trained models
- Stores vectors in a vector database (FAISS)
- Performs similarity searches to find semantically related words or phrases

## Technologies Used

- Python
- [PyPDF2](https://pypi.org/project/PyPDF2/) for extracting text from PDFs
- [Sentence-Transformers](https://www.sbert.net/) for creating word embeddings
- [FAISS](https://faiss.ai/) for storing and querying vectors

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/vectorpdf.git
    cd vectorpdf
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Extract Text from PDF:**

    Use the provided script to extract text from a PDF file:

    ```python
    import PyPDF2

    def extract_text_from_pdf(pdf_file):
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    pdf_file = 'example.pdf'
    text = extract_text_from_pdf(pdf_file)
    print(text)
    ```

2. **Generate Embeddings:**

    Convert extracted text into embeddings:

    ```python
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    words = text.split()
    word_embeddings = model.encode(words)

    for word, embedding in zip(words, word_embeddings):
        print(f"Word: {word}\nVector: {embedding}\n")
    ```

3. **Store and Search Vectors:**

    Store the embeddings in FAISS and perform similarity searches:

    ```python
    import faiss
    import numpy as np

    embedding_dim = word_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(word_embeddings))

    query_word = "example"
    query_vector = model.encode([query_word])

    k = 5
    distances, indices = index.search(query_vector, k)

    print(f"\nTop {k} words closest to '{query_word}':")
    for i in range(k):
        print(f"Word: {words[indices[0][i]]}, Distance: {distances[0][i]}")
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
