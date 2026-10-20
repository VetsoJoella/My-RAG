"""
embedding.py
------------
Ingestion des sources dans la base vectorielle Chroma.

Ce module couvre le pipeline :
  1. chargement des fichiers (PDF / TXT / MD)
  2. découpage en chunks (chunking)
  3. création du modèle d'embeddings
  4. écriture / lecture / suppression dans ChromaDB
"""

import os

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Dossier de persistance Chroma (vecteurs + métadonnées)
CHROMA_DIR = "chroma_db"

# Dossier local où sont sauvegardés les fichiers uploadés
UPLOAD_DIR = "documents"

# Crée le dossier documents/ s'il n'existe pas encore
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Accès à la base vectorielle
# ---------------------------------------------------------------------------

def load_vectorstore(embeddings):
    """Ouvre la base vectorielle Chroma déjà persistée sur disque."""

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    return vectorstore


def chroma_db_exists():
    """Vérifie si le dossier Chroma existe et contient déjà des données."""

    return (
        os.path.isdir(CHROMA_DIR)
        and len(os.listdir(CHROMA_DIR)) > 0
    )


def has_local_documents():
    """Vérifie qu'au moins un fichier PDF/TXT/MD est présent dans documents/."""

    if not os.path.isdir(UPLOAD_DIR):
        return False

    for name in os.listdir(UPLOAD_DIR):
        if name.lower().endswith((".pdf", ".txt", ".md")):
            return True

    return False


def get_indexed_sources(embeddings):
    """
    Retourne l'ensemble des noms de fichiers déjà indexés dans Chroma
    (via la métadonnée "source" enregistrée à l'indexation).
    """

    if not chroma_db_exists():
        return set()

    vectorstore = load_vectorstore(embeddings)
    stored = vectorstore.get(include=["metadatas"])

    sources = set()
    for metadata in stored.get("metadatas") or []:
        if metadata and metadata.get("source"):
            sources.add(metadata["source"])

    return sources


def vectorstore_has_documents(embeddings):
    """True si la base vectorielle contient au moins un chunk indexé."""

    if not chroma_db_exists():
        return False

    vectorstore = load_vectorstore(embeddings)
    stored = vectorstore.get(include=[])
    return len(stored.get("ids") or []) > 0


# ---------------------------------------------------------------------------
# Suppression / nettoyage
# ---------------------------------------------------------------------------

def delete_indexed_sources(embeddings, sources):
    """
    Supprime de Chroma tous les chunks liés aux fichiers sélectionnés,
    puis efface les fichiers locaux correspondants dans documents/.

    Retourne le nombre de chunks (vecteurs) supprimés.
    """

    if not sources:
        return 0

    vectorstore = load_vectorstore(embeddings)
    deleted_chunks = 0

    for source in sources:
        # Récupère les IDs de tous les chunks ayant cette métadonnée "source"
        stored = vectorstore.get(
            where={"source": source},
            include=[]
        )
        ids = stored.get("ids") or []

        if ids:
            vectorstore.delete(ids=ids)
            deleted_chunks += len(ids)

        # Supprime aussi le fichier physique s'il est encore dans documents/
        file_path = os.path.join(UPLOAD_DIR, source)
        if os.path.isfile(file_path):
            os.remove(file_path)

    return deleted_chunks


def clear_all_indexed_documents(embeddings):
    """
    Vide entièrement la collection Chroma (reset)
    et supprime les fichiers locaux qui étaient indexés.

    Retourne le nombre de fichiers (sources) concernés.
    """

    sources = get_indexed_sources(embeddings)
    vectorstore = load_vectorstore(embeddings)

    # Recrée une collection vide (plus simple et sûr qu'un delete massif)
    vectorstore.reset_collection()

    for source in sources:
        file_path = os.path.join(UPLOAD_DIR, source)
        if os.path.isfile(file_path):
            os.remove(file_path)

    return len(sources)


# ---------------------------------------------------------------------------
# Ingestion : chargement → chunking → embeddings → indexation
# ---------------------------------------------------------------------------

def load_documents(uploaded_files):
    """
    Charge les fichiers uploadés.

    Étapes pour chaque fichier :
      - sauvegarde dans documents/
      - choix du loader selon l'extension
      - extraction du texte
      - ajout de la métadonnée "source" (nom du fichier)

    Retourne un tuple (documents, errors) :
    - documents : textes extraits avec succès
    - errors    : messages d'erreur pour les fichiers corrompus / illisibles
    """

    documents = []
    errors = []

    for uploaded_file in uploaded_files:

        file_name = uploaded_file.name

        # Chemin local du fichier
        file_path = os.path.join(
            UPLOAD_DIR,
            file_name
        )

        try:
            # Sauvegarde du fichier sur le disque
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Sélection du loader adapté au format
            if file_name.endswith(".pdf"):

                loader = PyMuPDFLoader(file_path)

            elif file_name.endswith(".txt"):

                loader = TextLoader(
                    file_path,
                    encoding="utf-8"
                )

            elif file_name.endswith(".md"):

                loader = TextLoader(
                    file_path,
                    encoding="utf-8"
                )

            else:
                errors.append(
                    f"{file_name} : format non supporté (ignoré)."
                )
                continue

            # Extraction du contenu textuel
            loaded_documents = loader.load()

            # Fichier ouvert mais sans texte exploitable
            if not loaded_documents or all(
                not (doc.page_content or "").strip()
                for doc in loaded_documents
            ):
                errors.append(
                    f"{file_name} : fichier vide ou illisible "
                    "(aucune page de texte extraite)."
                )
                continue

            for doc in loaded_documents:
                # Conservation du nom du fichier pour anti-doublons / citations
                doc.metadata["source"] = file_name

            documents.extend(loaded_documents)

        # PDF corrompu, encodage invalide, fichier illisible, etc.
        except Exception as e:
            errors.append(
                f"{file_name} : impossible de lire le fichier "
                f"({type(e).__name__} — {e})."
            )

    return documents, errors


def split_documents(documents):
    """
    Découpe les documents en chunks pour l'indexation vectorielle.

    chunk_size    : taille max d'un chunk (en caractères)
    chunk_overlap : chevauchement entre chunks pour ne pas couper le sens
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=70
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


# Chargé une seule fois grâce à @st.cache_resource (évite de recharger
# le modèle HuggingFace à chaque question Streamlit).
# Modèle multilingue : bien adapté au français (contrairement à all-MiniLM-L6-v2,
# orienté anglais).
# Important : après un changement de modèle d'embeddings, il faut ré-indexer
# les documents (les vecteurs déjà en base ne sont plus comparables).
@st.cache_resource
def create_embeddings():
    """Charge le modèle d'embeddings multilingue (une seule fois en mémoire)."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return embeddings


def create_vectorstore(chunks, embeddings):
    """
    Indexe les chunks dans ChromaDB.

    Chaque chunk est transformé en vecteur (embedding) puis stocké
    dans le dossier persist_directory pour survivre aux redémarrages.
    """

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vectorstore
