"""
fonction_rag.py
---------------
Recherche dans la base vectorielle et génération de réponses RAG.

Ce module couvre :
  1. la recherche sémantique (similarity search) dans Chroma
  2. l'initialisation du LLM local via Ollama
  3. la construction du prompt RAG et l'appel au modèle
  4. le formatage des erreurs Ollama (serveur / modèle manquant)
"""

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


# ---------------------------------------------------------------------------
# Recherche sémantique
# ---------------------------------------------------------------------------

def search_documents(vectorstore, question, k=4):
    """
    Recherche les k chunks les plus proches de la question
    dans la base vectorielle (similarité cosinus via embeddings).

    Ne fait PAS appel au LLM : renvoie uniquement les extraits trouvés.
    """

    results = vectorstore.similarity_search(
        question,
        k=k
    )

    return results


# ---------------------------------------------------------------------------
# LLM (Ollama)
# ---------------------------------------------------------------------------

# Mis en cache pour ne pas réinstancier le client à chaque message Streamlit.
@st.cache_resource
def create_llm():
    """Initialise le LLM Ollama (une seule fois en mémoire)."""

    llm = OllamaLLM(
        model="mistral"
    )
    return llm


def format_ollama_error(error):
    """
    Transforme une exception Ollama en message clair pour l'utilisateur
    (serveur non lancé, modèle manquant, etc.).
    """

    message = str(error).lower()

    # Ollama n'est pas démarré / inaccessible
    if any(
        keyword in message
        for keyword in (
            "connection refused",
            "connect",
            "connection error",
            "failed to establish",
            "max retries",
            "name or service not known",
            "nodename nor servname",
        )
    ):
        return (
            "Ollama ne semble pas lancé ou est inaccessible. "
            "Démarrez-le (`ollama serve`), puis vérifiez que le modèle "
            "`mistral` est installé (`ollama pull mistral`)."
        )

    # Modèle non téléchargé / introuvable
    if any(
        keyword in message
        for keyword in ("not found", "404", "pull", "model")
    ) and any(
        keyword in message
        for keyword in ("not found", "404", "does not exist", "unknown model")
    ):
        return (
            "Le modèle Ollama `mistral` est introuvable. "
            "Installez-le avec : `ollama pull mistral`."
        )

    return f"Erreur lors de l'appel au LLM : {error}"


# ---------------------------------------------------------------------------
# Génération RAG
# ---------------------------------------------------------------------------

def generate_rag_response(llm, results, question):
    """
    Génère une réponse à partir des chunks récupérés + de la question.

    Principe RAG :
      - on construit un contexte avec les extraits trouvés
      - on force le LLM à répondre UNIQUEMENT à partir de ce contexte
      - s'il n'y a pas l'info, le modèle doit le dire clairement

    Lève l'exception en cas d'échec Ollama (à gérer côté UI).
    """

    # Concatène les contenus des chunks pour former le contexte
    context = "\n\n".join(
        doc.page_content
        for doc in results
    )

    # Prompt strict : pas de connaissances externes
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
            Tu es un assistant qui répond uniquement à partir
            des documents fournis.

            Utilise exclusivement le contexte ci-dessous pour
            répondre à la question.

            Si la réponse ne se trouve pas dans le contexte,
            dis clairement que l'information n'est pas présente
            dans les documents.

            Ne complète pas la réponse avec tes connaissances
            personnelles.

            ### Contexte
            {context}

            ### Question
            {question}

            ### Réponse
            """
        )

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    # Appel au modèle local (Ollama / mistral)
    response = llm.invoke(final_prompt)

    return response
