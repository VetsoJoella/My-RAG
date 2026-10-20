"""
app.py
------
Interface Streamlit du RAG local.

Ce fichier ne contient QUE l'UI et l'orchestration :
  - sidebar (upload, indexation, gestion des documents, toggle LLM)
  - zone de conversation (recherche sémantique ou RAG)

La logique métier est déléguée à :
  - embedding.py    → ingestion / Chroma / embeddings
  - fonction_rag.py → recherche sémantique + génération LLM
"""

import os

import streamlit as st

# --- Imports métier (ingestion) ---
from embedding import (
    chroma_db_exists,
    clear_all_indexed_documents,
    create_embeddings,
    create_vectorstore,
    delete_indexed_sources,
    get_indexed_sources,
    has_local_documents,
    load_documents,
    load_vectorstore,
    split_documents,
    vectorstore_has_documents,
)

# --- Imports métier (recherche / RAG) ---
from fonction_rag import (
    create_llm,
    format_ollama_error,
    generate_rag_response,
    search_documents,
)


# ---------------------------------------------------------------------------
# Helpers d'affichage (UI uniquement)
# ---------------------------------------------------------------------------

def file_emoji(filename):
    """Retourne une icône emoji selon l'extension du fichier."""

    name = filename.lower()

    if name.endswith(".pdf"):
        return "📄"
    if name.endswith(".txt"):
        return "📝"
    if name.endswith(".md"):
        return "📑"

    return "📁"


def short_filename(filename, max_len=18):
    """Tronque un nom de fichier long pour l'affichage en carte."""

    if len(filename) <= max_len:
        return filename

    # Garde le début + l'extension si possible
    root, ext = os.path.splitext(filename)
    keep = max_len - len(ext) - 1
    if keep < 4:
        return filename[: max_len - 1] + "…"

    return root[:keep] + "…" + ext


# ---------------------------------------------------------------------------
# Configuration de la page Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Local",
    page_icon="🤖",
    layout="wide"
)

# Titre principal
st.title("🤖 Mon RAG Local")
st.write("Système de Retrieval-Augmented Generation")

# Message flash après suppression / vidage (affiché une fois après st.rerun)
if "flash_message" in st.session_state:
    st.success(st.session_state.pop("flash_message"))
if "flash_error" in st.session_state:
    st.error(st.session_state.pop("flash_error"))


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    # Styles dédiés à la sidebar (cartes documents)
    st.markdown(
        """
        <style>
        /* Corps de carte document (icône + nom) */
        section[data-testid="stSidebar"] .doc-card-body {
            text-align: center;
            padding: 2px 4px 8px 4px;
            min-height: 72px;
        }
        section[data-testid="stSidebar"] .doc-card-emoji {
            font-size: 2.1rem;
            line-height: 1.15;
            margin-bottom: 6px;
        }
        section[data-testid="stSidebar"] .doc-card-name {
            font-size: 0.78rem;
            color: #344054;
            font-weight: 600;
            word-break: break-word;
            line-height: 1.25;
        }
        /* Bouton supprimer compact dans la carte */
        section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
        button[kind="secondary"] {
            min-height: 1.85rem;
            padding: 0.1rem 0.35rem;
            border: none;
            background: transparent;
        }
        section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
        button[kind="secondary"]:hover {
            background: #fef3f2;
        }
        /* Conteneurs bordés = cartes */
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
            border-radius: 14px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.header("Configuration")

    # Upload des documents (PDF, TXT, MD)
    uploaded_files = st.file_uploader(
        "Charger vos documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    # Lance le pipeline d'ingestion (embedding.py)
    index_button = st.button(
        "Indexer les documents",
        use_container_width=True
    )

    # Active / désactive le mode RAG avec LLM
    llm_enabled = st.toggle(
        "Activer le LLM",
        value=False
    )

    # =========================
    # GESTION DE LA BASE
    # =========================
    st.divider()
    st.subheader("🗂️ Documents indexés")

    # Fichier cliqué pour suppression immédiate (un par un)
    source_clicked_for_delete = None

    # Liste des documents déjà indexés dans Chroma
    indexed_sources = []
    try:
        if chroma_db_exists():
            sidebar_embeddings = create_embeddings()
            indexed_sources = sorted(
                get_indexed_sources(sidebar_embeddings)
            )
    except Exception as e:
        st.caption(
            f"Impossible de lister les documents indexés : {e}"
        )

    if indexed_sources:
        st.caption(
            f"{len(indexed_sources)} fichier(s)"
        )

        # Grille de cartes : 2 colonnes (icône + nom + bouton supprimer)
        for i in range(0, len(indexed_sources), 2):
            row_sources = indexed_sources[i:i + 2]
            cols = st.columns(2, gap="small")

            for col, source in zip(cols, row_sources):
                with col:
                    emoji = file_emoji(source)
                    label = short_filename(source)

                    # Carte : supprimer en haut à droite, icône au centre, nom en bas
                    with st.container(border=True):
                        top_row = st.columns([4, 1])
                        with top_row[1]:
                            if st.button(
                                "🗑️",
                                key=f"delete_doc_{source}",
                                help=f"Supprimer « {source} »",
                            ):
                                source_clicked_for_delete = source

                        st.markdown(
                            f"""
                            <div class="doc-card-body" title="{source}">
                                <div class="doc-card-emoji">{emoji}</div>
                                <div class="doc-card-name">{label}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        st.markdown("")  # petit espace avant le vidage total

        # Confirmation obligatoire avant de tout vider
        confirm_clear_all = st.checkbox(
            "Confirmer la suppression de toute la base"
        )
        clear_all_button = st.button(
            "Vider toute la base",
            use_container_width=True,
            disabled=not confirm_clear_all,
            type="primary"
        )
    else:
        st.caption("Aucun document indexé pour le moment.")
        clear_all_button = False


# --- Suppression d'un document (via icône 🗑️ sur la carte) ---
if source_clicked_for_delete:

    try:
        embeddings = create_embeddings()
        deleted_chunks = delete_indexed_sources(
            embeddings,
            [source_clicked_for_delete]
        )
        st.session_state["flash_message"] = (
            f"« {source_clicked_for_delete} » supprimé "
            f"({deleted_chunks} chunk(s) retiré(s))."
        )
        st.rerun()
    except Exception as e:
        st.session_state["flash_error"] = (
            "Échec de la suppression du document. "
            f"Détail : {e}"
        )
        st.rerun()


# --- Vidage complet de la base ---
if clear_all_button:

    try:
        embeddings = create_embeddings()
        nb_sources = clear_all_indexed_documents(embeddings)
        st.session_state["flash_message"] = (
            f"Base vidée : {nb_sources} document(s) supprimé(s) "
            "de ChromaDB et du dossier `documents/`."
        )
        st.rerun()
    except Exception as e:
        st.session_state["flash_error"] = (
            "Échec du vidage de la base. "
            f"Détail : {e}"
        )
        st.rerun()


# =========================
# INDEXATION (bouton sidebar)
# =========================
# Pipeline : anti-doublons → load → split → embed → Chroma

if index_button:

    if not uploaded_files:
        st.warning("Veuillez charger au moins un document.")

    else:
        try:
            # Embeddings nécessaires pour interroger Chroma (noms déjà indexés)
            embeddings = create_embeddings()
        except Exception as e:
            st.error(
                "Impossible de charger le modèle d'embeddings. "
                f"Détail : {e}"
            )
            st.stop()

        # Anti-doublons : compare les noms uploadés aux sources déjà indexées
        try:
            already_indexed = get_indexed_sources(embeddings)
        except Exception as e:
            st.error(
                "Impossible de lire la base ChromaDB. "
                f"Détail : {e}"
            )
            st.stop()

        uploaded_names = [f.name for f in uploaded_files]

        duplicates = [
            name for name in uploaded_names
            if name in already_indexed
        ]
        new_files = [
            f for f in uploaded_files
            if f.name not in already_indexed
        ]

        # Fichiers déjà indexés : on prévient et on ne les ré-indexe pas
        if duplicates:
            st.warning(
                "Document(s) déjà indexé(s), ignoré(s) pour éviter les doublons : "
                + ", ".join(duplicates)
            )

        # Rien de nouveau à indexer
        if not new_files:
            st.info(
                "Aucun nouveau document à indexer. "
                "Tous les fichiers sélectionnés sont déjà dans ChromaDB."
            )

        else:
            # 1. Extraction (uniquement les fichiers nouveaux)
            documents, load_errors = load_documents(new_files)

            for error_message in load_errors:
                st.error(error_message)

            if not documents:
                st.warning(
                    "Aucun document valide à indexer "
                    "(tous les fichiers ont échoué ou sont illisibles)."
                )

            else:
                st.success(
                    f"{len(documents)} document(s) chargé(s) "
                    f"({len(new_files) - len(load_errors)} fichier(s) valide(s))."
                )

                # 2. Chunking
                chunks = split_documents(documents)

                st.info(
                    f"{len(chunks)} chunk(s) créé(s)."
                )

                # 3. Embeddings déjà chargés plus haut
                st.success("Modèle d'embeddings chargé !")

                # 4. Écriture dans ChromaDB
                try:
                    vectorstore = create_vectorstore(
                        chunks,
                        embeddings
                    )
                    st.success(
                        "✅ Documents indexés dans ChromaDB !"
                    )
                except Exception as e:
                    st.error(
                        "Échec de l'indexation dans ChromaDB. "
                        f"Détail : {e}"
                    )


# =========================
# ZONE PRINCIPALE : Conversation
# =========================

st.header("Conversation")

# Indicateur d'état du LLM (visible en haut de la zone principale)
if llm_enabled:
    st.success("**LLM activé : ON** — mode RAG (réponse générée)")
else:
    st.warning("**LLM activé : OFF** — mode recherche sémantique uniquement")

# Historique de conversation (persisté entre les reruns Streamlit)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
question = st.chat_input("Posez une question sur vos documents...")

if question:

    # Afficher la question de l'utilisateur
    with st.chat_message("user"):
        st.write(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # --- Vérifications avant recherche / RAG ---
    # 1) Fichiers présents dans documents/
    # 2) Base Chroma non vide (documents effectivement indexés)
    if not has_local_documents():
        response = (
            "Aucun document trouvé dans le dossier `documents/`. "
            "Veuillez d'abord charger et indexer au moins un fichier."
        )
        with st.chat_message("assistant"):
            st.warning(response)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    else:
        # Charger le modèle d'embeddings (mis en cache : 1er appel seulement)
        try:
            embeddings = create_embeddings()
        except Exception as e:
            response = (
                "Impossible de charger le modèle d'embeddings. "
                f"Détail : {e}"
            )
            with st.chat_message("assistant"):
                st.error(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

        else:
            if not vectorstore_has_documents(embeddings):
                response = (
                    "Des fichiers sont présents, mais aucun document n'est indexé "
                    "dans ChromaDB. Cliquez sur « Indexer les documents » avant "
                    "de poser une question."
                )
                with st.chat_message("assistant"):
                    st.warning(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            else:
                try:
                    # Ouvre Chroma (non mis en cache : reflète une ré-indexation)
                    vectorstore = load_vectorstore(
                        embeddings
                    )

                    # Recherche sémantique des chunks pertinents
                    results = search_documents(
                        vectorstore,
                        question,
                        k=4
                    )
                except Exception as e:
                    response = (
                        "Erreur lors de la recherche dans ChromaDB. "
                        f"Détail : {e}"
                    )
                    with st.chat_message("assistant"):
                        st.error(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })

                else:
                    # ---------------------------------
                    # MODE 1 : RECHERCHE SÉMANTIQUE seule
                    # (aucune réponse LLM — uniquement les chunks Chroma)
                    # ---------------------------------

                    if not llm_enabled:

                        # Construit un rendu markdown clair (preuve visuelle TP)
                        history_parts = [
                            "**LLM activé : OFF**",
                            "",
                            "**Question :**",
                            f"> {question}",
                            "",
                            "### Résultats de recherche sémantique",
                            "",
                        ]

                        with st.chat_message("assistant"):
                            st.markdown("**LLM activé : OFF**")
                            st.markdown("**Question :**")
                            st.markdown(f"> {question}")
                            st.markdown("---")
                            st.markdown("### Résultats de recherche sémantique")

                            if not results:
                                st.info(
                                    "Aucun chunk pertinent trouvé dans la base vectorielle."
                                )
                                history_parts.append(
                                    "_Aucun chunk pertinent trouvé._"
                                )
                            else:
                                for doc in results:
                                    # Nom exact du fichier depuis les métadonnées
                                    source = doc.metadata.get(
                                        "source",
                                        "Source inconnue"
                                    )
                                    emoji = file_emoji(source)
                                    chunk_text = (
                                        doc.page_content or ""
                                    ).strip()

                                    st.markdown("---")
                                    st.markdown(f"{emoji} **{source}**")
                                    st.markdown(f"> {chunk_text}")

                                    history_parts.append("---")
                                    history_parts.append(
                                        f"{emoji} **{source}**"
                                    )
                                    history_parts.append(f"> {chunk_text}")
                                    history_parts.append("")

                        response = "\n".join(history_parts)

                    # ---------------------------------
                    # MODE 2 : RAG AVEC LLM
                    # (réponse LLM + sources dépliables = preuve RAG complet)
                    # ---------------------------------

                    else:

                        try:
                            llm = create_llm()
                            llm_answer = generate_rag_response(
                                llm,
                                results,
                                question
                            )
                        except Exception as e:
                            # Ollama non lancé, modèle manquant, etc.
                            response = format_ollama_error(e)
                            with st.chat_message("assistant"):
                                st.error(response)
                        else:
                            with st.chat_message("assistant"):
                                # 1) État du mode RAG
                                st.markdown("**LLM activé : ON**")

                                # 2) Question envoyée au système
                                st.markdown("**Question :**")
                                st.markdown(f"> {question}")

                                st.markdown("---")

                                # 3) Réponse réellement générée par le LLM local
                                st.markdown("**🤖 Réponse :**")
                                st.markdown(f"> {llm_answer}")

                                # 4) Sources / contexte (replié par défaut)
                                with st.expander(
                                    "📚 Voir les sources / contexte",
                                    expanded=False
                                ):
                                    if not results:
                                        st.info(
                                            "Aucun chunk n'a été récupéré "
                                            "comme contexte."
                                        )
                                    else:
                                        for doc in results:
                                            # Nom exact depuis les métadonnées
                                            source = doc.metadata.get(
                                                "source",
                                                "Source inconnue"
                                            )
                                            emoji = file_emoji(source)
                                            chunk_text = (
                                                doc.page_content or ""
                                            ).strip()

                                            st.markdown(
                                                f"{emoji} **{source}**"
                                            )
                                            st.markdown("---")
                                            st.markdown(chunk_text)
                                            st.markdown("")

                            # Historique : réponse LLM (les sources restent
                            # consultables dans le message affiché ci-dessus
                            # au moment de la génération)
                            response = (
                                f"**LLM activé : ON**\n\n"
                                f"**Question :**\n> {question}\n\n"
                                f"**🤖 Réponse :**\n> {llm_answer}"
                            )

                    # Sauvegarder la réponse dans l'historique
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
