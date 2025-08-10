# 🤖 ClassBot – A New Multilingual Chatbot Using Machine Learning

## 📌 Overview
ClassBot is a multilingual academic chatbot designed specifically for **MNNIT Allahabad students**.  
It helps bridge language barriers by allowing users to query academic materials, previous year questions, and institutional guidelines in multiple languages, making academic resources more accessible.

---

## 🚀 Features
- 🌏 **Multilingual Support** – Query in multiple languages with translation powered by **mBART**.
- 📚 **Academic Query Handling** – Course details, admission info, departmental guidelines, and more.
- 📄 **Document Interaction** – Upload PDFs, query notes, books, and previous year question papers.
- 🧠 **Retrieval-Augmented Generation (RAG)** for accurate, context-aware answers.
- 🖼 **OCR & Text Extraction** – Extracts text from scanned documents and images.
- 🎯 **Clean & Minimal UI** built with **Streamlit**.

---

## 🛠 Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: Mistral 7B LLM, Hugging Face Transformers, SentenceTransformers
- **Vector Search**: FAISS
- **Translation**: mBART
- **OCR**: Tesseract

---

## 🏗 System Architecture
1. **Query Processing** – User inputs a question, converted into semantic embeddings.
2. **Context Retrieval** – FAISS retrieves the top relevant document chunks.
3. **Generative Response** – Mistral LLM produces an answer.
4. **Post-Processing** – Format output for display in the chat interface.

---

## 🔄 RAG Pipeline
1. **Embed query** using SentenceTransformers.
2. **Retrieve top-k chunks** from FAISS vector database.
3. **Pass context + query** to Mistral 7B via LangChain’s RetrievalQA.
4. **Generate final response** in the requested language.

---

## 📊 Results
- Multilingual translation of academic queries.
- Retrieval of links to previous year questions.
- Real-time document querying.
- OCR for scanned notes/images.

---

## ⚠ Limitations
- ❌ No multi-turn memory – each question is independent.
- ❌ No external real-time data fetching.
- ❌ Limited to pre-uploaded MNNIT datasets.

---

## 🔮 Future Scope
- 🎙 **Voice input/output** for better accessibility.
- 👍 **Feedback loop** for improving model accuracy.
- 📝 **Conversation history** for personalized interactions.

---

