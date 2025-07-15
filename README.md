# Hermes: A Conversational Warehouse Assistant

**Hermes** is a natural language-driven chatbot developed in Python for managing warehouse inventory tasks such as stock checking, ordering, and removal. The chatbot features dynamic conversation flow, intent recognition via machine learning, and personality-adaptable small talk powered by TF-IDF and cosine similarity.

---

## Features

 **Inventory Management**:
  - Check available stock by item or aisle
  - Place and finalize stock orders
  - Remove items or entire orders from inventory

   **Intent Recognition**:
  - ML-based classifier (Logistic Regression, SVM, Naive Bayes, Random Forest via GridSearchCV)
  - Cosine similarity verification for enhanced intent accuracy

   **Small Talk with Style**:
  - Supports 5 chatbot personalities: *Friendly*, *Caring*, *Enthusiastic*, *Professional*, *Witty*
  - Personalised responses with personality-driven TF-IDF vectorizers and augmented datasets

 **User Customization**:
  - User can rename themselves and the chatbot
  - Personality switching during conversation
  - Persistent user settings saved across sessions

 **Multi-turn Dialogue Support**:
  - Modular state management via a `state_stack` to guide logical flow across ordering, checking, and removal

---

## Project Structure

```bash
Hermes Chatbot
├── Coursework.py                # Main script (entry point)
├── friendly_smalltalk.csv       # Friendly tone small talk
├── caring_smalltalk.csv         # Caring tone small talk
├── enthusiastic_smalltalk.csv   # Enthusiastic tone small talk
├── professional_smalltalk.csv   # Professional tone small talk
├── witty_smalltalk.csv          # Witty tone small talk
├── trained_classifier           # Serialized ML model
├── smalltalk_model_data_*       # Serialized TF-IDF vectorizers and datasets
├── inventory_dataset.csv        # Generated warehouse inventory
├── user_data.json               # Persistent user and chatbot preferences
