# Downloads - In case the NLTK package doesn't include the stopwords, wordnet, or punkt:
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Imports
import re
import os
import joblib
import random
import json
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# Global Initialisations:
# Gets the current directory:
current_dir = os.path.dirname(os.path.abspath(__file__))

# Initialise the Stemmer and the Lemmatizer:
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Gets the list of stopwords for English language:
stop_words = set(stopwords.words('english'))

# Initialise the Vectorizers that will be used
# These vectorizers are customised based on the various personalisations
# that the chatbot can be configured to use:
vectorizer_friendly = TfidfVectorizer(use_idf = True, sublinear_tf = True)
vectorizer_caring = TfidfVectorizer(use_idf = True, sublinear_tf = True)
vectorizer_enthusiastic = TfidfVectorizer(use_idf = True, sublinear_tf = True)
vectorizer_professional = TfidfVectorizer(use_idf = True, sublinear_tf = True)
vectorizer_witty = TfidfVectorizer(use_idf = True, sublinear_tf = True)
vectorizer_commands = TfidfVectorizer(use_idf = True, sublinear_tf = True)

# Reads the smalltalk datasets into the program:
vocab_processed_friendly = pd.read_csv(os.path.join(current_dir, "friendly_smalltalk.csv"))
vocab_processed_caring = pd.read_csv(os.path.join(current_dir, "caring_smalltalk.csv"))
vocab_processed_enthusiastic = pd.read_csv(os.path.join(current_dir, "enthusiastic_smalltalk.csv"))
vocab_processed_professional = pd.read_csv(os.path.join(current_dir, "professional_smalltalk.csv"))
vocab_processed_witty = pd.read_csv(os.path.join(current_dir, "witty_smalltalk.csv"))

def save_model(vectorizer, tfidf_matrix, processed_data, filename):
    """
    A helper function which saves a copy of the model that has been 
    trained using the selected vectorizer, tfidf_matrix, and processed data, 
    along the file path that has been specified.

    Args:
        vectorizer (Transformer): The Vectorizer Object (from TfidfVectorizer)
        tfidf_matrix (Sparse Matrix): A TFIDF Sparse Matrix which holds the
        representation of the vocabulary that has been used to produce it.
        processed_data (Pandas DataFrame): The vocabulary from the selected dataset.
        filename (String): Holds the file path to the corresponding dataset.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, filename)
    
    model_data = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "processed_data": processed_data,
    }
    joblib.dump(model_data, save_path)
    # print(f"Model saved to {save_path}")

def save_model_ml(classifier, vectorizer_ml):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "trained_classifier")
    
    model_data = {
        'classifier': classifier,
        'vectorizer': vectorizer_ml
    }
    
    joblib.dump(model_data, save_path)
    # print(f"Classifier & Vectorizer saved to {save_path}")
    
def load_model_ml(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    load_path = os.path.join(current_dir, filename)
    
    model_data = joblib.load(load_path)
    
    classifier = model_data['classifier']
    vectorizer = model_data['vectorizer']
    
    # print(f"Classifier and Vectorizer loaded from {load_path}")
    return classifier, vectorizer

def load_model(filename):
    """
    A helper function which loads the model from the 
    location specified by the filename.

    Args:
        filename (String): The path of the file that's being referred to.

    Returns:
        Vectorizer: The Vectorizer that is associated with this dataset.
        TFIDF_Matrix: The TFIDF Matrix that is associated with this dataset.
        Processed_Data: The processed vocabulary that is associated with this dataset.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    load_path = os.path.join(current_dir, filename)
    model_data = joblib.load(load_path)
    # print(f"Model loaded from {load_path}")
    return model_data["vectorizer"], model_data["tfidf_matrix"], model_data["processed_data"]

def load_or_save_ml_model(filename, intents):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, filename)
    if os.path.exists(filepath):
        # print("Model (ML) file exists. Loading the model...")
        classifier, vectorizer_ml = load_model_ml(filepath)
        return classifier, vectorizer_ml
    else:
        # print(f"Model (ML) file doesn't exist. Saving the model...")
        classifier, vectorizer_ml = machine_learning_model_creation(intents)
        save_model_ml(classifier, vectorizer_ml)
        return classifier, vectorizer_ml
    
def read_and_save_model(filepath, vocab, vectorizer):
    """
    Either reads in the model specified by the filepath,
    or saves the model if the filepath cannot be found.

    Args:
        filepath (String): The location of the file.
        vocab (Pandas DataFrame): The vocabulary associated with the file.
        vectorizer (Transformer): The Vectorizer associated with the file.

    Returns:
        Transformer: The Vectorizer associated with the file.
        Sparse Matrix: The TFIDF Matrix associated with the file.
        Pandas DataFrame: The vocabulary associated with the file.
    """
    if os.path.exists(filepath):
        # print("Model file exists. Loading the model...")
        vectorizer, tfidf_matrix_small_talk, augmented_data = load_model(filepath)
        return vectorizer, tfidf_matrix_small_talk, augmented_data
    else:
        # print("Model file does not exist. Saving the model...")
        tfidf_matrix_small_talk, augmented_data = preprocess_smalltalk_dataset(vocab, vectorizer)
        save_model(vectorizer, tfidf_matrix_small_talk, augmented_data, filepath)
        return vectorizer, tfidf_matrix_small_talk, augmented_data

# Defines the list of possible Intents that will be used 
# for Intent Matching within the Chatbot's Architecture
intents = {
    "Check Stock": [
        "How many item do we have in stock?",
        "Can you tell me the current stock levels of item?",
        "What's the inventory status for item?",
        "How many item are available in the warehouse?",
        "Could you check how many item we have in stock?",
        "check stock item",
        "check stock",
        "Can I check the amount of item we have?",
        "Can I check stock for item?",
        "What is the quantity of item in stock?",
        "What's the stock for item like?",
        "How many units of item do we have?",
        "Do we have enough item available?",
        "What items do we have available?",
        "What is currently in the warehouse",
        "Whats the current count of item in the inventory?",
        "Can you provide an update on the stock for item?",
        "Is there enough stock of item",
        "How much of item do we have left",
        "Do we need to restock item"
    ],

    "Order Stock": [
        "Can I order more item?",
        "Please place an order for additional stock of item.",
        "We need to replenish our stock of item",
        "Can you make an order for more item?",
        "I'd like to order some item",
        "order stock item",
        "order many item",
        "order stock",
        "order more item",
        "Can you add more staplers to the stock?",
        "Can I reorder item?",
        "Can you order me item?",
        "Can you delete stock from the warehouse",
        "Can I put in an order for more item",
        "I'd like to place a bulk order for item",
        "Can you add more stock of item to the inventory",
        "I need to order additional supplies of item",
        "Please create an order for item",
        "I'd like to order quantity item please"
    ],

    "Remove Stock": [
        "Can you remove stock of item?",
        "Please reduce the amount of item in the warehouse.",
        "We need to remove some item from our inventory.",
        "Can you take item off the shelf?",
        "remove stock item",
        "remove item",
        "remove some item",
        "I need to decrease the quantity of item.",
        "Can I reduce the stock of item?",
        "Can you remove item?",
        "Id like to remove stock",
        "Please take item out of stock",
        "Can you lower the stock level for item",
        "I want to delete item from the warehouse",
        "Can I place an order for an item"
    ],
    
    "Remove Order": [
        "Can you remove this order",
        "Can you remove an order",
        "Remove order",
        "I need to remove this order",
        "I'd like to remove an order",
        "Please cancel this order for me",
        "Take this order off the list",
        "Can you delete this pending order",
        "I want to remove the order I just placed",
        "Erase this order from the system"
    ],

    "Change Username": [
        "Change my name",
        "Can you change my name?",
        "Can you call me name?",
        "Call me name",
        "I'd like to change my name"
        "I'd like to update ny username to name",
        "Can you set my name to name",
        "Please change the name you call me to name",
        "Switch my name to name"
    ],

    "Change Chatbot Name": [
        "Change your name",
        "Can you change your name?",
        "Can I call you name?",
        "Call yourself name",
        "Whats your new name?",
        "I'd like to call you name",
        "Can we change your name to name",
        "Please update your name to name",
        "Set your name to name",
        "How do I change your name"
    ],

    "Discoverability_General": [
        "What can you do?",
        "What are you capable of?",
        "What are your functions?",
        "What are some of your features?",
        "What are you able to do?",
        "How do I decrease the stock of item?",
        "How do I request more items for the warehouse?",
        "How do I order more stock for the warehouse?",
        "What operations can I perform with you?",
        "What tasks can you help me with",
        "What functionalities do you offer",
        "What services are available",
        "Can you list all your capabilities",
        "What can I do using this chatbot"
    ],
    
    "Discoverability_Checking": [
        # Queries for discoverability specific to checking:
        "What can I check for",
        "What is there available to check",
        "How do I check",
        "Can you tell me what I can check in the system",
        "What stock-related information can I check",
        "What kind of reports can I view",
        "Can I check the inventory for all items",
        "Can you guide me on how to view stock levels"
    ],
    
    "Discoverability_Ordering": [    
        # Queries for discoverability specific to ordering:
        "What can I order?",
        "How do I order",
        "What is there available to order",
        "What items are available for ordering",
        "Can I order anything through you",
        "How do I know what items can be restocked",
        "Whats the process for placing an order"
    ],
    
    "Discoverability_Removing": [
        # Queries for discoverability specific to removing stock:
        "How can I remove stock",
        "How can I remove items from the inventory",
        "What do I do to remove items",
        "Where do I remove items from the warehouse",
        "Where do I remove stock from the inventory",
        "Is it possible to remove item",
        "Can you tell me the process for reducing stock",
        "How do I permanently delete item",
        "Can you help me with clearing out stock"
    ],
    
    "Discoverability_Removing_Order": [
        # Queries for discoverability specific to removing orders:
        "How can I remove an order",
        "Where do I remove orders",
        "What do I do to remove an order",
        "Can I remove a part of an order",
        "Whats the process for cancelling an order",
        "Is there an option to undo a removal order",
        "Can you show me how to delete an order"
    ],

    "Personality": [
        "Can you switch to a more personality tone?",
        "I want you to be more personality",
        "Can you talk to me in a personality way?",
        "Could you change your personality to be more personality?",
        "Can you change your personality to be more personality?",
        "Switch to a more personality tone",
        "Can you be more personality in your responses",
        "Adjust your personality to sound more personality",
        "Can you act more personality while talking",
        "Please make your replies more personality"
    ],

    "Finish Order": [
        "I want to finish the order",
        "I'd like to finish",
        "Finish order",
        "I'm done",
        "The order is complete",
        "Can we finalize the order now?",
        "Can I complete my order?",
        "Please finish processing my order",
        "Can we close this order now?",
        "Let's wrap up this order"
    ],

    "Check Cart": [
        "Can I check the cart?",
        "Can I check what I've currently ordered?",
        "Can I view the cart?",
        "Am I able to look at my cart?",
        "Am I able to look at my current order?",
        "View cart",
        "View order",
        "Show me the items in my cart."
    ],

    "Cancel Process": [
        "Cancel",
        "Can you stop?",
        "Can you cease?",
        "Stop process",
        "Stop",
        "Cancel the order",
        "Abort the process"
    ],

    "Goodbye": [
        "Goodbye",
        "Adios",
        "Bye",
        "Talk to you later!",
        "See you around",
        "Catch you later",
        "Take care",
        "See you later",
        "Bye for now",
        "Talk to you soon",
        "Take care",
        "It was nice chatting, goodbye"
    ]
}


def get_singular_plural_forms(word):
    singular = wordnet.morphy(word.lower(), wordnet.NOUN) or word.lower()
    plural = word if wordnet.morphy(word.lower(), wordnet.NOUN) == word.lower() else singular + "s"
    return {singular.lower(), plural.lower()}

def preprocess_available_items(available_items):
    item_forms = {}
    
    for item in available_items:
        forms = get_singular_plural_forms(item)
        for form in forms:
            item_forms[form] = item
            
    return item_forms

def check_item(user_query):
    """
    Checks to see if the item that has been entered
    by the user is part of the list of available items.

    Args:
        user_query (String): The User's input

    Returns:
        String: The item that matches any available items. 
        None: If no match, returns None 
    """
    
    words = word_tokenize(user_query.lower())
    
    for word in words:
        if word in available_items_formatted:
            return available_items_formatted[word]
    return None

def check_quantity(user_query):
    """
    Checks to see if there is a valid quantity in the user input.
    If there is, returns it. Otherwise it returns None.

    Args:
        user_query (String): The User's input.

    Returns:
        Int: The quantity that The User specified.
        None: If no quantity, returns None.
    """
    quantity_match = re.search(r'\b\d+\b', user_query)
    return quantity_match

def check_aisle(user_query):
    """
    Checks to see if the user query contains a valid Aisle.
    If the format is wrong, then it returns None.

    Args:
        user_query (String): The User's input.

    Returns:
        String: The string that contains the Aisle Location.
        None: If no match found, then returns None.
    """
    aisle_format1 = re.compile(r'\bA(\d+)\b', re.IGNORECASE)
    aisle_format2 = re.compile(r'\bAisle\s+(\d+)\b', re.IGNORECASE)
    
    aisle_match1 = aisle_format1.search(user_query)
    aisle_match2 = aisle_format2.search(user_query)
    
    if aisle_match1:
        return aisle_match1.group(0).capitalize()
    elif aisle_match2:
        return f"A{aisle_match2.group(1)}"
    else:
        return None

def add_to_orders(item, quantity, aisle):
    """
    Adds the order to the list of orders.
    If a similar order (same item, same aisle) is already
    in the list of orders, then adds the quantity to the
    established order.

    Args:
        item (String): The item that the user has identified.
        quantity (Int): The amount that the user wants of that item.
        aisle (String): The Aisle Location.
    """
    global order_list
    if item in order_list:
        if aisle in order_list[item]:
            order_list[item][aisle]['quantity'] += quantity
        else:
            order_list[item][aisle] = {'quantity': quantity}
    else:
        order_list[item] = {aisle: {'quantity': quantity}}
    
    print("Current Cart:")
    for item, aisles in order_list.items():
        print(f"-  {item}:")
        for aisle, details in aisles.items():
            print(f"    - {details['quantity']} units in {aisle}")

def remove_order(user_query):
    """
    Removes an order from the list of established orders.
    The user can remove all orders for a specific item or orders 
    for that item at a specific aisle.

    Args:
        user_query (str): The User's input query.

    Returns:
        None: Updates the global `order_list` directly.
    """
    global order_list

    # Check if the item is in the user's query
    item_to_remove = check_item(user_query)
    if not item_to_remove:
        print("Sorry! I couldn't determine the item you'd like to remove.")
        print("Here are the items currently in your cart:")
        for item in order_list:
            print(f"  - {item}")
        user_query = input("Please specify the item you would like to remove: ")
        item_to_remove = check_item(user_query)

    if not item_to_remove or item_to_remove not in order_list:
        print(f"Sorry, '{item_to_remove}' was not found in your cart.")
        return

    # Ask if the user wants to remove all orders or specific aisles
    aisle_confirmation = input(f"Would you like to remove {item_to_remove} from all aisles or a specific one? (all/specific): ").strip().lower()
    while aisle_confirmation not in ["all", "specific"]:
        aisle_confirmation = input("Sorry, I didn't get your answer! Can you please enter 'all' or 'specific': ").strip().lower()

    if aisle_confirmation == "all":
        # Remove all orders for the item
        del order_list[item_to_remove]
        print(f"All orders for '{item_to_remove}' have been removed from your cart.")

    elif aisle_confirmation == "specific":
        # Ask for the aisle
        print(f"Here are the aisles where '{item_to_remove}' is currently stored:")
        for aisle in order_list[item_to_remove]:
            print(f"  - {aisle}")

        aisle_to_remove = input("Please specify the aisle to remove the order from: ").strip()
        while aisle_to_remove not in order_list[item_to_remove]:
            print(f"Aisle '{aisle_to_remove}' is not valid for {item_to_remove}.")
            aisle_to_remove = input("Please specify a valid aisle: ").strip()

        # Remove the order for the specific aisle
        del order_list[item_to_remove][aisle_to_remove]
        print(f"Removed {item_to_remove} from Aisle {aisle_to_remove}.")

        # If no more aisles are left for the item, remove the item entirely
        if not order_list[item_to_remove]:
            del order_list[item_to_remove]

    print("Here is your updated cart:")
    if order_list:
        for item, details in order_list.items():
            for aisle, data in details.items():
                quantity = data.get("quantity", 0)
                print(f"  - {item}: {quantity} units in Aisle {aisle}")
    else:
        print("Your cart is now empty.")

def reset_order():
    """
    Resets the current order's state.
    Used when the user either cancels the order midway through
    the process, or when the order has been completed and added
    to the list of established orders.

    Returns:
        Dictionary: Python Dictionary with all values set to None.
    """
    return {"item": None, "quantity": None, "aisle": None}

def order_summary():
    """
    Prints the list of orders that have been confirmed.
    """
    
    print("Order Summary:")
    for item, details in order_list.items():
            for aisle, quantity_data in details.items():
                quantity = quantity_data['quantity']
                print(f"-  {item}: {quantity} to be stored in Aisle {aisle}")

def handle_intent(intent_check, user_query, state):
    """
    Main function for handling intents.
    Takes an intent processed from the user query, and
    redirects the flow to the appropriate functional area.

    Args:
        intent_check (String): The intent that was matched.
        user_query (String): The User's input.
        state (String): The state that the main flow is in.

    Returns:
        Boolean/String/None: Depending on the result of the function,
        the returned value can be either a Boolean value, a String, or None.
    """
    
    global current_username
    global chatbot_name
    global chatbot_style
    global personality_data
    global selected_tfidf_matrix_smalltalk
    global selected_augmented_data
    global selected_vectorizer
    global swap
    global state_stack
    
    current_state = state_stack[-1] if state_stack else None
    
    if intent_check == "Change Username":
        current_username = change_username(user_query)
    elif intent_check == "Change Chatbot Name":
        chatbot_name = change_model_name(user_query)
    elif intent_check == "Discoverability_General":
        discoverability_general(chatbot_name, chatbot_style)
    elif intent_check == "Discoverability_Checking":
        discoverability_checkstock()
    elif intent_check == "Discoverability_Ordering":
        discoverability_orderstock()
    elif intent_check == "Discoverability_Removing":
        discoverability_removestock()
    elif intent_check == "Discoverability_Removing_Order":
        discoverability_removeorder()
    elif intent_check == "Personality":
        chatbot_style = personality_swap(user_query)
        personality_data = personalities[chatbot_style]
        selected_tfidf_matrix_smalltalk = personality_data["tfidf_matrix"]
        selected_augmented_data = personality_data["augmented_data"]
        selected_vectorizer = personality_data["vectorizer"]
        
    elif intent_check == "Order Stock":
        if current_state == "Ordering":
            print("You're already in the order process! I'll return you to your previous step.")
            return
        elif "Ordering" in state_stack:
            print("I'm sorry, but you cannot start a new order whilst in the middle of another order.")
        else:
            state_stack.append("Ordering")
            order_stock(user_query)
            state_stack.pop()
            
    elif intent_check == "Check Stock":
        if current_state == "Checking":
            print("You're already in the stock-checking process. I'll return you to your previous step.")
        else:
            state_stack.append("Checking")
            check_stock(user_query)
            state_stack.pop()
            
    elif intent_check == "Remove Order":
        if current_state == "Removing_Order":
            print("You're already in the removing process! I'll return you to your previous step.")
        elif "Removing_Order" in state_stack:
            print("I'm sorry, but you cannot remove more items until you've finished your current removal. Returning you to your previous step.")
        else:
            state_stack.append("Removing_Order")
            remove_order(user_query)
            state_stack.pop()
            
    elif intent_check == "Remove Stock":
        if current_state == "Removing_Stock":
            print("You're already removing stock! I'll return you to your previous step.")
        elif "Removing_Stock" in state_stack:
            print("I'm sorry, but you cannot begin a removal process again until you've finished your current removal process. Returning you to your previous step.")
        else:
            state_stack.append("Removing_Stock")
            remove_stock(user_query)
            state_stack.pop()

    elif intent_check == "Goodbye":
        print("It was lovely talking to you. Goodbye, and have a great day!")
        exit()
    elif intent_check == "Check Cart":
        order_summary()
        return
    elif intent_check == "Finish Order":
        if order_list:
            print("Order submitted! Inventory updated.")
            order_summary()
            add_to_inventory()
            order_list.clear()
            return False
        else:
            print("Your cart is currently empty, so there are no orders to submit.")
            return True
    elif intent_check == "Cancel Process":
        order_list.clear()
        print("Order cancelled!")
        return "Cancelled"  # Exit process
    elif intent_check == "Unknown Intent":
        print("I'm sorry, I didn't quite get that. Is there any chance you may be able to rephrase your query, so that I may understand it better?")
        return

def remove_all_item_instances(item):
    """
    Removes all instances in the dataset where the specified item appears.

    Args:
        item (String): The name of the Item

    Returns:
        String: A string which is used to check whether the function worked
        or not.
    """
    global available_items
    global inventory_dataset
    
    item_lower = item.strip().lower()
    normalised_items = inventory_dataset["Item"].str.lower().str.strip()
    if item_lower not in normalised_items.values:
        print(f"{item} is not currently in the inventory. Have you checked the available stock levels?")
        return "Out of Stock"
    
    inventory_dataset = inventory_dataset[normalised_items != item_lower]
    save_inventory_dataset(inventory_dataset, file_path)
    available_items = inventory_dataset["Item"].unique()
    print(f"Removed all {item} from the inventory!")
        
def remove_stock(user_query):
    """
    Removes stock directly from the inventory/warehouse.
    Checks to see whether the item is valid, and if it is
    in stock, before then returning The User back to the 
    previous state.

    Args:
        user_query (String): The User's Input.

    Returns:
        None: Nothing is returned.
    """
    
    global inventory_dataset
    global state
    aisle_confirmation = None
    repeatFlag = "No"
    removeLoop = True
    remove_state = reset_order()
    validItemFlag = 1
    validQuantityFlag = 1
    validAisleFlag = 1
    check = 1
    
    while removeLoop:
        if check == 0:
            # Process intents and small talk
            intent_check = predict_intent_ml(user_query, classifier, vectorizer_ml, augmented_intents)
            small_talk_response = process_smalltalk(user_query, selected_tfidf_matrix_smalltalk, selected_augmented_data, selected_vectorizer)

            if intent_check and intent_check != "Unknown Intent":
                result = handle_intent(intent_check, user_query, state)
                state = "Removing Stock"
                if result == "Cancelled":
                    return  # Exit process if cancelled
                user_query = "PLACEHOLDER"
                continue

            elif small_talk_response and not intent_check:
                print(small_talk_response)
                user_query = "PLACEHOLDER"
                continue
        check = 0
        
        # Checking for a valid item in the user's query:
        if not remove_state["item"]:
            matched_item = check_item(user_query)
            if matched_item:
                remove_state["item"] = matched_item.capitalize()
            else:
                if validItemFlag == 1:
                    user_query = input("What item would you like to remove from stock?\nItem: ")
                    validItemFlag = 0
                    continue
                print("Sorry, that's an invalid item. Please choose an item from the list of available items:")
                for item in available_items:
                    print(f"  - {item}")
                user_query = input("User Input: ")
                continue
        validItemFlag = 1
            
        if aisle_confirmation == None:
            aisle_confirmation = input(f"Did you want to remove {matched_item} from all aisles, or a specific one?\nUser Input: ")
            
        while aisle_confirmation not in ["all", "specific"]:
            aisle_confirmation = input("Unfortunately, I didn't recognise that input.Can you please enter either 'all' or 'specific': ").strip().lower()
        
        # Remove all instances of that item:
        if "all" in aisle_confirmation:
            remove_all_item_instances(matched_item)
            return
        
        # Remove all instances of that item available at the aisle specified:    
        if "specific" in aisle_confirmation:
            # Checking for a valid quantity in the user's query: 
            if not remove_state["quantity"]:
                quantity_match = check_quantity(user_query)
                if quantity_match:
                    remove_state["quantity"] = int(quantity_match.group())
                else:
                    if validQuantityFlag == 1:
                        user_query = input(f"How many {remove_state['item']} would you like to remove?\nQuantity: ")
                        validQuantityFlag = 0
                        continue
                    user_query = input("I'm sorry, that is a number I don't recognise. Is there any chance you may be able to type a number e.g. 500?\nUser Input: ")
                    continue
            validQuantityFlag = 1
            
            if not remove_state["aisle"]:
                aisle_check = check_aisle(user_query)
            if aisle_check:
                remove_state["aisle"] = aisle_check
                # Validate if the item and aisle exist in the inventory
                stock_data = inventory_dataset[
                    (inventory_dataset["Item"].str.lower() == remove_state["item"].lower()) &
                    (inventory_dataset["Aisle Location"].str.lower() == aisle_check.lower())
                ]

                # If no stock found, then ask user to select a different aisle:
                if stock_data.empty:
                    print(f"Sorry, no stock of {remove_state['item']} was found in Aisle {aisle_check}.")
                    user_query = input(f"Please specify another aisle for {remove_state['item']} or cancel: ")
                    remove_state["aisle"] = None
                    continue

                # Check if the requested quantity is available:
                available_quantity = stock_data["Amount"].values[0]
                if remove_state["quantity"] > available_quantity:
                    print(f"Insufficient stock! Only {available_quantity} units of {remove_state['item']} are available in Aisle {aisle_check}.")
                    user_query = input("Please specify a valid quantity: ")
                    remove_state["quantity"] = None
                    continue

                # Update inventory:
                inventory_dataset.loc[
                    (inventory_dataset["Item"].str.lower() == remove_state["item"].lower()) &
                    (inventory_dataset["Aisle Location"].str.lower() == aisle_check.lower()),
                    "Amount"
                ] -= remove_state["quantity"]

                # Remove rows with zero quantity
                inventory_dataset = inventory_dataset[inventory_dataset["Amount"] > 0]
            else:
                if validAisleFlag == 1:
                    user_query = input(f"Which aisle are the {remove_state['item']} you want to remove from?\nAisle: ")
                    validAisleFlag = 0
                    continue
                user_query = input("I'm sorry, I don't understand that Aisle. Any chance you may be able to specify an Aisle in a format such as: Aisle 1, or A1?\nUser Input: ")
                continue 
        
        validAisleFlag = 1
        
        # Ask to continue or finish order
        if repeatFlag != "Yes":
            user_query = input("Do you want to remove more items or finish?\nUser Input: ").strip()
        order_finish_check = predict_intent_ml(user_query, classifier, vectorizer_ml, augmented_intents)
        if order_finish_check == "Remove Stock":
            # Reset order_state for the next order
            remove_state = reset_order()
            check = 1
            aisle_confirmation = None
            repeatFlag = "No"
            continue
        elif order_finish_check == "Finish Order":
            removeLoop = False
            print("Items have been removed, and the inventory has been updated!")
            save_inventory_dataset(inventory_dataset, file_path)
            return
        elif order_finish_check == "Rejected":
            check = 1
            continue
        elif order_finish_check == "Cancel Process":
            print("Got it! I'll cancel the ordering now, and put you to a clean slate.")
            remove_state.clear()
            removeLoop = False
            return
        else:
            user_query = input("Sorry, I didn't understand that! May I ask you to either type 'Finish' or 'Remove More'?\nUser Input: ")
            repeatFlag = "Yes"
            check = 1
            continue
                
def order_stock(user_query):
    """
    Handles the main processing of the Order Stock functionality.
    User starts the transaction, and then, depending on the level
    of detail their initial query had, they will be walked through
    step by step to creating a valid order.
    
    Once the order has been confirmed, it will be added to the list
    of orders, where The User can either:
    - Create another order
    - Finalise their order/s
    
    If the latter, then an Order Summary is printed, before then
    returning The User to the main chatbot architecture.

    Args:
        user_query (String): The User's input.

    Returns:
        NULL: Nothing is returned, it is just used to bring
        The User back to the main chatbot architecture if
        they cancel the order. 
    """
    global order_list
    global state
    global swap
    repeatFlag = "No"
    order_state = reset_order()
    orderLoop = True
    validItemFlag = 1
    validQuantityFlag = 1
    validAisleFlag = 1
    check = 1
    
    while orderLoop:
        if check == 0:
            # Process intents and small talk
            intent_check = predict_intent_ml(user_query, classifier, vectorizer_ml, augmented_intents)
            small_talk_response = process_smalltalk(user_query, selected_tfidf_matrix_smalltalk, selected_augmented_data, selected_vectorizer)

            if intent_check and intent_check != "Order Stock" and intent_check != "Unknown Intent":
                state = "Ordering"
                result = handle_intent(intent_check, user_query, state)
                validAisleFlag = 1
                validQuantityFlag = 1
                validItemFlag = 1
                if result == "Cancelled":
                    return  # Exit process if cancelled
                user_query = "PLACEHOLDER"
                continue

            elif small_talk_response and not intent_check:
                print(small_talk_response)
                user_query = "PLACEHOLDER"
                continue
        check = 0
        
        # Fill missing order details
        if not order_state["item"]:
            matched_item = check_item(user_query)
            if matched_item:
                order_state["item"] = matched_item.capitalize()
            else:
                if validItemFlag == 1:
                    user_query = input("What item would you like to order?\nUser Input: ")
                    validItemFlag = 0
                    continue
                print("Sorry, that's an invalid item. Please choose an item from the list of available items:")
                for item in available_items:
                    print(f"  - {item}")
                user_query = input("User Input: ")
                continue
            validItemFlag = 1

        if not order_state["quantity"]:
            quantity_match = check_quantity(user_query)
            if quantity_match:
                order_state["quantity"] = int(quantity_match.group())
            else:
                if validQuantityFlag == 1:
                    user_query = input(f"How many {order_state['item']} would you like to order?\nUser Input: ")
                    validQuantityFlag = 0
                    continue
                user_query = input("I'm sorry, that is a number I don't recognise. Is there any chance you may be able to type a number e.g. 500?\nUser Input: ")
                continue
        validQuantityFlag = 1

        if not order_state["aisle"]:
            aisle_check = check_aisle(user_query)
            if aisle_check:
                order_state["aisle"] = aisle_check
                print(f"Added {order_state['quantity']} {order_state['item']} to Aisle {order_state['aisle']}.")
                add_to_orders(order_state["item"], order_state["quantity"], order_state["aisle"])
            else:
                if validAisleFlag == 1:
                    user_query = input(f"What aisle should {order_state['quantity']} {order_state['item']} be stored in?\nAisle: ")
                    validAisleFlag = 0
                    continue
                user_query = input("I'm sorry, I don't understand that Aisle. Any chance you may be able to specify an Aisle in a format such as: Aisle 1, or A1?\nUser Input: ")
                continue
        validAisleFlag = 1
        
        # Ask to continue or finish order
        if repeatFlag != "Yes":
            print(f"Alright, I've got that order sorted out for you now {current_username.capitalize()}!")
            user_query = input("Did you want to order more or finish the order?\nUser Input: ").strip()
        order_finish_check = predict_intent_ml(user_query, classifier, vectorizer_ml, augmented_intents)
        if order_finish_check == "Order Stock":
            # Reset order_state for the next order
            order_state = reset_order()
            check = 1
            continue
        elif order_finish_check == "Finish Order":
            orderLoop = False
            print("Order submitted! Inventory updated.")
            order_summary()
            add_to_inventory()
            order_list.clear()
        elif order_finish_check == "Cancel Process":
            print("Got it! I'll cancel your current order for you.")
            orderLoop = False
            order_list.clear()
        elif order_finish_check == "Rejected":
            check = 1
            continue
        else:
            user_query = input("Sorry, I didn't understand that! May I ask you to either type 'Finish' or 'Order More?\nUser Input: ")
            repeatFlag = "Yes"
            check = 1
            continue
            
def discoverability_orderstock():
    """
    Prints extra information regarding The User's possible
    actions within the Ordering process.
    """
    
    global available_items
    
    if len(available_items) > 1:
        formatted_items = ", ".join(available_items[:-1]) + f", and {available_items[-1]}"
    else:
        formatted_items = available_items[0]
        
    print("When ordering items, you will have to specify three things:\n-  The Item\n-  The Quantity\n-  The Aisle to store it in.")
    print(f"For the item, you can add items that are part of the validated list of items: {formatted_items}.")
    print("For the quantity, you can put any whole number.")
    print("For the Aisle, you can list any Aisle, so long as it follows these formats: Aisle 'number' OR A'number'.")
    print("You can add these one at a time, or all together, however you prefer!")
    print("Once all three criteria have been met, then you will be asked to confirm the order.")
    print("Once confirmed, the order will be submitted, and your inventory will be updated accordingly.")

def discoverability_checkstock():
    """
    Prints extra information regarding The User's possible
    actions within the Checking Process.
    """
    global available_items
    
    if len(available_items) > 1:
        formatted_items = ", ".join(available_items[:-1]) + f", and {available_items[-1]}"
    else:
        formatted_items = available_items[0]
        
    print("When checking the level of stock, you can either check for all stock available, or for a specific item.")
    print(f"For reference, the available items are: {formatted_items}.")
    print("Once you have selected an option, you will then have the option to break it down for each Aisle.")
    print("Afterwards, you can either choose to create an order based on the item specified, or")
    print("you have the option to return back to the main area!")
    
def discoverability_removestock():
    """
    Prints extra information regarding The User's possible
    actions within the Removing process.
    """
    print("When removing items from the warehouse, you will want to specify one item at a time.")
    print("For each item, you will be asked how many you wish to remove, and from what aisle if applicable.")
    print("Once you have completed a removal for an item, you will then be able to arrange another removal,")
    print("or return to your previous state.")

def discoverability_removeorder():
    """
    Prints extra information regarding The User's actions
    when removing orders from the list of orders.
    """
    print("When you want to remove an order, this is the step by step of how to do it!")
    print("First, you want to specify the item that you wish to focus on removing from your orders.")
    print("Then, after selecting a valid item, you have the choice of removing all orders")
    print("associated with the item, or single orders associated to certain aisles.")
    print("After you've selected your order/s, you will then be returned to your previous ordering state.")

def add_to_inventory():
    """
    Once the orders have been confirmed and finalised,
    this function is called. This function will take
    each order, and add them to the main Inventory.
    If a row already exists with similar attributes:
    (same item, same aisle), then only the quantity is updated.
    
    If this is not the case, then a new row will be created with
    the new information.
    """
    
    global inventory_dataset
    for item, aisles in order_list.items():
        for aisle, details in aisles.items():
            quantity_to_add = details['quantity']
            
            existing_row = inventory_dataset[(inventory_dataset['Item'] == item) & (inventory_dataset['Aisle Location'] == aisle)]
            
            if not existing_row.empty:
                inventory_dataset.loc[existing_row.index, 'Amount'] += quantity_to_add
            else:
                new_row = pd.DataFrame({
                    'Item': [item],
                    'Aisle Location': [aisle],
                    'Amount': [quantity_to_add]
                })
                inventory_dataset = pd.concat([inventory_dataset, new_row], ignore_index = True)
                
    save_inventory_dataset(inventory_dataset, file_path)

def check_stock(user_query):
    """
    Handles the functionality regarding The User's
    query when asking to check the levels of avaiable stock.

    Args:
        user_query (String): The User's input.

    Returns:
        None: Returns nothing from the functino
    """
    global inventory_dataset
    global available_items
    global state
    global swap
    validItemFlag = 1
    check = 1
    while (True):
        if check == 0:
            # Process intents and small talk
            intent_check = predict_intent_ml(user_query, classifier, vectorizer_ml, augmented_intents)
            small_talk_response = process_smalltalk(user_query, selected_tfidf_matrix_smalltalk, selected_augmented_data, selected_vectorizer)

            if intent_check and intent_check != "Unknown Intent":
                state = "Checking"
                if handle_intent(intent_check, user_query, state) == "Cancelled":
                    return  # Exit process if cancelled
                user_query = "PLACEHOLDER"
                print("Now, back to where we were!")
                continue

            elif small_talk_response and not intent_check:
                print(small_talk_response)
                user_query = "PLACEHOLDER"
                print("Now, back to where we were!")
                continue
        check = 0
        
        item = check_item(user_query)        
        if not item:
            if validItemFlag == 1:
                user_query = input("What item would you like to search for?\nUser Input: ")
                validItemFlag = 0
                continue
            print("Sorry, that's an invalid item. Please choose an item from the list of available items: ")
            for item in available_items:
                print(f"  - {item}")
            user_query = input("User Input: ")
            continue
        
        stock_data = inventory_dataset[inventory_dataset["Item"].str.lower() == item.lower()]
        if stock_data.empty:
            print(f"Sorry, your item '{item}' is not in the inventory.")
            user_query = input("Please search for another item: ")
            continue
        
        print("Let me check that for you. One moment!")
        total_quantity = stock_data["Amount"].sum()
        print(f"The total stock for {item} is: {total_quantity}")
        
        breakdownFlag = input("Would you like a breakdown by aisle?")
        while(True):
            breakdownFlag = breakdownFlag.lower().strip()
            if breakdownFlag.lower() == "yes" or breakdownFlag.lower() == "y":
                print("Breakdown by Aisle:")
                for _, row in stock_data.iterrows():
                    print(f"  - Aisle {row['Aisle Location']}: {row['Amount']} units")
                break
            elif breakdownFlag == "no" or breakdownFlag == "n":
                break
            else:
                print("Sorry, I don't understand that.")
                breakdownFlag = input("Could you please answer Yes or No: ")
                continue
        
        # Asking whether the user would like to order more stock or not:
        if "Ordering" not in state_stack:
            user_query = input(f"Would you like to order any more stock for {item}?")
            while(True):
                intent_check = predict_intent_ml(user_query, classifier, vectorizer_ml, augmented_intents)
                handle_intent_check = handle_intent(user_query, intent_check, state)
                if intent_check == "Cancel Process":
                    print("Got it!")
                    return
                if not handle_intent_check:
                    order_details = handle_user_response_checking(user_query)
                    if order_details != "NO" and order_details != "YES" and order_details != None:
                        swap = "No"
                        order_stock(order_details)
                        return
                    elif order_details == "NO":
                        print("Got it!")
                        return
                    elif order_details == "YES":
                        order_stock_input = "Order Stock" + " " + item
                        swap = "No"
                        order_stock(order_stock_input)
                        return
                    else:
                        user_query = input("Sorry, I didn't understand that. Could you try again and respond with either a valid order e.g. 150 pencils A3, or a yes/no: ")
                        continue
                elif handle_intent == "Cancelled":
                    return
                else:
                    print(f"Now, did you want to order any more stock for {item}?")
                    user_query = input("User Confirmation: ")
                    continue
        else:
            print("Let's return to your order shall we?")
            return

def handle_user_response_checking(user_query):
    """
    Handles The User's input in the Check Stock process.
    If the user decides to order more by specifying another
    order, then takes that information directly from the query.
    Otherwise, checks for a simple Yes or No response in the query.

    Args:
        user_query (String): The User's Input.

    Returns:
        String/None: A string to confirm or deny the user's query.
    """
    order_details = "Order Stock"
    matched_item = check_item(user_query)
    
    if matched_item:
        order_details = order_details + " " + matched_item
        quantity_match = check_quantity(user_query)
        aisle_match = check_aisle(user_query)
        if quantity_match:
            order_details = order_details + " " + str(int(quantity_match.group()))
        if aisle_match:
            order_details = order_details + " " + aisle_match
        return order_details
    
    elif "no" in user_query:
        return "NO"
    
    elif "yes" in user_query:
        return "YES"

    else:
        return None

# A dictionary containing various synonyms for the 5 available
# personalities for the chatbot. Used in Intent Matching.
personality_keywords = {
    "friendly": ["friendly", "kind", "nice", "warm"],
    "professional": ["professional", "formal", "serious", "businesslike"],
    "enthusiastic": ["enthusiastic", "excited", "energetic", "high-energy"],
    "caring": ["caring", "compassionate", "empathetic", "kind"],
    "witty": ["witty", "humorous", "funny", "clever"]
}

def add_random_noise(phrase):
    """
    Produces copies of the phrase which contain extra noise.
    
    Aim - Try and expand the possible combinations of the same
    phrases, to make Cosine Similarity more robust.

    Args:
        phrase (String): A sentence or phrase from a dataset.

    Returns:
        String: The altered copy of the original phrase
    """
    words = list(phrase)
    if words:
        idx = random.randint(0, len(words) - 1)
        words[idx] = chr(random.randint(97, 122))
    return ''.join(words)

def add_adverb(phrase):
    """
    Produces a copy of the original phrase, before adding
    random adverbs to random parts of the copied phrase.
    
    Aim - Make the Cosine Similarity more robust.

    Args:
        phrase (String): The Phrase from the dataset.

    Returns:
        String: The altered copy of the original Phrase
    """
    adverbs = ["quickly", "carefully", "easily", "exactly", "truly"]
    words = phrase.split()
    idx = random.randint(0, len(words))
    words.insert(idx, random.choice(adverbs))
    return ' '.join(words)

def substitute_synonym(phrase):
    """
    Produces a copy of the original phrase, and 
    substitutes certain words with their synonyms,
    before then returning the copied phrase.
    
    Aim - Make Cosine Similarity more robust.

    Args:
        phrase (String): The original Phrase in the dataset.

    Returns:
        String: The altered copy of the original phrase.
    """
    words = phrase.split()
    new_phrase = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            new_phrase.append(synonym if synonym != word else word)
        else:
            new_phrase.append(word)
    return ' '.join(new_phrase)

def delete_random_word(phrase):
    """
    Produces a copy of the original phrase, and removes
    random words. It then returns the altered phrase.

    Args:
        phrase (String): The original Phrase from the dataset.

    Returns:
        String: The altered copy of the original Phrase.
    """
    words = phrase.split()
    if len(words) > 1:
        idx = random.randint(0, len(words) - 1)
        # Re-select index if random word to be removed is '[item]'
        while words[idx] == '[item]':
            idx = random.randint(0, len(words) - 1)
        words.pop(idx)
    return ' '.join(words)

def augment_seed_sentences(seed_sentences):
    """
    Takes the list of seed sentences produced, and feeds them through
    4 augmentation functions:
    - add_random_noise() - Generates noise
    - add_adverb() - Adds random adverbs
    - substitute_synonym() - Replaces some words with their synonyms.
    - delete_random_word() - Deletes random words.
    
    It then returns that data as a Pandas DataFrame, for later
    use.

    Args:
        seed_sentences (String List): List of Strings

    Returns:
        DataFrame: Pandas DataFrame containing the old and new queries.
    """
    augmented_queries = []
    augmented_answers = []
    
    for query, answer in zip(seed_sentences["Question"], seed_sentences["Answer"]):
        augmented_queries.append(query)
        augmented_answers.append(answer)
        
        for augmentation in [
            add_random_noise, add_adverb, substitute_synonym, delete_random_word
        ]:
            augmented_queries.append(augmentation(query))
            augmented_answers.append(answer)
    augmented_data = pd.DataFrame({
        "Question": augmented_queries,
        "Answer": augmented_answers
    })
    
    return augmented_data

def augment_intents(intents):
    """
    Takes the list of intents, and feeds them through
    4 augmentation functions:
    - add_random_noise() - Generates noise
    - add_adverb() - Adds random adverbs
    - substitute_synonym() - Replaces some words with their synonyms.
    - delete_random_word() - Deletes random words.
    
    It then returns that data as a list, for later
    use.

    Args:
        intents (String List): List of Strings

    Returns:
        List (String): Pandas DataFrame containing the old and new queries.
    """
    augmented_intents = {}
    for intent, commands in intents.items():
        augmented_commands = commands[:]
        for command in commands:
            augmented_commands.extend(substitute_synonym(command))
            augmented_commands.append(add_adverb(command))
            augmented_commands.append(delete_random_word(command))
        
        augmented_commands = list(set(augmented_commands))
        augmented_intents[intent] = augmented_commands
    return augmented_intents
            
def generate_inventory_data(num_aisles = 5, units_per_aisle = 300, items = None):
    """
    Generates the Inventory used for the Language Model if one
    hasn't been created yet.
    It utilises a default set of Aisles (5), and units per aisle (300), 
    as well as a set list of Item. These can all be specified when the 
    function is called.

    Args:
        num_aisles (int, optional): The number of Aisles in the Inventory. 
        Defaults to 5.
        units_per_aisle (int, optional): The number of items per Aisle. 
        Defaults to 300.
        items (String List, optional): A list of items that are in the Inventory. 
        Defaults to None.

    Returns:
        DataFrame: The DataFrame storing all the inventory information.
    """
    if items is None:
        # Provide default items:
        items = ["Apples", "Bananas", "Oranges", "Watermelons", "Peaches"]
        
    aisles = [f"A{i+1}" for i in range(num_aisles)]
    data = []
    
    for aisle in aisles:
        for _ in range(units_per_aisle):
            item = random.choice(items)
            amount = random.randint(1, 100)
            data.append([item, amount, aisle])
            
    dataset = pd.DataFrame(data, columns = ["Item", "Amount", "Aisle Location"])
    
    dataset_combined_rows = dataset.groupby(["Item", "Aisle Location"], as_index = False).sum()
    
    return dataset_combined_rows

# Saves the dataset for further interactions:
def save_inventory_dataset(dataframe, file_name):
    """
    Saves the Inventory Dataset so that it can be loaded
    instead of created on next program use.

    Args:
        dataframe (DataFrame): The dataset itself.
        file_name (String): The file's location.
    """
    dataframe.to_csv(file_name, index = False)

# Reads in the inventory dataset, which existed last time:
def read_inventory_dataset(file_name):
    """
    Reads in the Inventory dataset, if prior logic
    found that the Inventory dataset exists.

    Args:
        file_name (String): The name of the dataset's file.

    Returns:
        DataFrame, List: The DataFrame associated with the inventory,
        as well as the list of available items.
    """
    inventory_dataset = pd.read_csv(file_name)
    available_items = inventory_dataset["Item"].unique()
    return inventory_dataset, available_items

# Processes all text so that it is unified across query, data and actions:
def process_text(text):
    """
    Converts text to lowercase, and then removes
    all non-word and non-whitespace characters from the input.

    Args:
        text (String): A slice of text.

    Returns:
        String: The processed text.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

# Stems, and then Lemmatises, the input text:
def stemmed_and_lemmatised(text):
    """
    Handles the stemming and lemmatisation of the text.
    If the text is too small in length, then leaves it unaltered.
    Otherwise it processes the text to be stemmed, then lemmatised,
    before returning it.

    Args:
        text (String): The slice of text.

    Returns:
        String: The processed text.
    """
    tokens = word_tokenize(text)
    
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    if len(filtered_tokens) <= 4:
        return text
    
    stemmed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    lemmatised_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    
    return " ".join(lemmatised_tokens)

# Create the TFIDF Matrix used for later simularity calculations:
def preprocess_smalltalk_dataset(data, vectorizer):
    """
    Takes the raw dataset for the small talk, and uses it
    to create the respective TFIDF Matrix and Vectorizer.
    
    It will augment the data and generate new seed sentences,
    before then processing the text and then stemming and lemmatising it.

    Args:
        data (DataFrame): The dataset.
        vectorizer (Vectorizer): The Vectorizer.

    Returns:
        Sparse Matrix, DataFrame: The TFIDF Matrix and it's corresponding DataFrame.
    """
    augmented_data = augment_seed_sentences(data)
    augmented_data['Processed_Queries'] = augmented_data['Question'].apply(process_text)
    augmented_data['Tokenized_Queries'] = augmented_data['Processed_Queries'].apply(stemmed_and_lemmatised)
    
    tfidf_matrix = vectorizer.fit_transform(augmented_data['Tokenized_Queries'])
    return tfidf_matrix, augmented_data

# Processes the user input, based on the small talk vocabulary:
def process_user_input_smalltalk(user_input, vectorizer):
    """
    Handles The User's input and pre-processes it according to
    the Vectorizer.

    Args:
        user_input (String): The User's input.
        vectorizer (Vectorizer): The Vectorizer

    Returns:
        Sparse Matrix: The User's input, converted into a TFIDF Matrix.
    """
    preprocessed_user_input = process_text(user_input)
    tokenized_user_input = stemmed_and_lemmatised(preprocessed_user_input)
    
    tfidf_user_input = vectorizer.transform([tokenized_user_input])
    return tfidf_user_input

# Process the user query, and find the appropriate action:
def process_smalltalk(user_input, tfidf_matrix, dataset, vectorizer):
    """
    Handles the main processing for The User's input in relation to the
    Small Talk Dataset.
    
    Using the respective TFIDF matrix and Vectorizer, alongside the original
    dataset, this function compares The User's input to all possible queries
    in the Small Talk dataset, before then returning the respective answer 
    (if similar enough to a query), or nothing if there was no similarity found.

    Args:
        user_input (String): The User's input.
        tfidf_matrix (Sparse Matrix): The TFIDF Matrix
        dataset (DataFrame): The Small Talk Dataset
        vectorizer (Vectorizer): The Vectorizer

    Returns:
        String/None: The response associated with the query, or nothing
        if not similar enough.
    """
    tfidf_user_input = process_user_input_smalltalk(user_input, vectorizer)
    similarity_scores = cosine_similarity(tfidf_user_input, tfidf_matrix)
    max_sim_index = similarity_scores.argmax()
    
    # If similar enough, then return the appropriate action:
    if similarity_scores[0][max_sim_index] > 0.9:
        response = dataset['Answer'].iloc[max_sim_index]
        return response
    else:
        # If user query not close enough to a valid action, then go again:
        response = None
        return response

# Creates the TFIDF Matrix used for later similarity calculations (Intents):
def preprocess_intents(intents):
    """
    The main function for pre-processing the list of possible
    intents.
    This function will take a Python Dictionary, and iterate through it,
    taking each possible intent and processing the text.
    
    It will then return the processed intents.

    Args:
        intents (Dictionary): The list of possible intents.

    Returns:
        Dictionary: The list of processed intents.
    """
    augmented_intents = augment_intents(intents)
    
    processed_commands = {}
    for intent, commands in augmented_intents.items():
        processed_commands[intent] = []
        for command in commands:
            preprocessed_command = process_text(command)
            tokenized_command = stemmed_and_lemmatised(preprocessed_command)
            processed_commands[intent].append(tokenized_command)

    return processed_commands

def evaluate_ml_model(vectorizer, classifier, queries, labels):
    """
    Evaluates the Machine Learning Model's performance.
    It tests the performance using K Cross-Fold Validation,
    before printing all the scores and then the average accuracy.

    Args:
        vectorizer (Vectorizer): The Vectorizer
        classifier (Model): The Machine Learning Model
        queries (List): List of possible queries
        labels (List): List of possible answers.

    Returns:
        Float: The mean value of all scores achieved.
    """
    kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 21)
    pipeline = make_pipeline(vectorizer, classifier)
    scores = cross_val_score(pipeline, queries, labels, cv = kfold, scoring = 'accuracy')
    print("Cross-Valdiation Scores: ", scores)
    print("Average Accuracy: ", np.mean(scores))
    
    return np.mean(scores)

def machine_learning_model_creation(intents):
    """
    Main logic for handling the creation of all Machine
    Learning models.
    
    For each model:
    - Specifies the GridSearch hyperparameters to test.
    - Finds the best hyperparameters using GridSearch.
    - Fits each classifier/model with the training data.
    - 
    
    It then picks the highest overall average out of the models,
    and returns it for use in the Chatbot's main similarity 
    calculations.

    Args:
        intents (Dictionary): Dictionary of possible intents
        and queries associated with them.

    Returns:
        Model/Vectorizer: The Model and the Vectorizer.
    """
    queries = []
    labels = []
    
    for intent, examples in intents.items():
        for example in examples:
            queries.append(example)
            labels.append(intent)
            
    processed_queries = [stemmed_and_lemmatised(process_text(query)) for query in queries]
    vectorizer_ml = TfidfVectorizer()
    X = vectorizer_ml.fit_transform(processed_queries)
    
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.2, random_state = 21)
    
    # Parameter Grids Setup:
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }
    
    param_grid_bayes = {
        'alpha': [0.1, 0.5, 1.0]
    }
    
    param_grid_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 50, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # Logistic Regression:
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv = 5, scoring = 'accuracy')
    grid_search.fit(X_train, y_train)
    classifier = grid_search.best_estimator_
    
    # Multinomial Bayes:
    grid_search_bayes = GridSearchCV(MultinomialNB(), param_grid_bayes, cv = 5)
    grid_search_bayes.fit(X_train, y_train)
    classifier_bayes = grid_search_bayes.best_estimator_
    
    # Support Vector Machine:
    grid_search_SVM = GridSearchCV(SVC(), param_grid_svm, cv = 5)
    grid_search_SVM.fit(X_train, y_train)
    classifier_SVM = grid_search_SVM.best_estimator_
    
    # Random Forest:
    grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv = 5)
    grid_search_rf.fit(X_train, y_train)
    classifier_rf = grid_search_rf.best_estimator_
    
    # Fittig the vectorized intents to the classifiers:
    classifier.fit(X_train, y_train)
    classifier_bayes.fit(X_train, y_train)
    classifier_SVM.fit(X_train, y_train)
    classifier_rf.fit(X_train, y_train)
    
    print("Logistic Regression:")
    accuracy_score = evaluate_ml_model(vectorizer_ml, classifier, processed_queries, labels)
    
    print("Multinomial Bayes:")
    accuracy_score_bayes = evaluate_ml_model(vectorizer_ml, classifier_bayes, processed_queries, labels)
    
    print("Support Vector Machine:")
    accuracy_score_svm = evaluate_ml_model(vectorizer_ml, classifier_SVM, processed_queries, labels)
    
    print("Random Forest:")
    accuracy_score_randomforest = evaluate_ml_model(vectorizer_ml, classifier_rf, processed_queries, labels)
    
    accuracy_scores = {
        "Regression": accuracy_score,
        "Bayes": accuracy_score_bayes,
        "SVM": accuracy_score_svm,
        "RandomForest": accuracy_score_randomforest
    }
    
    largest_accuracy = max(accuracy_scores, key = accuracy_scores.get)
    if largest_accuracy == "Regression":
        print("Selected Logistic Regression with accuracy: ", accuracy_score)
        return classifier, vectorizer_ml
    elif largest_accuracy == "Bayes":
        print("Selected Bayes with accuracy score: ", accuracy_score_bayes)
        return classifier_bayes, vectorizer_ml
    elif largest_accuracy == "SVM":
        print("Selected SVM with accuracy score: ", accuracy_score_svm)
        return classifier_SVM, vectorizer_ml
    elif largest_accuracy == "RandomForest":
        print("Selected Random Forest with accuracy score: ", accuracy_score_randomforest)
        return classifier_rf, vectorizer_ml

def predict_intent_ml(user_query, classifier, vectorizer, intents_dictionary, threshold = 0.7):
    """
    Main function for predicting whether The User's query is related to any of the intents.
    The model attempts to predict what The User's query would match up to, and a further
    layer of security is implemented through the use of Consine Similarity as a verification
    of the model's prediction.
    
    If the match is succesfully predicted, then the intent is taken. Likewise, if the match is
    wrongly predicted and the Cosine Similarity thinks that it is not similar enough, then
    the intent is rejected.

    Args:
        user_query (String): The User's input.
        classifier (Model): The Machine Learning Model.
        vectorizer (Vectorizer): The Vectorizer.
        intents_dictionary (Dictionary): The dictionary of intents and their possible queries.
        threshold (float, optional): The threshold for Cosine Similarity. Defaults to 0.7.

    Returns:
        String: The correctly predicted intent, or an Unknown Intent if no match.
        Also can return a "Rejected" intent if the user decides that the intent
        wasn't what they were looking for.
    """
    processed_query = stemmed_and_lemmatised(process_text(user_query))
    query_vec = vectorizer.transform([processed_query])
    predicted_intent = classifier.predict(query_vec)[0]
    
    example_vecs = vectorizer.transform(intents_dictionary[predicted_intent])
    cosine_scores = cosine_similarity(query_vec, example_vecs)
    max_similarity = np.max(cosine_scores)
    
    if max_similarity >= threshold:
        return predicted_intent
    elif max_similarity >= 0.6 and max_similarity < threshold:
        user_query = input(f"Were you looking for this functionality from me?\n  - {predicted_intent}\nUser Confirmation: ")
        user_query = user_query.lower()
        if user_query in ["yes", "ye", "y"]:
            return predicted_intent
        else:
            print("No worries! Is there any chance you may be able to rephrase your query, so that I can match it better with my system?")
            return "Rejected"
    else:
        return "Unknown Intent"

# Process the user input based on Vectorizer used for Commands:
def process_user_input_commands(user_query):
    """
    Takes The User's input, and processes the text.
    Afterwards, it stems and lemmatises the text, before
    transforming it into a TFIDF Sparse Matrix.

    Args:
        user_query (String): The User's input.

    Returns:
        Sparse Matrix: The User's input, converted to a Sparse Matrix.
    """
    preprocessed_user_query = process_text(user_query)
    tokenized_user_input = stemmed_and_lemmatised(preprocessed_user_query)
    
    tfidf_user_input = vectorizer_commands.transform([tokenized_user_input])
    return tfidf_user_input

# Retrieves the username and saves it in the last user file:
def get_username():
    """
    Get's The User's name from The User.
    Any name can be entered here, so there's a double verification
    between The User and the system to ensure that there is no error.

    Returns:
        String: The User's current name
    """
    username = input("May I have your name?\nEnter Here: ")
    confirmation = input(f"Thank you. Just to double check, it's {username}, right?\nPlease type either Yes or No: ")
    while(True):  
        confirmation = confirmation.strip().lower()
        if confirmation in ("yes", "ye", "y"):
            print("That's a lovely name! I'll save it for later so that I can refer to you correctly.")
            with open(user_data_file_path, "r") as file:
                user_data = json.load(file)
            user_data["username"] = username
            with open(user_data_file_path, "w") as file:
                json.dump(user_data, file, indent = 4)
            return username
        elif confirmation in ("no", "n"):
            print("No? Let's try again!")
            get_username()
            return username
        else:
            confirmation = input("I didn't quite get that. Can you try again? Please ensure you type only Yes or No in response: ")

# A brief introduction as to what the bot can do.
def introduction():
    """
    Used only for new Users, or the previous User when they want
    an introduction to the chatbot's capabilities.
    It will also let The User know what the currently available items are within the Warehouse.
    """
    print("Now, to introduce myself, I am Hermes!")
    print("I am your new AI assistant that is designed to help you organise your warehouse.\nFeel free to ask me anything, but I would recommend asking me what I can do first if you are stuck!")
    print("As a quick refresher the items that are currently available in the warehouse are:")
    for item in available_items:
                print(f"  - {item}")

# A function which explains the basic capabilities of what the chatbot can do:
def discoverability_general(chatbot_name, chatbot_style):
    """
    A function which handles the General Discoverability.
    This is called when The User specifically requests help
    for anything that is not within the constraints of 
    any functionalities in the Chatbot.

    Args:
        chatbot_name (String): The Chatbot's current name.
        chatbot_style (String): The Chatbot's current personality.
    """
    print(f"As {chatbot_name}, I am equipped with a few capabilities:")
    print("I am primarily an assistant to help you manage your warehouse. Specifically, I can:")
    print("-  Order Stock\n-  Check Stock\n-  Remove Stock\n")
    print("I can also engage in small talk, in case you wanted to chat!")
    print("I am also equipped with the ability to change my name to a name that you prefer.")
    print("And finally, I can change personality types to suit your preference. My 5 options are:")
    print("-  Caring\n-  Enthusiastic\n-  Friendly\n-  Professional\n-  Witty\n")

# Used to change the username saved in the user_data JSON:
def change_username(user_query):
    """
    Handles the main logic that comes with changing The User's name.
    Handles edge cases where The User rejects their name, but in a
    strange way.

    Args:
        user_query (String): The User's input.

    Returns:
        String: The User's current username.
    """
    if re.search(r"(?i)\bcall me ([\w]+)", user_query):
        username = change_username_specific(user_query)
        return username
    else:
        username = input("What would you now like to be called?\nUsername: ")
        username = username.strip()
        confirmation = input(f"Just to double check, it was this name: '{username}', right? Enter either Yes or No: ")
        while(True):
            confirmation = confirmation.strip().lower()
            if confirmation in ("yes", "ye", "y"):
                with open(user_data_file_path, "r") as file:
                    user_data = json.load(file)
                user_data["username"] = username
                with open(user_data_file_path, "w") as file:
                    json.dump(user_data, file, indent = 4)
                print("Alright, I got it!")
                print("Now, let's get back to it!")
                return username
            elif confirmation in ("no", "n"):
                print("No? Let's try again!")
                change_username()
                return username
            else:
                confirmation = input("I didn't quite get that. Can you try again? Please may I ask that you type only 'Yes' or 'No' in response: ")

# Used to change the username to the specific username found in the query:
def change_username_specific(user_query):
    """
    A helper function that works alongside change_username().
    This function is used when The User has specified their name
    within the initial query e.g. 'Call me Joshua'. It wil
    extract the name from the query, and set the current username
    as the new name specified.

    Args:
        user_query (String): The User's input.

    Returns:
        String: The User's current username.
    """
    match = re.search(r"(?i)\bcall me ([\w]+)", user_query)
    if match:
        username = match.group(1)
        username = username.strip()
        confirmation = input(f"Just to double check, it was this name: '{username}', right? Enter either Yes or No: ")
        while(True):
            confirmation = confirmation.strip().lower()
            if confirmation in ("yes", "ye", "y"):
                with open(user_data_file_path, "r") as file:
                    user_data = json.load(file)
                user_data["username"] = username
                with open(user_data_file_path, "w") as file:
                    json.dump(user_data, file, indent = 4)
                print("Now, back to it!")
                return username
            elif confirmation in ("no", "n"):
                print("No? Let's try again!")
                user_query = input("What would you like to be called? \nUsername: ")
                user_query_combined = "call me " + user_query
                username = change_username_specific(user_query_combined)
                return username
            else:
                confirmation = input("I didn't quite get that. Can you try again? Please ensure you type only Yes or No in response: ")
    else:
        print("Sorry, I didn't manage to catch your name in that sentence.")
        user_query = ("What did you want to be called? \nUsername: ")
        user_query_combined = "call me " + user_query
        change_username_specific(user_query_combined)
    
# Used to change the model's name:
def change_model_name(user_query):
    """
    This function changes the chatbot's name, and saves
    it within the user_data.json, so that it can be
    loaded from memory in subsequent runs.

    Args:
        user_query (String): The User's input.

    Returns:
        String: The Chatbot's current name.
    """
    if re.search(r"(?i)\bcall yourself ([\w]+)", user_query):
        chatbot_name = change_model_name_specific(user_query)
        return chatbot_name
    else:
        model_name = input("Sure! What would you like to call me? \nChatbot Name: ")
        model_name = model_name.strip()
        confirmation = input(f"Just to double check, it was this name: '{model_name}', right? Enter either Yes or No: ")
        while(True):
            confirmation = confirmation.strip().lower()
            if confirmation in ("yes", "ye", "y"):
                with open(user_data_file_path, "r") as file:
                    user_data = json.load(file)
                user_data["chatbot_name"] = model_name
                with open(user_data_file_path, "w") as file:
                    json.dump(user_data, file, indent = 4)
                print("Now, back to it!")
                return model_name
            elif confirmation in ("no", "n"):
                print("No? Let's try again!")
                change_model_name()
                return model_name
            else:
                confirmation = input("I didn't quite get that. Can you try again? Please ensure you type only Yes or No in response: ")

# Used to change the model's name, based on the query's input:
def change_model_name_specific(user_query):
    """
    A helper function which is used alongside change_model_name().
    This function detects whether a user has specified the model's 
    name within the query. If that is the case, then it will directly
    set The User's query to the Chatbot's new name.

    Args:
        user_query (String): The User's input.

    Returns:
        String: The Chatbot's current name.
    """
    match = re.search(r"(?i)\bcall yourself ([\w]+)", user_query)
    if match:
        model_name = match.group(1)
        model_name = model_name.strip()
        confirmation = input(f"Just to double check, it was this name: '{model_name}', right? Enter either Yes or No: ")
        while(True):
            confirmation = confirmation.strip().lower()
            if confirmation in ("yes", "ye", "y"):
                with open(user_data_file_path, "r") as file:
                    user_data = json.load(file)
                user_data["chatbot_name"] = model_name
                with open(user_data_file_path, "w") as file:
                    json.dump(user_data, file, indent = 4)
                print("Now, back to it!")
                return model_name
            elif confirmation in ("no", "n"):
                print("No? Let's try again!")
                user_query = input("What would you like me to be called? \nModel Name: ")
                user_query_combined = "call yourself " + user_query
                model_name = change_username_specific(user_query_combined)
                return model_name
            else:
                confirmation = input("I didn't quite get that. Can you try again? Please ensure you type only Yes or No in response: ")
    else:
        print("Sorry, I didn't manage to catch that.")
        user_query = ("What did you want me to be called? \nModel Name: ")
        user_query_combined = "call yourself " + user_query
        change_model_name_specific(user_query_combined)

# Used to get the personality that the user requested for the chatbot from their query:
def get_personality_from_query(user_query):
    """
    A helper function which extracts the personality
    from The User's query.

    Args:
        user_query (String): The User's input.

    Returns:
        None: Nothing is returned.
    """
    for personality, keywords in personality_keywords.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', user_query):
                return personality
    
    return None

# Swaps the personality of the chatbot so that the user responses are from the correct personality:
def personality_swap(user_query):
    """
    When the personality is altered by The User, this function
    swaps the personality of the Chatbot so that the Small Talk
    is updated correctly.

    Args:
        user_query (String): The User's input.

    Returns:
        String: The current personality of the Chatbot.
    """
    user_query_lower = user_query.lower()
    personality = get_personality_from_query(user_query_lower)
    while(True):
        if personality:
            with open(user_data_file_path, "r") as file:
                user_data = json.load(file)
                user_data["chatbot_style"] = personality.capitalize()
            with open(user_data_file_path, "w") as file:
                json.dump(user_data, file, indent = 4)
            print("Alright, I've updated my personality for you! Give it a whirl!")
            return personality.capitalize()
        else:
            ask_again = input("What personality are you looking for?\nUser Input: ")
            personality = get_personality_from_query(ask_again)

# Start of Program:

# Sets the filepaths for the inventory dataset, and the user_data:
file_path = os.path.join(current_dir,"inventory_dataset.csv")
user_data_file_path = os.path.join(current_dir,"user_data.json")

# Sets the stopping condition for the subsection of the introduction loop:
stopCondition = True

# Sets the stopping condition for the introduction loop:
userintroStopCondition = True

# Sets the stopping condition for the main loop for the chatbot:
mainLoop = True

# Sets the stopping condition used for the loop regarding the last user:
last_user_stopCondition = True

# Sets the flag used to keep track of whether the user is the last user or not:
last_user_flag = False

# Used to keep track of the state of the order:
order_list = {}

# A stack used to keep track of where the user is
# in regards to the different chatbot functionalities.
# This is here to effectively stop the user calling the same
# function whilst a previous process is ongoing e.g.
# Ordering -> Check -> Ordering
# should not occur, as the user hasn't finished the original
# ordering process.
state_stack = []

# Checks whether the dataset exists.
# If it does, then read the data in.
# Otherwise, generate the data and then read that data in.
if os.path.exists(file_path):
    inventory_dataset, available_items = read_inventory_dataset(file_path)
else:
    dataset_need_save = generate_inventory_data(items = ['Notebooks', 'Sharpeners', 'Rubbers', 'Pencils', 'Pens'])
    save_inventory_dataset(dataset_need_save, file_path)
    inventory_dataset, available_items = read_inventory_dataset(file_path)
    

available_items_formatted = preprocess_available_items(available_items)

# Checks to see if there is a user_data file
# If not, then creates the file, with basic information
# If there is, then reads in the data and sets it accordingly.
if os.path.exists(user_data_file_path) and os.path.getsize(user_data_file_path) > 0:
    with open(user_data_file_path, "r") as file:
        user_data = json.load(file)
        current_username = user_data["username"]
        chatbot_name = user_data["chatbot_name"]
        chatbot_style = user_data["chatbot_style"]
else:
    user_data = {
        "username": "PLACEHOLDER",
        "chatbot_name": "Hermes",
        "chatbot_style": "Friendly"
    }
    with open(user_data_file_path, "w") as file:
        json.dump(user_data, file, indent = 4)
        current_username = user_data["username"]
        chatbot_name = user_data["chatbot_name"]
        chatbot_style = user_data["chatbot_style"]
        


# Set-up for Users:
print("Greetings! Are you a new user to Hermes?")
user_confirmation = input("Please type Yes or No: ")
while(userintroStopCondition):
    user_confirmation.strip().lower()
    if user_confirmation in ("yes", "y", "ye"):
        username = get_username()
        introduction()
        userintroStopCondition = False
    elif user_confirmation in ("no", "n"):
        user_confirmation = input("Were you the last user? Please type Yes or No: ")
        while(last_user_stopCondition):
            user_confirmation.strip().lower()
            if user_confirmation in ("yes", "y"):
                last_user_flag = True
                print(f"Welcome back {current_username}! It's a pleasure to see you again.")
                user_introduction = input("Would you like an introduction? Please type Yes or No: ")
                while(stopCondition):
                    user_introduction.strip().lower()
                    if user_introduction in ("yes", "ye", "y"):
                        introduction()
                        stopCondition = False
                    elif user_introduction not in ("no", "n"):
                        user_introduction = input("I didn't get that. Can you please type Yes or No: ")
                    else:
                        stopCondition = False
                userintroStopCondition = False
                last_user_stopCondition = False
            elif user_confirmation in ("no", "n"):
                username = get_username()
                user_introduction = input("Would you like an introduction? Please type Yes or No: ")
                while(stopCondition):
                    user_introduction.strip().lower()
                    if user_introduction in ("yes", "ye", "y"):
                        introduction()
                        stopCondition = False
                    elif user_introduction not in ("no", "n"):
                        user_introduction = input("I didn't get that. Can you please type Yes or No: ")
                    else:
                        stopCondition = False
                userintroStopCondition = False
                last_user_stopCondition = False
            else:
                print("I didn't quite get that.")
                user_confirmation = input("Please type either Yes or No: ")
                user_confirmation.strip().lower()
        userintroStopCondition = False  
    else:
        user_confirmation = input("Sorry, I didn't get that. \nCan you please enter either Yes or No: ")

# Loading/Saving the models:
# Sets the names of each of the files:
file_path_smalltalk_friendly = os.path.join(current_dir, "smalltalk_model_data_friendly")
file_path_smalltalk_caring = os.path.join(current_dir, "smalltalk_model_data_caring")
file_path_smalltalk_enthusiastic = os.path.join(current_dir, "smalltalk_model_data_enthusiastic")
file_path_smalltalk_professional = os.path.join(current_dir, "smalltalk_model_data_professional")
file_path_smalltalk_witty = os.path.join(current_dir, "smalltalk_model_data_witty")
file_path_intents = os.path.join(current_dir, "intents_model_data")

# Checks to see whether each model exists:
# If it does, load the model's details and set them to the respective variables
# If it does not, then save the model after processing the respective data:
vectorizer_friendly, tfidf_matrix_small_talk_friendly, augmented_data_friendly = read_and_save_model(file_path_smalltalk_friendly, 
                                                                                                     vocab_processed_friendly, 
                                                                                                     vectorizer_friendly)
vectorizer_caring, tfidf_matrix_small_talk_caring, augmented_data_caring = read_and_save_model(file_path_smalltalk_caring, 
                                                                                               vocab_processed_caring, 
                                                                                               vectorizer_caring)
vectorizer_enthusiastic, tfidf_matrix_small_talk_enthusiastic, augmented_data_enthusiastic = read_and_save_model(file_path_smalltalk_enthusiastic, 
                                                                                                                 vocab_processed_enthusiastic, 
                                                                                                                 vectorizer_enthusiastic)
vectorizer_professional, tfidf_matrix_small_talk_professional, augmented_data_professional = read_and_save_model(file_path_smalltalk_professional, 
                                                                                                                 vocab_processed_professional, 
                                                                                                                 vectorizer_professional)
vectorizer_witty, tfidf_matrix_small_talk_witty, augmented_data_witty = read_and_save_model(file_path_smalltalk_witty, 
                                                                                            vocab_processed_witty, 
                                                                                            vectorizer_witty)

# Defines the dictionary that holds similar words to the available personalities:
# (Used in the similarity checking stage to determine the personality the user is referring to)
personalities = {
    "Friendly": {"tfidf_matrix": tfidf_matrix_small_talk_friendly,
                 "augmented_data": augmented_data_friendly,
                 "vectorizer": vectorizer_friendly},
    "Caring": {"tfidf_matrix": tfidf_matrix_small_talk_caring,
               "augmented_data": augmented_data_caring,
               "vectorizer": vectorizer_caring},
    "Enthusiastic": {"tfidf_matrix": tfidf_matrix_small_talk_enthusiastic, 
                     "augmented_data": augmented_data_enthusiastic,
                     "vectorizer": vectorizer_enthusiastic},
    "Professional": {"tfidf_matrix": tfidf_matrix_small_talk_professional,
                     "augmented_data": augmented_data_professional,
                     "vectorizer": vectorizer_professional},
    "Witty": {"tfidf_matrix": tfidf_matrix_small_talk_witty, 
              "augmented_data": augmented_data_witty,
              "vectorizer": vectorizer_witty}
}

# Checks to see if the user was the last user.
# If that is the case, then it loads their previous customisations.
# Else, it loads a default version of the Chatbot:
if last_user_flag:
    personality_data = personalities[chatbot_style]
    selected_tfidf_matrix_smalltalk = personality_data["tfidf_matrix"]
    selected_augmented_data = personality_data["augmented_data"]
    selected_vectorizer = personality_data["vectorizer"]
else:
    personality_data = personalities["Friendly"]
    selected_tfidf_matrix_smalltalk = personality_data["tfidf_matrix"]
    selected_augmented_data = personality_data["augmented_data"]
    selected_vectorizer = personality_data["vectorizer"]
    user_data = {
        "username": username,
        "chatbot_name": "Hermes",
        "chatbot_style": "Friendly"
    }
    with open(user_data_file_path, "w") as file:
        json.dump(user_data, file, indent = 4)
        current_username = user_data["username"]
        chatbot_name = user_data["chatbot_name"]
        chatbot_style = user_data["chatbot_style"]

# Either saves the ML model after creating it, or
# loads it from memory without re-training.
augmented_intents = preprocess_intents(intents)
classifier, vectorizer_ml = load_or_save_ml_model("trained_classifier", augmented_intents)

# Main Conversational Loop:
state = ""
swap = "No"
generalFlag = 1
while(mainLoop):
    state = "General"
    if generalFlag == 1:
        user_query = input(f"How may {chatbot_name} assist you today? \nUser Input:  ")
        generalFlag = 0
    
    small_talk_response = process_smalltalk(user_query, selected_tfidf_matrix_smalltalk, selected_augmented_data, selected_vectorizer)
    intent_check = predict_intent_ml(user_query, classifier, vectorizer_ml, augmented_intents)

    if small_talk_response and intent_check == "Unknown Intent":
        print(small_talk_response)
        generalFlag = 1
        continue
    elif intent_check != "Unknown Intent":
        handle_intent_check = handle_intent(intent_check, user_query, state)
        generalFlag = 1
    else:
        user_query = input("Sorry, I don't understand what you mean. Any chance you may be able to try again?\nUser Input: ")
        continue