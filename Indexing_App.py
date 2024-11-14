import streamlit as st
import os
import math
from collections import defaultdict, Counter
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from CustomeDictionary import CustomDictionary, CustomDefaultDict

def load_base_forms():
    """Load dictionary of base forms for irregular nouns."""
    return {
        'children': 'child',
        'mice': 'mouse',
        'men': 'man',
        'women': 'woman',
        'teeth': 'tooth',
        'feet': 'foot',
        'geese': 'goose',
        'oxen': 'ox',
        'data': 'datum',
        'criteria': 'criterion',
        'phenomena': 'phenomenon'
    }

def get_base_form(word, irregular_forms):
    """Get the base form of a word, handling both regular and irregular nouns."""
    word = word.lower()
    
    # Check irregular forms first
    if word in irregular_forms:
        return irregular_forms[word]
    
    # Handle regular plural forms
    if word.endswith('ies'):
        return word[:-3] + 'y'
    elif word.endswith('es'):
        if word.endswith('ses') or word.endswith('zes') or word.endswith('ches') or word.endswith('shes'):
            return word[:-2]
        return word[:-1]
    elif word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    
    return word

def is_likely_noun(word):
    """Check if a word is likely to be a noun based on rules and common patterns."""
    word = word.lower()
    
    # Common noun suffixes
    noun_suffixes = (
        'tion', 'ment', 'ness', 'ship', 'age', 'ity', 'ant', 
        'ence', 'er', 'or', 'ist', 'ing', 'ism', 'dom',
        'ary', 'ery', 'ory', 'cy', 'phy', 'ogy', 'ice',
        'ade', 'al', 'an', 'ance', 'ancy', 'ant', 'arc',
        'ard', 'ate', 'ent', 'ess', 'ice', 'ics', 'ide',
        'ine', 'ion', 'ite', 'let', 'oid', 'oma', 'ose',
        'ric', 'sis', 'ure'
    )
    
    # Check for noun suffixes
    if any(word.endswith(suffix) for suffix in noun_suffixes):
        return True
    
    # Check for capitalization (proper nouns)
    if word[0].isupper():
        return True
    
    return False

def get_word_variants(word):
    """Generate possible variants of a word."""
    variants = {word.lower()}
    
    # Add common suffix variations
    word_lower = word.lower()
    if word_lower.endswith('y'):
        variants.add(word_lower[:-1] + 'ies')
    elif word_lower.endswith('s'):
        variants.add(word_lower[:-1])
    else:
        variants.add(word_lower + 's')
        if word_lower.endswith('e'):
            variants.add(word_lower + 's')
        elif word_lower.endswith('y'):
            variants.add(word_lower[:-1] + 'ies')
        else:
            variants.add(word_lower + 'es')
    
    return variants

def extract_nouns_and_entities(content):
    """Extract nouns from content using custom rules."""
    # Load irregular forms (only done once)
    if not hasattr(extract_nouns_and_entities, 'irregular_forms'):
        extract_nouns_and_entities.irregular_forms = load_base_forms()
    
    # Common stop words that aren't nouns
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'with', 'by', 'from', 'up', 'down', 'is', 'are', 'was',
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
        'does', 'did', 'will', 'would', 'shall', 'should', 'may',
        'might', 'must', 'can', 'could', 'of', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }
    
    # Tokenize the content
    words = tokenize_text(content)
    
    # Filter out stop words and extract likely nouns
    nouns = []
    for word in words:
        word_lower = word.lower()
        if word_lower not in stop_words and (
            is_likely_noun(word) or 
            len(word) > 3  # Include longer words as they're more likely to be nouns
        ):
            # Get base form of the word
            base_form = get_base_form(word, extract_nouns_and_entities.irregular_forms)
            nouns.append(base_form)
    

    lemmatizer = WordNetLemmatizer()
    lemmatized_nouns = [lemmatizer.lemmatize(noun) for noun in nouns]
    
    return nouns

def expand_query_with_variants(query_tokens):
    """Expand query tokens with their variants."""
    expanded_query = set()
    for token in query_tokens:
        expanded_query.update(get_word_variants(token))
    return list(expanded_query)

# Helper to get synonyms
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word): # Extracts synonyms sets for the word
        for lemma in syn.lemmas(): # extracts lemma from the synonym such as word.POS.name
            synonyms.add(lemma.name().lower()) # extracts name from lemma and add it to synonym set
    return synonyms

def expand_query_with_synonyms(query_tokens):
    expanded_query = set(query_tokens)
    for word in query_tokens:
        expanded_query.update(get_synonyms(word)) # gets synonyms from helper function and update query set
    return list(expanded_query) # convert set -> list

def extract_nouns_and_entities(content):
    words = word_tokenize(content.lower()) # split words on the whitespaces
    pos_tags = pos_tag(words) # extract parts of speech tags from the words list it works as [Fan: NN(Common Noun)]
    
    nouns = [word for word, pos in pos_tags if pos in ('NN', 'NNS','NNP', 'NNPS')] # extract NN(Common Noun - Singular) and NNS (Common Noun - Plural), because we only require nouns in our assignment scope 
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_nouns = [lemmatizer.lemmatize(noun) for noun in nouns] # lemmatizes nouns i.e removes ing, es from words
    
    return lemmatized_nouns

def calculate_tf(doc_words):
    tf = CustomDictionary()
    total_words = len(doc_words) # length of a single document
    word_counts = Counter(doc_words) # word occurence attached with each word
    
    for word, count in word_counts.items(): # method to get keys, values pairs from dictionary
        tf[word] = count / total_words # calculate word frequency
    return tf

def calculate_idf(documents):
    idf = CustomDictionary()
    total_docs = len(documents) # size of dataset
    word_doc_counts = CustomDefaultDict(int) # default dictionary of type int, so no need for initialization of data
    
    for doc_words in documents: # iterate over the dataset, doc_words is content of each file
        unique_words = set(doc_words) # set removes duplicate occurences of a word
        for word in unique_words: 
            word_doc_counts[word] += 1 # Populate words in the dataset with their occurences in complete dataset
    
    for word, doc_count in word_doc_counts.items():
        idf[word] = math.log(total_docs / (1 + doc_count)) # formula to calculate idf for each word the the dataset
    return idf

def calculate_tfidf(doc_words, tf, idf):
    tfidf = CustomDictionary()
    for word in doc_words:
        if word in idf:
            tfidf[word] = tf[word] * idf[word]
    return tfidf

def phrase_match(content, query):
    content_words = word_tokenize(content.lower()) # content is a single document, split document on basis of whitespace
    query_words = word_tokenize(query.lower()) # query is a user search query, split query on basis of whitespace
    
    for i in range(len(content_words) - len(query_words) + 1):  # iterate over each word in the document, content words - query words so that list index out of range does not occure
        if content_words[i:i+len(query_words)] == query_words: # phrase matching
            return True
    return False

class SearchEngine:
    def __init__(self, docs_directory="documents"):
        self.index = CustomDefaultDict(list) # default dictionary for indexes
        self.documents = CustomDictionary()
        self.docs_directory = docs_directory
        
        if not os.path.exists(self.docs_directory):
            os.makedirs(self.docs_directory)
        
        #self.load_documents()
    
    def load_documents(self):
        """Load all documents from the documents directory."""
        print("\nLoading existing documents...")
        loaded_count = 0
        
        for filename in os.listdir(self.docs_directory): # lists all files in the provided folder
            if filename.endswith(".txt"):
                try:
                    with open(os.path.join(self.docs_directory, filename), 'r', encoding='utf-8') as file: # append filename with the base folder address
                        title = filename
                        content = file.read().strip() # read file and strip (removes extra whitespaces from end)
                        self.documents[title] = content
                        loaded_count += 1
                except Exception as e:
                    print(f"Error loading document {filename}: {str(e)}")
        
        if loaded_count > 0:
            print(f"Successfully loaded {loaded_count} documents.")
            self.update_index()
            print(len(self.index.items()))
        else:
            print("No existing documents found.")
        
    def update_index(self):
        doc_word_list = []
        
        for doc_id, content in self.documents.items():
            nouns_and_entities = extract_nouns_and_entities(content) # extracts nouns and entities from the content
            doc_word_list.append(nouns_and_entities)
        
        idf = calculate_idf(doc_word_list)
        
        self.index.clear()
        
        for doc_id, content in self.documents.items():
            doc_words = extract_nouns_and_entities(content)
            tf = calculate_tf(doc_words)
            tfidf = calculate_tfidf(doc_words, tf, idf)
            important_terms = {term for term, score in tfidf.items() if score > 0.01} # apply a threshold and extract important terms
            for term in important_terms:
                self.index[term].append(doc_id) # important terms are made indexes for our search engine
        
        print(self.index)
        
    
    def add_document(self, title, content):
        self.documents[title] = content
        print(f"\nDocument '{title}' added successfully!")
    
    def list_documents(self):
        if not self.documents:
            print("\nNo documents in the system.")
            return
        
        print("\nList of documents:")
        for idx, title in enumerate(self.documents.keys(), 1):
            print(f"{idx}. {title}")
    
    def search_by_title(self, query):
        matching_docs = []
        for doc_id in self.documents.keys():
            if query.lower() in doc_id.lower():
                matching_docs.append(doc_id)
        return matching_docs
    
    def search_by_content(self, query):
        matching_docs = set()
        
        # First try exact phrase matching
        for doc_id, content in self.documents.items():
            if phrase_match(content, query):
                matching_docs.add(doc_id)
        
        # If no exact matches found, try semantic search
        if not matching_docs:
            query_tokens = word_tokenize(query.lower())
            # Extract only nouns from the query for index-based search
            query_pos_tags = pos_tag(query_tokens)
            query_nouns = [word for word, pos in query_pos_tags 
                        if pos in ('NN', 'NNS', 'NNP', 'NNPS')]
            
            # Get expanded tokens only for nouns
            expanded_noun_tokens = {}
            for noun in query_nouns:
                # Store noun and its synonyms
                expanded_noun_tokens[noun] = expand_query_with_synonyms([noun])
            
            print(f"Query Nouns: {expanded_noun_tokens}")
            # Find documents that match the nouns using the index
            noun_matching_docs = set()
            
            if expanded_noun_tokens:
                # Get the first noun and its synonyms
                first_noun = list(expanded_noun_tokens.keys())[0]
                # Start with documents matching the first noun
                noun_matching_docs = set(self.index.get(first_noun, []))
                # Add documents matching its synonyms
                for synonym in expanded_noun_tokens[first_noun]:
                    noun_matching_docs |= set(self.index.get(synonym, []))
                
                # Process remaining nouns
                for noun in list(expanded_noun_tokens.keys())[1:]:
                    # Get documents matching current noun
                    current_docs = set(self.index.get(noun, []))
                    # Add documents matching its synonyms
                    for synonym in expanded_noun_tokens[noun]:
                        current_docs |= set(self.index.get(synonym, []))

                    # Intersect with previous results
                    noun_matching_docs &= current_docs
            return noun_matching_docs
        
        return matching_docs

def display_results(docs):
    if docs:
        print("\nFound documents:")
        for idx, doc in enumerate(docs, 1):
            print(f"{idx}. {doc}")
    else:
        print("\nNo matching documents found.")

def get_multiline_input(prompt):
    print(prompt)
    print("(Enter an empty line to finish)")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)

st.set_page_config(page_title="Search Engine", layout="centered")

if 'search_engine' not in st.session_state:
    st.session_state.search_engine = SearchEngine(docs_directory="./archive/business")

# Access the search engine instance
search_engine = st.session_state.search_engine
st.title("Document Search Engine")
st.write("This app allows you to add, list, and search documents by title or content.")

# Sidebar navigation
option = st.sidebar.selectbox("Choose an option", ["Home", "Add Document", "List Documents", "Search by Title", "Search by Content"])

if option == "Home":
    st.subheader("Welcome to the Document Search Engine")
    st.write("Select an option from the sidebar to get started.")

elif option == "Add Document":
    st.subheader("Add New Documents")
    uploaded_files = st.file_uploader("Upload text files", type="txt", accept_multiple_files=True)
    if st.button("Add Documents"):
        if uploaded_files:
            added_files = []
            for uploaded_file in uploaded_files:
                title = uploaded_file.name
                content = uploaded_file.read().decode("utf-8").strip()
                search_engine.add_document(title, content)
                added_files.append(title)

            search_engine.update_index()
            st.success(f"Successfully added {len(added_files)} documents!")
            st.write("Added documents:")
            for file in added_files:
                st.write(file)
        else:
            st.error("Please upload at least one file.")

elif option == "List Documents":
    st.subheader("List of Documents")
    docs = search_engine.documents.keys()
    
    if docs:
        for idx, doc in enumerate(docs, 1):
            st.write(f"{idx}. {doc}")
    else:
        st.warning("No documents found.")

elif option == "Search by Title":
    st.subheader("Search Documents by Title")
    query = st.text_input("Enter title to search")
    
    if st.button("Search"):
        results = search_engine.search_by_title(query)
        if results:
            st.write("Found documents:")
            for result in results:
                st.write(result)
        else:
            st.warning("No matching documents found.")

elif option == "Search by Content":
    st.subheader("Search Documents by Content")
    query = st.text_input("Enter content to search")
    
    if st.button("Search"):
        results = search_engine.search_by_content(query)
        if results:
            st.write("Found documents:")
            for result in results:
                st.write(f"**Title:** {result}")
                st.write(f"**Content:** {search_engine.documents[result]}")
                st.write("---")
        else:
            st.warning("No matching documents found.")