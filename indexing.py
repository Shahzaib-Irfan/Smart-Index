import os
import math
from collections import defaultdict, Counter
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

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
    tf = {}
    total_words = len(doc_words) # length of a single document
    word_counts = Counter(doc_words) # word occurence attached with each word
    
    for word, count in word_counts.items(): # method to get keys, values pairs from dictionary
        tf[word] = count / total_words # calculate word frequency
    return tf

def calculate_idf(documents):
    idf = {}
    total_docs = len(documents) # size of dataset
    word_doc_counts = defaultdict(int) # default dictionary of type int, so no need for initialization of data
    
    for doc_words in documents: # iterate over the dataset, doc_words is content of each file
        unique_words = set(doc_words) # set removes duplicate occurences of a word
        for word in unique_words: 
            word_doc_counts[word] += 1 # Populate words in the dataset with their occurences in complete dataset
    
    for word, doc_count in word_doc_counts.items():
        idf[word] = math.log(total_docs / (1 + doc_count)) # formula to calculate idf for each word the the dataset
    return idf

def calculate_tfidf(doc_words, tf, idf):
    tfidf = {}
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
        self.index = defaultdict(list) # default dictionary for indexes
        self.documents = {}
        self.docs_directory = docs_directory
        
        if not os.path.exists(self.docs_directory):
            os.makedirs(self.docs_directory)
        
        self.load_documents()
    
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
            important_terms = {term for term, score in tfidf.items() if score > 0.1} # apply a threshold and extract important terms
            for term in important_terms:
                self.index[term].append(doc_id) # important terms are made indexes for our search engine
        
    
    def add_document(self, title, content):
        self.documents[title] = content
        self.update_index()
        print(len(self.index.items()))
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
        for doc_id in self.documents:
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
            expanded_noun_tokens = []
            for noun in query_nouns:
                expanded_noun_tokens.extend(expand_query_with_synonyms([noun]))
            
            # Find documents that match the nouns using the index
            noun_matching_docs = set()
            if expanded_noun_tokens:
                noun_matching_docs = set(self.index.get(expanded_noun_tokens[0], []))
                for token in expanded_noun_tokens[1:]:
                    if token in self.index:
                        noun_matching_docs &= set(self.index[token])
            
            # If we found documents matching nouns, filter them further using non-noun terms
            if noun_matching_docs:
                non_noun_tokens = [word for word, pos in query_pos_tags 
                                if pos not in ('NN', 'NNS', 'NNP', 'NNPS')]
                
                # If there are no non-noun tokens, return the noun matches
                if not non_noun_tokens:
                    return noun_matching_docs
                
                # For each document that matched nouns, check if it contains the non-noun terms
                for doc_id in noun_matching_docs:
                    content = self.documents[doc_id].lower()
                    if all(token.lower() in content for token in non_noun_tokens):
                        matching_docs.add(doc_id)
            else:
                # If no noun matches found, fall back to basic content search
                for doc_id, content in self.documents.items():
                    content_lower = content.lower()
                    if all(token.lower() in content_lower for token in query_tokens):
                        matching_docs.add(doc_id)
        
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

def main():
    search_engine = SearchEngine(docs_directory="./archive/business")
    
    while True:
        print("\n=== Search Engine ===")
        print("1. Add new document")
        print("2. List all documents")
        print("3. Search by title")
        print("4. Search by content")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            title = get_multiline_input("\nEnter document title:")
            if not title:
                print("Title cannot be empty!")
                continue
                
            content = get_multiline_input("\nEnter document content:")
            if not content:
                print("Content cannot be empty!")
                continue
                
            search_engine.add_document(title, content)
            
        elif choice == "2":
            search_engine.list_documents()
            
        elif choice == "3":
            query = input("\nEnter title to search: ")
            results = search_engine.search_by_title(query)
            display_results(results)
            
        elif choice == "4":
            query = input("\nEnter content to search: ")
            results = search_engine.search_by_content(query)
            display_results(results)
            
        elif choice == "5":
            print("\nThank you for using the Search Engine!")
            break
            
        else:
            print("\nInvalid choice! Please try again.")

if __name__ == "__main__":
    main()