import time

import spacy
from rank_bm25 import BM25Okapi

from sentence_transformers import CrossEncoder

import os
import json
import sys

from tqdm import tqdm
import pickle


class BaseRetriever:
    def __init__(self, index_name, language="en"):
        self.index = []  # Initialize an empty index
        self.name = "Base"
        index_file_path = os.path.join("./indices", index_name)
        with open(index_file_path, 'r') as file:
            data = json.load(file)
        self.text_corpus = [item["context"] for item in data]
        self.language = language

    def indexing(self, documents):
        """
        Index a list of documents.

        Args:
            documents (list): A list of strings to be indexed.
        """
        self.text_corpus = documents

    def get_top_k_documents(self, query, k=10, scores=False):
        """
        Retrieve the top k documents for a given query.

        Args:
            query (str): The query text.
            k (int): The number of top documents to retrieve.
            scores (bool): Whether to return scores along with documents.

        Returns:
            list: A list of top k documents (and scores if scores=True).
        """
        # Implement retrieval logic here
        # You can override this method in derived classes

        # For demonstration purposes, return the first k documents as top documents
        if scores:
            return self.index[:k], [1.0] * k  # Dummy scores
        else:
            return self.index[:k]

    def predict(self, query, text):
        """
        Predict a score for a given query and text.

        Args:
            query (str): The query text.
            text (str): The text to calculate the score for.

        Returns:
            float: The calculated score for the query and text.
        """
        # Implement scoring logic here
        # You can override this method in derived classes

        # For demonstration purposes, return a dummy score
        return 1.0  # Dummy score


class BM25Retriever(BaseRetriever):
    _instances = {}

    def __new__(cls, index_name, language, *args, **kwargs):
        key = (index_name, language)
        if key not in cls._instances:
            print("Creating new instance of BM25Retriever for index_name={}, language={}".format(index_name, language))
            instance = super(BM25Retriever, cls).__new__(cls)
            cls._instances[key] = instance
        else:
            print("Loading previous instance of BM25Retriever for index_name={}, language={}".format(index_name, language))
        return cls._instances[key]

    def __init__(self, index_name, k=10, language="en"):
        if not hasattr(self, 'initialized'):
            print("Setting up BM25")
            super().__init__(index_name=index_name, language=language)
            self.index_name = index_name
            self.name = "BM25"
            self.k = k
            if self.language == "en":
                print("Load English Spacy ...")
                self.nlp = spacy.load("en_core_web_sm")  # Load the English SpaCy model
                print("Loaded English Spacy")
            elif self.language == "de":
                print("Load German Spacy ..")
                self.nlp = spacy.load("de_core_news_sm")
                print("Loaded German Spacy")
            print("Done")
            self.initialized = True

    def indexing(self):
        """
        Index a list of documents using BM25Okapi.

        Args:
            documents (list): A list of strings to be indexed.
        """
        start_time_indexing = time.time()
        
        # Define the filename for the tokenized_corpus
        filename = f'./indices/{self.index_name.split(".")[0]}.pkl'
        
        # Check if the file exists
        if os.path.exists(filename):
            # Load the tokenized_corpus from the file
            with open(filename, 'rb') as f:
                tokenized_corpus = pickle.load(f)
            print("Loaded tokenized_corpus from file")
        else:
            # If the file doesn't exist, preprocess the text_corpus and save it
            tokenized_corpus = self.batch_preprocess(self.text_corpus)
            with open(filename, 'wb') as f:
                pickle.dump(tokenized_corpus, f)
            print("Saved tokenized_corpus to file")
        
        self.index = BM25Okapi(tokenized_corpus)
        print(f"Indexing took {time.time() - start_time_indexing} seconds")
    
    def batch_preprocess(self, texts):
        print("Starting Batch Processing")
        docs = list(tqdm(self.nlp.pipe(texts, disable=["parser", "ner"], batch_size=1000), total=len(texts), desc="Batch Processing", file=sys.stdout))
        print("Done Splitting to Docs")
        
        processed_texts = []
        total_texts = len(texts)

        # Use tqdm for a progress bar
        for doc in tqdm(docs, total=total_texts, desc="Preprocessing", file=sys.stdout):
            tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.is_stop]
            processed_texts.append(tokens)
        print("Done Tokenization")

        return processed_texts

    def preprocess(self, text):
        # Tokenize, stem, and remove stop words using SpaCy
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.is_stop]
        return tokens

    def get_top_k_documents(self, query, k=None, scores=False):
        """
        Retrieve the top k documents for a given query using BM25Okapi.

        Args:
            query (str): The query text.
            k (int): The number of top documents to retrieve.
            scores (bool): Whether to return scores along with documents.

        Returns:
            list: A list of top k documents (and scores if scores=True).
        """
        if k is None:
            k = self.k
        if not self.index:
            raise ValueError("BM25Okapi index has not been created. Please call indexing() first.")

        tokenized_query = self.preprocess(query)

        doc_scores = self.index.get_scores(tokenized_query)

        # Sort documents by score and get the top k documents
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k]
        print(f"BM25 Top-Indizes retrieved ...")  # Debugging-Ausgabe

        top_documents = [self.text_corpus[i] for i in top_indices]

        if scores:
            top_scores = [doc_scores[i] for i in top_indices]
            print(f"BM25 Top-Scores retrieved ...")  # Debugging-Ausgabe
            return top_documents, top_scores
        else:
            return top_documents

    def predict(self, query, passage):
        # Tokenize, stem, and remove stop words from the passage and query, then calculate BM25 score
        passage_tokens = self.preprocess(passage)
        query_tokens = self.preprocess(query)
        bm25 = BM25Okapi([passage_tokens])
        return bm25.get_scores(query_tokens)

class CERetriever(BaseRetriever):
    _instances = {}

    def __new__(cls, index_name, language, *args, **kwargs):
        key = (index_name, language)
        if key not in cls._instances:
            print("Creating new instance of CERetriever for index_name={}, language={}".format(index_name, language))
            instance = super(CERetriever, cls).__new__(cls)
            cls._instances[key] = instance
        else:
            print("Loading previous instance of CERetriever for index_name={}, language={}".format(index_name, language))
        return cls._instances[key]

    def __init__(self, index_name, k=10, language="en"):
        if not hasattr(self, 'initialized'):
            print("Setting up CE")
            super().__init__(index_name=index_name, language=language)
            if self.language == "de":
                model_name = "cross-encoder/msmarco-MiniLM-L6-en-de-v1"
            else:
                model_name = "cross-encoder/ms-marco-MiniLM-L-2-v2"
            self.ce = CrossEncoder(model_name, max_length=512)
            self.name = "Cross-Encoder"
            self.k = k
            print("Done")
            self.initialized = True    

    def get_top_k_documents(self, query, k=None, scores=False):
        """
        Retrieve the top k documents for a given query.

        Args:
            query (str): The query text.
            k (int): The number of top documents to retrieve.
            scores (bool): Whether to return scores along with documents.

        Returns:
            list: A list of top k documents (and scores if scores=True).
        """
        if k is None:
            k = self.k

        if not self.text_corpus:
            raise ValueError("The text corpus has not been set. Please call indexing() first.")

        query_passage_pairs = [(query, passage) for passage in self.text_corpus]

        print("CE: Got query-passage pairs") 

        # Calculate cross encoder scores between the query and documents
        scores_ce = self.ce.predict(query_passage_pairs)

        print("CE: Got scores")

        # Sort the pairs based on scores in descending order
        sorted_pairs = sorted(zip(query_passage_pairs, scores_ce), key=lambda x: x[1], reverse=True)

        # Unzip the sorted pairs to get the sorted documents and scores
        sorted_documents, sorted_scores = zip(*sorted_pairs)

        sorted_passages = [pair[1] for pair in sorted_documents]

        print("CE: Got sorted documents")

        # Extract the top-k documents and scores if requested
        if scores:
            return sorted_passages[:k], sorted_scores[:k]
        else:
            return sorted_passages[:k]

class RetrieverEnsemble(BaseRetriever):
    _instances = {}

    def __new__(cls, retrievers, index_name, language="en", *args, **kwargs):
        key = (index_name, language)
        print(f"instances: {cls._instances}\nnew key: {key}")
        if key not in cls._instances:
            print("Creating new instance of RetrieverEnsemble for index_name={}, language={}".format(index_name, language))
            instance = super(RetrieverEnsemble, cls).__new__(cls)
            cls._instances[key] = instance
        else:
            print("Loading previous instance of RetrieverEnsemble for index_name={}, language={}".format(index_name, language))
        return cls._instances[key]

    def __init__(self, retrievers, index_name, language="en"):
        if not hasattr(self, 'initialized'):
            # super init
            super().__init__(index_name=index_name, language=language)
            self.index_name = index_name
            self.language = language
            print("Setting up Ensemble")
            self.retrievers = retrievers
            self.name = "Ensemble"
            self.retrievers[0].indexing()
            print("Done")
            self.initialized = True

    def get_top_k_documents(self, query, k=None, scores=True):
        # if k is None:
        #     # Use the default k value of the first retriever in the list
        #     k = self.retrievers[0].k

        retrieved_passages = self.text_corpus
        for i in range(len(self.retrievers)):
            if i != 0:
                self.retrievers[i].indexing(retrieved_passages)
            # results = self.retrievers[i].get_top_k_documents(query,k,scores=True)
            results = self.retrievers[i].get_top_k_documents(query,scores=True)
            if i == len(self.retrievers):
                if scores:
                    return results
                else:
                    return results[0]
            else:
                retrieved_passages = results[0]

        return results