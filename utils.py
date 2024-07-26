from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

import pandas as pd
import numpy as np
import json
import torch
import os
from tqdm import tqdm    
import pingouin as pg


class Emotion_Attribution :

    def __init__(self) :
      self.model = SentenceTransformer('all-mpnet-base-v2')

      self.queries = None
      self.embeddings = None

      self.query_embeddings = None
      self.corpus_embeddings = None

      self.idx_to_spotify = None            # Query id -> Spotify id
      self.idx_to_query = None              # Query id -> tag
      self.idx_to_corpus = None             # Corpus id -> word from NRC Lexicon
      self.query_to_idx = None              # Tag -> query id

      self.idx_to_match = dict()            # Query id -> matched words after semantic search
      self.idx_to_nrc_vectors = dict()      # Query id -> emotional vectors of matched words
      self.idx_to_vector = dict()           # Query id -> emotional vector

      self.emotion_to_vector = dict()       # Emotion -> vector
      self.nrc_to_vector = dict()           # Word from NRC Lexicon -> emotion vector
      self.tag_to_vector = dict()           # Tag -> emotion vector

      self.spotify_to_tags = dict()         # Spotify id -> selected tags after filtering for maximizing inter-rater agreement
      self.spotify_to_kappa = dict()        # Spotify id -> final kappa score after filtering for inter-rater agreement
      
      self.spotify_to_vector = dict()       # Spotify id -> emotion vector
      self.spotify_to_metadata = dict()     # Spotify id -> (title, artist, preview_url)

      self.idx_to_mtg = None                # Query id -> track id (mtg)
      self.mtg_to_vector = dict()           # Track id (mtg) -> emotion vector
      self.mtg_to_metadata = dict()         # Track id (mtg) -> (title, artist)

      self.initial_icc = []
      self.final_icc = []
      self.initial_nb_tags = []
      self.final_nb_tags = []

      self.emotion_to_idx = {               # Emotion -> id
          "anger": 0,
          "anticipation": 1,
          "disgust": 2,
          "fear": 3,
          "joy": 4,
          "sadness": 5,
          "surprise": 6,
          "trust": 7,
      }
      self.idx_to_emotion = {v: k for k, v in self.emotion_to_idx.items()}      # id -> emotion 

    # ============================== #
    # |       DATA WRANGLING       | #
    # ============================== #

    def get_nrc_lexicon(self, path_nrc) :
        """Retrieve words from the NRC Lexicon
        
        Parameters:
            - path_nrc (str): path to the NRC Lexicon file
        
        Returns:
            - nrc (pd.DataFrame): DataFrame containing the words from the NRC Lexicon and their associated emotions
        """
        nrc = pd.read_csv(path_nrc, sep='\t', header=None)
        nrc.columns = ['word', 'emotion', 'association']
        nrc = nrc[nrc['association'] == 1]

        return nrc


    # ============================== #
    # |    SENTENCE EMBEDDINGS     | #
    # ============================== #

    def sentence_embeddings(self, tags_df, nrc_df, to_save_queries, to_save_corpus, compute_embeddings = False) :
        """Compute the embeddings of each unique tag and each word from the NRC Lexicon, or load them from saved files.
        
        Parameters:
            - tags_df (pd.DataFrame): DataFrame containing the tags
            - nrc_df (pd.DataFrame): DataFrame containing the words from the NRC Lexicon
            - to_save_queries (str): path to save the embeddings of the tags
            - to_save_corpus (str): path to save the embeddings of the words from the NRC Lexicon
            - compute_embeddings (bool): whether to compute the embeddings or load them from the saved files

        Returns:
            - query_embeddings (torch.Tensor): embeddings of the tags
            - corpus_embeddings (torch.Tensor): embeddings of the words from the NRC Lexicon        
        """
        # Corpus : words from the NRC Lexicon
        self.corpus = list(nrc_df['word'].unique())

        # Queries : tags
        self.queries = tags_df['tag'].unique().tolist()

        # Index mapping
        self.idx_to_query = {idx: query for idx, query in enumerate(self.queries)}
        self.idx_to_corpus = {idx: c for idx, c in enumerate(self.corpus)}
        
        self.idx_to_spotify = {idx: spotify_id for idx, spotify_id in enumerate(tags_df['spotify_id'].unique())}
        self.query_to_idx = {query: idx for idx, query in self.idx_to_query.items()}

        if compute_embeddings :
            # Get sentence embeddings
            self.corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=True)
            self.query_embeddings = self.model.encode(self.queries, convert_to_tensor=True)

            # Save tensors
            torch.save(self.corpus_embeddings, to_save_corpus)
            torch.save(self.query_embeddings, to_save_queries)
        
        else :
            # Load tensors
            self.corpus_embeddings = torch.load(to_save_corpus)
            self.query_embeddings = torch.load(to_save_queries)

        # Move to GPU if available
        if torch.cuda.is_available():
            self.corpus_embeddings = self.corpus_embeddings.cuda()
            self.query_embeddings = self.query_embeddings.cuda()
        
        return self.query_embeddings, self.corpus_embeddings


    # ============================== #
    # |      EMOTION VECTORS       | #
    # ============================== #

    def create_emotional_basis(self) :
        """Create the emotional basis vectors."""
        for emotion, idx in self.emotion_to_idx.items() :
            if idx <= 7 :
                v = np.zeros(8)
                v[idx] = 1
                self.emotion_to_vector[emotion] = v

    def create_emotional_vectors_nrc(self, nrc) :
        """Create the emotional vectors for each word in the NRC Lexicon.
        
        Parameters:
            - nrc (pd.DataFrame): DataFrame containing the words from the NRC Lexicon and their associated emotions
        """
        words_nrc = nrc['word'].unique()

        for word in words_nrc:
            v = np.zeros(8)
            # Retrieve all emotions associated with the word
            emotions = nrc[nrc['word'] == word]["emotion"].values

            for emotion in emotions:
                # We don't want the "positive" and "negative" emotions
                if emotion not in ["positive", "negative"] :
                    v[self.emotion_to_idx[emotion]] = 1

            self.nrc_to_vector[word] = v


    # ============================== #
    # |       SEMANTIC SEARCH      | #
    # ============================== #

    def semantic_matching(self, query_embeddings, corpus_embeddings,
                          threeshold_exact = 0.95, threeshold_high = 0.9, threeshold_medium = 0.5, 
                          proportion_keep = 0.5, topk = 7) :
        """Perform semantic search between the queries (tags) and words from the NRC Lexicon (corpus), and 
        weighted majority vote to derive emotion vectors of all tags in the dataset.

        Parameters:
            - query_embeddings (torch.Tensor): embeddings of the tags
            - corpus_embeddings (torch.Tensor): embeddings of the words from the NRC Lexicon
            - threeshold_exact (float): threeshold for exact matches
            - threeshold_high (float): threeshold for high-quality matches
            - threeshold_medium (float): threeshold for moderate quality matches
            - proportion_keep (float): threeshold for keeping emotions after match
            - topk (int): number of matches to retrieve for each query in Semantic Search
        """

        # Perform Semantic Search using SentenceTransformer library between each query and all entries from corpus 
        semantic_matching = util.semantic_search(query_embeddings, corpus_embeddings, top_k=topk)

        # To store some statistics
        no_matches_count = 0
        exacts_matches_count = 0 
        matches_count = 0
        matches_lengths = []
        nb_queries = len(self.queries)

        for idx, query in enumerate(self.queries):

            nrc_matches = semantic_matching[idx]        #Top k NRC matches for the query
            exact_match = False
            
            # 1. Only retrieve high-quality matches with a score >= threeshold_high 
            matches = [res for res in nrc_matches if res['score'] >= threeshold_high]

            # 2. Fallback to moderate quality matches if no high-quality matches are found
            if not matches:
                matches = [res for res in nrc_matches if res['score'] >= threeshold_medium]

            # 3. Check for exact match (score >= threeshold_exact)
            if len(matches) > 0 and matches[0]['score'] >= threeshold_exact:   # Exact match found -> only keep this match
                res = matches[0]
                exact_match = True

                # Store (matched word, similarity score)
                nrc_word = self.idx_to_corpus[res['corpus_id']]     
                self.idx_to_match[idx] = [(nrc_word, res['score'])]

                # Store vector of matched word 
                self.idx_to_nrc_vectors[idx] = self.nrc_to_vector[nrc_word]

                # Store the final emotion vector of the tag
                nrc_vector = self.nrc_to_vector[nrc_word]
                self.idx_to_vector[idx] = nrc_vector

                exacts_matches_count += 1
                matches_lengths.append(1)
            
            # 4. No exact match found -> perform weighted majority vote
            if not exact_match :

                # Store the matches (word, similarity score) 
                self.idx_to_match[idx] = [(self.idx_to_corpus[res['corpus_id']], res['score']) for res in matches]
                
                # Retrieve the emotion vectors of the matched words
                emotion_vectors = [self.nrc_to_vector[self.idx_to_corpus[res['corpus_id']]] for res in matches]
                self.idx_to_nrc_vectors[idx] = emotion_vectors

                # Weighted average of the emotion vectors of the matched words (weights = similarity scores)
                if emotion_vectors:
                    similarities = np.array([res['score'] for res in matches])
                    emotion_vectors = np.array(emotion_vectors)
                    
                    weighted_sum = np.dot(similarities, emotion_vectors)
                    weighted_avg = weighted_sum / similarities.sum()
                    
                    # Only pick emotions with agreement accross matches (intensity >= proportion_keep)
                    final_vector = (weighted_avg >= proportion_keep).astype(int)
                    self.idx_to_vector[idx] = final_vector

                    matches_count += 1
                
                # No valid matches above the threshold (proportion_keep)
                else:
                    self.idx_to_match[idx] = []
                    self.idx_to_vector[idx] = [] 
                    self.idx_to_nrc_vectors[idx] = []
                    no_matches_count += 1
                
                matches_lengths.append(len(self.idx_to_match[idx]))      

        # Store the tags emotion vectors
        self.tag_to_vector = {self.idx_to_query[idx] : value for idx, value in self.idx_to_vector.items()
                              if len(value) > 0 and np.sum(value) > 0}

        # Compute some statistics
        avg_matches_lengths = sum(matches_lengths) / nb_queries 
        prop_no_matches = no_matches_count / nb_queries
        prop_exacts_matches = exacts_matches_count / nb_queries
        prop_matches = matches_count / nb_queries

        print("Initial number of tags: ", len(self.queries))
        print("Number of tags with emotional vector (non null): ", len(self.tag_to_vector))
        print(f"Percentage of matches: {100*prop_matches:.2f} %")
        print(f"Percentage of no matches: {100*prop_no_matches:.2f} %")
        print(f"Percentage of exact matches: {100*prop_exacts_matches:.2f} %")
        print(f"Average number of matches: {avg_matches_lengths:.2f}")


    # ============================== #
    # | CROSS SOURCE NORMALIZATION | #
    # ============================== #    
    
    def normalize_occurences(self, df) : 
        """Normalize occurences of tracks across each source, using min-max scaling.

        Parameters:
            - df (pd.DataFrame): DataFrame containing the tags and their occurences
        
        Returns:
            - df (pd.DataFrame): DataFrame with normalized occurences
        """

        # Remove tags with no associated emotion vector 
        tags = list(self.tag_to_vector.keys())
        df = df[df['tag'].isin(tags)]

        # Normalize by dividing by the maximal occurence in the source
        total_occurences = df.groupby("source")['count'].max().to_dict()
        df['normalized_count'] = df.apply(lambda x : x['count']/total_occurences[x['source']], axis=1)

        # For 'Rate your music', assign as occurence the average count of tags for a given track  
        avg_occurences = df.groupby(["spotify_id"])['normalized_count'].mean().to_dict()
        df['average_count'] = df["spotify_id"].apply(lambda x : avg_occurences[x])
        df['normalized_count'] = df.apply(lambda x : x['average_count'] if x['source'] == "Rate your music" else x['normalized_count'], axis=1)
        df.drop(columns = ['average_count'], inplace=True)

        return df


    # ============================== #
    # |   INTER-RATER AGREEMENT    | #
    # ============================== #   
     
    def get_icc_score(self, df) :
        """Compute ICC score on the dataset, a measure of inter-rater agreement between ratings of raters (tags) on subjects (emotions)

        Parameters : 
            - df (pd.DataFrame) : dataframe containing tags, their occurences and their emotions for a given track
        
        Returns : 
            - icc_s (float) : ICC score
        """

        # Only keep emotions of each tag and reformat to have a matrix where :
        # - rows = emotions (subjects)
        # - columns = tags (raters)
        # - values = occurences * emotion value

        records = []
        for idx, row in df.iterrows():
            counts = row['normalized_count']
            for emotion_idx, emotion_value in enumerate(['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']):
                records.append([idx, counts * row[emotion_value], emotion_idx])

        icc_df = pd.DataFrame(records, columns=['rater', 'rating', 'emotion_id'])

        # Calculate ICC
        icc = pg.intraclass_corr(data=icc_df, targets='emotion_id', raters='rater', ratings='rating')
        icc_s = icc.iloc[3,2]       # ICC(1,k) : one-way random effects for absolute agreement

        return icc_s

    def backward_selection_icc_score(self, df, spotify_id, threeshold_agreement=0.75, min_tags_to_keep=2) :
        """Perform backward selection to iteratively eliminate conflicting tags. Starting with the initial 
        set of tags for a given track, we remove the tag whose removal yields the highest ICC score, 
        until the threshold is attained or only two tags remain.
        
        Parameters : 
            - df (pd.DataFrame) : dataframe containing tags, their occurences and their emotions for a given track
            - spotify_id (str) : Spotify id of the track
            - threeshold_agreement (float) : minimum ICC score to reach
            - min_tags_to_keep (int) : minimum number of tags to keep
        """

        # Initialize variables 
        tag_to_remove = None
        tags = df['tag'].unique()
        icc_s = self.get_icc_score(df)
        old_icc_s = None
        
        # Run until the desired agreement level is reached, using ICC as the metric for inter-rater agreement, or until only two tags remain, 
        # or until there is no improvement in the ICC score 
        while icc_s <= threeshold_agreement and len(tags) > min_tags_to_keep and old_icc_s != icc_s : 

            old_icc_s = icc_s

            # Retrieve the subset of selected tags
            df = df[df['tag'].isin(tags)]

            # For each tag in the current list of tags, compute the new ICC score when this tag is removed
            for tag in tags : 
                # Remove one of the tag
                df_tmp = df[df['tag'] != tag]

                # Compute ICC score
                new_icc_s = self.get_icc_score(df_tmp)
                
                # Store tag and score if the tag removal leads to a higher score
                if new_icc_s >= icc_s : 
                    tag_to_remove = tag
                    icc_s = new_icc_s

            # Remove the tag that maximizes the score
            tags = [tag for tag in tags if tag!= tag_to_remove]

        # Store the final set of tags and the final ICC score
        self.spotify_to_tags[spotify_id] = tags
        self.spotify_to_kappa[spotify_id] = icc_s

    def select_tags_inter_rater_agreement(self, tags_df, path_tags_to_emotions, threeshold_agreement=0.75, min_tags_to_keep=2):
        """
        Select the tags for each track that maximize inter-rater agreement, using backward selection.

        Parameters :
            - tags_df (pd.DataFrame) : DataFrame containing the tags, their occurences
            - path_tags_to_emotions (str) : path to the dataframe containing tags emotion vectors
            - threeshold_agreement (float) : minimum ICC score to reach
            - min_tags_to_keep (int) : minimum number of tags to keep
        """
        # Retrieve emotions associated with each tag
        tags_to_emotions = pd.read_csv(path_tags_to_emotions)
        df = tags_df[['spotify_id', 'tag', 'normalized_count']]
        tags_to_emotions = tags_to_emotions.drop(columns='emotion_vector')
        df = df.merge(tags_to_emotions, how='left', on='tag')

        # For each track, filter the set of tags to maximize inter-rater agreement, using backward selection 
        spotify_ids = df['spotify_id'].unique()
        for spotify_id in tqdm(spotify_ids) : 
            df_tmp = df[df['spotify_id']==spotify_id]
            self.backward_selection_icc_score(df_tmp, spotify_id, threeshold_agreement, min_tags_to_keep)


    # ============================== #
    # |    FINAL EMOTION VECTOR    | #
    # ============================== #    

    def create_emotion_vector_track(self, df) :
        """Create the emotion vector of the track by taking the weighted average
        of the tag's vectors, with normalized occurences as weights. 
        
        Parameters:
            - df (pd.DataFrame): DataFrame containing the tags and their occurences
        
        Returns:
            - df (pd.DataFrame): DataFrame with the emotion vector of each track
        """

        # Retrieve the emotion vector 
        df["vector"] = df["tag"].apply(lambda x : self.tag_to_vector[x])

        for spotify_id in df['spotify_id'].unique() : 
            df_unique = df[df['spotify_id'] == spotify_id]

            # Set the normalized count to 0 for the non-selected tags in inter-agreement step
            tags_to_keep = self.spotify_to_tags[spotify_id]  
            tags = df_unique['tag'].tolist()
            tags_to_remove = [tag for tag in tags if tag not in tags_to_keep]
            
            df_unique['normalized_count'] = df_unique.apply(
                lambda x : 0. if x['tag'] in tags_to_remove else x['normalized_count'],
                axis = 1
            )

            # Retrieve all tag vectors as a matrix, and all occurences as a vector
            emotion_matrix = np.array(df_unique['vector'].tolist())
            occurences = np.array(df_unique['normalized_count'].tolist())
            total_count = np.sum(occurences)
            
            # Weighted average of the emotion matrix
            weighted_sum = np.dot(occurences, emotion_matrix)
            weighted_average = weighted_sum / total_count

            # Assign the final emotion vector to the track
            self.spotify_to_vector[spotify_id] = weighted_average


    # ============================== #
    # |          SAVE DATA         | #
    # ============================== #

    def save_tags_data(self, path_tags_to_emotions, path_tags_to_nrc_matches) :
        """Save all results related to tags : tags emotion vectors, and matches from the NRC Lexicon 
        along with their similarity scores.

        Parameters:
            - path_tags_to_emotions (str) : path to save tags emotion vectors
            - path_tags_to_nrc_matches (str) : path to save matches from the NRC Lexicon 
        """

        # ----- Tags emotion vectors -----
        df = pd.DataFrame.from_dict(self.tag_to_vector, orient='index').reset_index()
        df.columns = ['tag', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        df['emotion_vector'] = df.apply(
            lambda x : f"{str(x['anger'])} {str(x['anticipation'])} {str(x['disgust'])} {str(x['fear'])} {str(x['joy'])} {str(x['sadness'])} {str(x['surprise'])} {str(x['trust'])}", axis=1
        )
        df.set_index('tag', inplace=True)
        df.to_csv(path_tags_to_emotions)

        # ----- Matches from the NRC Lexicon -----
        query_to_matches = []
        for idx, query in self.idx_to_query.items() : 
            for match, score in self.idx_to_match[idx] :
                query_to_matches.append({ 'tag' : query, 'match' : match, 'similarity_score' : score})
        df = pd.DataFrame(query_to_matches)
        df.to_csv(path_tags_to_nrc_matches, index=False)
            
    def save_tracks_data(self, tags_df, path_tracks_to_emotions, path_tracks_to_tags) :
        """Save all results related to tracks : emotion vectors and tags associated with each track.

        Parameters:
            - tags_df (pd.DataFrame): DataFrame containing the tags, their occurences and their emotions
            - path_tracks_to_emotions (str): path to save the emotion vectors of each track
            - path_tracks_to_tags (str): path to save the tags associated with each track
        """
        # ----- Tracks emotion vectors -----
        df = pd.DataFrame.from_dict(self.spotify_to_vector, orient='index').reset_index()
        df.columns = ['spotify_id', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        df['emotion_vector'] = df.apply(
            lambda x : f"{str(x['anger'])} {str(x['anticipation'])} {str(x['disgust'])} {str(x['fear'])} {str(x['joy'])} {str(x['sadness'])} {str(x['surprise'])} {str(x['trust'])}", axis=1
        )
        df.to_csv(path_tracks_to_emotions, index=False)

        # ----- Tags associated with each track -----
        df = tags_df[['spotify_id', 'tag', 'count', 'normalized_count', 'source']]

        # Mark tags that were selected for inter-rater agreement
        dfs = []
        for spotify_id in df['spotify_id'].unique() : 
            tags_to_keep = self.spotify_to_tags[spotify_id]
            df_tmp = df[df['spotify_id'] == spotify_id]
            df_tmp['selected'] = df_tmp['tag'].apply(lambda x : 1 if x in tags_to_keep else 0)
            dfs.append(df_tmp)
        df = pd.concat(dfs)
        
        df.to_csv(path_tracks_to_tags, index=False)

