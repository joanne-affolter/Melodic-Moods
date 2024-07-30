# Utilizing Listener-Provided Tags for Music Emotion Recognition: A Data-Driven Approach

## Abstract

This work introduces a data-driven approach for assigning emotions to music tracks. Consisting of two distinct phases, our framework enables the creation of synthetic emotion-labeled datasets that can serve both Music Emotion Recognition and Auto-Tagging tasks.
The first phase presents a versatile method for collecting listener-generated verbal data, such as tags and playlist names, from multiple online sources on a large scale. We compiled a dataset of $5,892$ tracks, each associated with textual data from four distinct sources. The second phase leverages Natural Language Processing for representing music-evoked emotions, relying solely on the data acquired during the first phase. By semantically matching user-generated text to a well-known corpus of emotion-labelled English words, we are ultimately able to represent each music track as an 8-dimensional vector that captures the emotions perceived by listeners. Our method departs from conventional labeling techniques:
instead of defining emotions as generic ``mood tags'' found on social platforms, we leverage a refined psychological model drawn from Plutchik's theory \cite{Plutchik80}, which appears more intuitive than the extensively used Valence-Arousal model.

## Project Structure

Below is the tree-like structure of the project directory, detailing all included files and folders:

```
root/
├── data/
│ ├── corpus_embeddings.pt 
│ ├── tags_embeddings.pt 
│ ├── tracks_tags.csv 
│ └── NRC-Emotion-Lexicon-Wordlevel-v0.92.txt 
│
├── dataset/
│ ├── original/
│ │ ├── tags_to_emotions.csv
│ │ ├── tags_to_nrc_matches.csv
│ │ ├── tracks_to_emotions.csv
│ │ ├── tracks_to_tags.csv
│ │ └── metadata.csv
│ │
│ ├── balanced/
│ │ ├── tags_to_emotions.csv
│ │ ├── tags_to_nrc_matches.csv
│ │ ├── tracks_to_emotions.csv
│ │ ├── tracks_to_tags.csv
│ │ └── metadata.csv
│ │
│ └──
│
├── pyplutchik/
│ └── * (modified library files)
│
├── Emotion_Attribution.ipynb
├── Results.ipynb
├── utils.py
└── requirements.txt
```

### Data

- `corpus_embeddings.pt`, `tags_embeddings.pt` : files containing pre-computed Sentence-BERT Embeddings of tags (queries) and corpus (words from NRC Lexicon)
- `tracks_tags.csv`: File with refined tags following the data cleaning procedure of the data collection stage.
    - | spotify_id | artist | title | genre | count | source | tag |  
- `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` : NRC Lexicon 

### Dataset (`dataset/original/` and `dataset/balanced/`)

- `tags_to_emotions.csv`: Emotion vectors of tags in the dataset.
    - | tag  | anger | anticipation | disgust | fear | joy | sadness | surprise | trust | emotion_vector |
- `tags_to_nrc_matches.csv`: Matched words from the NRC Lexicon for each tag.
    - | tag  | match | similarity_score |
- `tracks_to_emotions.csv`: Emotion vectors of tracks in the dataset.
    - | spotify_id  | anger | anticipation | disgust | fear | joy | sadness | surprise | trust | emotion_vector |
- `tracks_to_tags.csv`: Tags of tracks in the dataset, along with their occurrences and sources
    - | spotify_id  | tag | count | normalized_count | source |
- `metadata.csv`: Metadata about tracks in the dataset, retrieved from Spotify.
    - | spotify_id  | name | artist | genre | release_date | popularity | preview_url | cover_image | 

The `balanced/` directory mirrors the structure of `original/`, tailored to provide a balanced subset of the dataset.

### Notebooks and code 

- `Emotion_Attribution.ipynb`: This notebook outlines the main steps for assigning emotion vectors to music tracks. 
- `utils.py`: This file contains the functions used in the `Emotion_Attribution.ipynb` notebook. 
- `Results.ipynb`: This notebook showcases some insights and visualizations from the data.


## Installation and Requirements

To set up your environment to work with the dataset, follow these steps:

1. Navigate to the directory in your terminal.
2. Install the required Python libraries using pip:

```bash
python3 -m venv myenv

source myenv/bin/activate       #macOS and Linux
.\myenv\Scripts\activate        #Windows

pip install -r requirements.txt
```
3. Open the `Emotion_Attribution.ipynb` notebook (or `Results.ipynb`), located in the root folder.
4. Select the virtual environment `myenv` as kernel
5. Run the notebook. 
