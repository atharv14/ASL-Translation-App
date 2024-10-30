import os
import time
import logging
import json
import requests
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_word_list(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def search_asl_signs(word):
    encoded_word = word.replace(' ', '-')
    url = f"https://www.signasl.org/sign/{encoded_word}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        videos = []
        for video in soup.find_all('video'):
            source = video.find('source')
            if source and source.get('src'):
                videos.append({
                    'word': word,
                    'url': source['src']
                })
        return videos
    except requests.RequestException as e:
        logging.error(f"Error fetching {word}: {e}")
        return []

def create_dataset(words, max_videos_per_word=5):
    dataset = []
    for i, word in enumerate(words):
        logging.info(f"Searching for '{word}' ({i+1}/{len(words)})")
        videos = search_asl_signs(word)
        logging.info(f"Found {len(videos)} videos for '{word}'")
        # dataset.extend(videos[:max_videos_per_word])
        dataset.extend(videos)
        time.sleep(1)  # Rate limiting
    logging.info(f"Total videos collected: {len(dataset)}")
    return dataset

def save_dataset(dataset, filepath):
    with open(filepath, 'w') as f:
        json.dump(dataset, f, indent=2)
    logging.info(f"Dataset saved to {filepath}")

def main():
    words_list = load_word_list("MSASL_classes.json")

    output_dir = 'asl_data'
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Starting data collection...")
    dataset = create_dataset(words_list)

    logging.info("Data collection completed.")
    logging.info(f"Total items in dataset: {len(dataset)}")

    # Save the dataset
    dataset_filepath = os.path.join(output_dir, 'asl_dataset.json')
    save_dataset(dataset, dataset_filepath)

    # Log a sample of the data
    for item in dataset[:5]:  # Log first 5 items as a sample
        logging.info(f"Word: {item['word']}, URL: {item['url']}")

if __name__ == "__main__":
    main()
    