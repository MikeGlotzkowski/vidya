import argparse
import openai
import argparse
# import nltk
from nltk import tokenize
import requests
# nltk.download('punkt')
import re
import pyshorteners
import spacy
from gtts import gTTS
import os
from moviepy.editor import VideoFileClip, ImageSequenceClip, concatenate_videoclips
import json


api_key_unsplash = "XXX"
api_key_openAI = "XXX"
api_key_pixabay = "XXX"


def make_api_call(phrase):
    openai.api_key = f'{api_key_openAI}'

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are now a very famous YouTube star and assist somebody in making a video."},
            {"role": "user", "content": f"The topic is {phrase}."},
            {"role": "assistant", "content": """
            Generate a script, but under all circumstances keep this format:
            
            [Suggested Title]
            <Title goes here>

            [Summary]
            <A summary of the video goes here>

            [Chapters]
            [Introduction]
            <Introduction to the topic goes here>

            [Main]
            <Here goes the explanation about the topic. This part can be quick, 
            but must include the topic of the video and be explaining the topic very well!>

            [End]
            <A small summary of the topic. Do not recap everything, be concise>
            """}
        ]
    )

    return response['choices'][0]['message']['content']


def create_video_with_voiceover(audio_file, image_folder, output_file):
    # Read the audio file
    audio_clip = VideoFileClip(audio_file)

    # Get the video length in seconds
    video_length = audio_clip.duration

    # Get a list of image files in alphabetical order
    image_files = sorted(os.listdir(image_folder))
    image_clip_list = []

    for image_file in image_files:
        # Read each image and create a video clip with the same duration as the audio
        image_path = os.path.join(image_folder, image_file)
        image_clip = ImageSequenceClip([image_path], durations=[video_length])
        image_clip_list.append(image_clip)

    # Concatenate all image clips to create the final video
    final_video = concatenate_videoclips(image_clip_list, method="compose")

    # Add the audio to the video
    final_video = final_video.set_audio(audio_clip)

    # Save the final video with voiceover
    final_video.write_videofile(
        output_file, codec='libx264', audio_codec='aac', fps=24)


def save_images_from_text(text, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define a regular expression pattern to match image URLs
    image_url_pattern = r'Image URL: (\S+)'

    # Find all matches of image URLs in the text
    image_urls = re.findall(image_url_pattern, text)

    # Download and save the images
    for idx, image_url in enumerate(image_urls, start=1):
        image_filename = os.path.join(
            output_folder, f"{idx}__{os.path.basename(image_url)}")
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(image_filename, 'wb') as file:
                file.write(response.content)
            print(f"Image {idx} saved as {image_filename}")
        else:
            print(f"Failed to download image {idx}: {image_url}")


def text_to_speech(text, audio_file):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(audio_file)


def shorten_url(url):
    try:
        # Increase the timeout to 5 seconds
        s = pyshorteners.Shortener(timeout=5)
        return s.tinyurl.short(url)
    except (requests.exceptions.Timeout, requests.exceptions.RequestException):
        print("Error: Failed to fetch the shortened URL.")
        return url  # Return the original URL as a fallback


def extract_keywords(sentence, limit=1):
    # Load the spaCy English model
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    # Extract relevant keywords from the sentence
    keywords = [token.text for token in doc if token.pos_ in [
        "NOUN", "VERB", "ADJ", "ADV"]]

    return (" ".join(keywords))[:limit]


def fetch_image_url(sentence, provider):
    if provider == "pixabay":
        return fetch_image_url_from_pixabay(sentence)
    if provider == "unsplash":
        return fetch_image_url_from_unsplash(sentence)


def fetch_image_url_from_pixabay(sentence):
    # Use extracted keywords as the search query
    query = extract_keywords(sentence)

    url = "https://pixabay.com/api/"
    params = {
        "key": api_key_pixabay,
        "q": query,
        "image_type": "photo",
        "per_page": 3,
    }

    try:
        # Prepare the request object without sending it
        request = requests.Request('GET', url, params=params)
        prepared_request = request.prepare()

        # Print the request details before executing it
        print("Request URL:", prepared_request.url)
        print("Request Headers:", prepared_request.headers)

        # Execute the request and get the response
        response = requests.Session().send(prepared_request)

        # Continue with the rest of your code
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()

        # If we got a result, return the URL of the first image
        if data["hits"]:
            return data["hits"][0]["webformatURL"]
        else:
            return None
    except requests.exceptions.RequestException as e:
        print("Error: Failed to fetch image URL from Pixabay API.")
        print(e)
        return None
    except json.JSONDecodeError as e:
        print("Error: Invalid JSON data received from Pixabay API.")
        print(e)
        return None


def fetch_image_url_from_unsplash(sentence):
    # Use extracted keywords as the search query
    query = extract_keywords(sentence)

    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": query,
        "per_page": 1,
    }
    headers = {"Authorization": f"Client-ID {api_key_unsplash}"}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()

        # If we got a result, return the URL of the first image
        if data["results"]:
            return shorten_url(data["results"][0]["urls"]["small"])
        else:
            return None
    except requests.exceptions.RequestException as e:
        print("Error: Failed to fetch image URL from Unsplash API.")
        print(e)
        return None
    except json.JSONDecodeError as e:
        print("Error: Invalid JSON data received from Unsplash API.")
        print(e)
        return None


def filter_director_parts(text):
    director_pattern = r'\[.*?\]'
    filtered_text = re.sub(director_pattern, '', text)
    return filtered_text.strip()


def save_to_text_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)


def augment_text_with_images(text):
    sentences = tokenize.sent_tokenize(text)
    for i in range(len(sentences)):
        # Skip director parts in square brackets
        if not re.match(r'\[.*\]', sentences[i]):
            image_url = fetch_image_url(sentences[i], "pixabay")
            if image_url:
                sentences[i] = sentences[i] + "\nImage URL: " + image_url
    return "\n".join(sentences)


def main():
    parser = argparse.ArgumentParser(description='Process some strings.')
    parser.add_argument('phrase', type=str, help='A phrase to send to the API')
    args = parser.parse_args()
    audio_file = "output/voiceover.mp3"
    image_folder = "output/pics"
    output_file = "finished.mp4"
    response = make_api_call(args.phrase)
    save_to_text_file("output/script_with_director", response)
    response_wo_director = filter_director_parts(response)
    save_to_text_file("output/script_without_director", response_wo_director)
    response_with_pictures = augment_text_with_images(response)
    save_to_text_file("output/script_with_pictures", response_with_pictures)
    save_images_from_text(response_with_pictures, image_folder)
    text_to_speech(filter_director_parts(response), audio_file)

    create_video_with_voiceover(audio_file, image_folder, output_file)


if __name__ == "__main__":
    main()
