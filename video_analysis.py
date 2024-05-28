#video_analysis.py
import asyncio
import json
import os
import re
import random
import string
from google.cloud import storage
from google.api_core import exceptions as gcp_exceptions
from pytube import YouTube
from vertexai.preview.generative_models import GenerativeModel, Part
from urllib.parse import urlparse, parse_qs
from PyPDF2 import PdfReader

from vertexai.generative_models import (
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Image,
    Part,
)

from pydantic import BaseModel, ValidationError
from typing import List, Optional
import json

class ComplianceIssue(BaseModel):
    timecode: str
    category: str
    description: str
    threshold: int

class IMDAResult(BaseModel):
    is_compliance_issues: bool
    compliance_issues: Optional[List[ComplianceIssue]] = []
    final_suggestion: str
    content_summary: str
    speaking_language: str
    content_rating: str
    rating_rationale: str

# Set up GCP project and bucket details
PROJECT_ID = "rust-ry"
LOCATION = "us-central1"
BUCKET_NAME = "gemini-video-analysis"
BUCKET_URI = f"gs://{BUCKET_NAME}"

# Initialize GCP clients
storage_client = storage.Client(project=PROJECT_ID)

def create_bucket_if_not_exists(bucket_name):
    """Create a new bucket if it doesn't exist."""
    try:
        bucket = storage_client.get_bucket(bucket_name)
        print(f"Bucket {bucket_name} already exists.")
    except gcp_exceptions.NotFound:
        bucket = storage_client.create_bucket(bucket_name, location=LOCATION)
        print(f"Bucket {bucket_name} created in location {LOCATION}.")
    return bucket

def get_video_id(video_url):
    """Extract the video ID from a YouTube URL."""
    query = urlparse(video_url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            p = parse_qs(query.query)
            return p['v'][0]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]
    return None

def generate_video_id():
    """Generate a random 4-digit video ID."""
    characters = string.ascii_letters + string.digits
    video_id = ''.join(random.choice(characters) for _ in range(4))
    return video_id

def download_youtube_video(video_url):
    """Download a video from YouTube."""
    try:
        video_id = generate_video_id()
        yt = YouTube(video_url)
        video_title = yt.title
        video_filename = re.sub(r'[^\w\-_\. ]', '_', video_title) + ".mp4"  # Replace special characters with underscore
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if stream:
            print(f"Downloading video: {video_url}")
            full_video_dir = os.path.join("full_video", video_id)
            os.makedirs(full_video_dir, exist_ok=True)
            video_path = os.path.join(full_video_dir, video_filename)
            stream.download(output_path=full_video_dir, filename=video_filename)
            print(f"Video downloaded: {video_path}")
            return video_path, video_id, video_title
        else:
            print("No suitable stream found for downloading the video.")
            return None, None, None
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None, None, None

def save_uploaded_video(uploaded_file):
    """Save the uploaded video file."""
    try:
        video_id = generate_video_id()
        video_filename = uploaded_file.name
        video_filename = re.sub(r'[^\w\-_\. ]', '_', video_filename)  # Replace special characters with underscore
        upload_dir = os.path.join("uploaded_videos", video_id)
        os.makedirs(upload_dir, exist_ok=True)  # Create the directory if it doesn't exist
        video_path = os.path.join(upload_dir, video_filename)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        print(f"Video saved: {video_path}")
        return video_path, video_id, video_filename
    except Exception as e:
        print(f"Error saving uploaded video: {str(e)}")
        return None, None, None

def upload_video_to_gcs(video_path, bucket_name):
    """Upload a video to Google Cloud Storage."""
    try:
        bucket = create_bucket_if_not_exists(bucket_name)
        blob_name = os.path.basename(video_path)
        blob_name = re.sub(r'\s+', '_', blob_name)  # Replace spaces with underscores in the blob name
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(video_path)
        print(f"Video uploaded to GCS: {blob_name}")
        return f"gs://{bucket_name}/{blob_name}"  # Return the GCS URI
    except Exception as e:
        print(f"Error uploading video to GCS: {str(e)}")
        return None

def extract_guidelines_from_pdf(pdf_path):
    """Extract the IMDA Film Classification Guidelines from the PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text

async def detect_imda_violations(prompt, video_uri, max_retries=1, initial_delay=1, temperature=0.5):
    """Detect IMDA violations in a video using Gemini."""
    retry_count = 0
    while retry_count < max_retries:
        try:
            model = GenerativeModel("gemini-1.5-pro-preview-0409")
            mime_type = "video/mp4"
            parameters = {"mime_type": mime_type, "uri": video_uri}
            video_part = Part.from_uri(**parameters)

            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                candidate_count=1,
                max_output_tokens=8190,
            )

            # Set safety settings
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            }

            response = model.generate_content(
                [prompt, video_part],
                generation_config=generation_config,
                stream=False,
                safety_settings=safety_settings,
            )

            # Handle streaming response
            if isinstance(response, GenerationResponse):
                result_text = response.text
            else:
                result = [r.text for r in response]
                result_text = "\n".join(result)

            # Preprocess the output to remove specific markdown code blocks if present
            processed_text = result_text.strip().replace("```json", "").replace("```", "").strip()

            print(f"Processed response from Gemini:\n{processed_text}\n")
            return processed_text

        except Exception as e:
            if "Quota exceeded" in str(e) and retry_count < max_retries - 1:
                retry_count += 1
                delay = initial_delay * (2 ** retry_count) + random.uniform(0, 1)
                print(f"Quota exceeded. Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
            elif "PROHIBITED_CONTENT" in str(e):
                print("Prohibited content detected. Skipping video.")
                return None
            else:
                print(f"Error detecting IMDA violations: {str(e)}")
                if temperature > 0 and retry_count < max_retries - 1:
                    temperature = 0
                    retry_count += 1
                    print(f"Retrying with temperature set to 0.")
                    await asyncio.sleep(1)
                else:
                    print("Skipping IMDA violation detection.")
                    return None

    print("Max retries exceeded. Skipping IMDA violation detection.")
    return None

def parse_imda_violation_result(result_text: str):
    """Parse the IMDA violation detection result and extract structured data."""
    try:
        # Load the JSON text into a Python dictionary
        data = json.loads(result_text)

        # Create an instance of IMDAResult using the dictionary
        result = IMDAResult(**data)

        return result.dict()

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON data: {str(e)}")
        return None
    except ValidationError as e:
        print(f"Validation error in IMDA result structure: {str(e)}")
        return None
    except Exception as e:
        print(f"General error when parsing IMDA violation detection result: {str(e)}")
        return None



async def process_video(video_path, video_id, video_title, guidelines_text):
    """Process a video and detect compliance violations."""
    # Step 1: Upload the video to GCS
    video_uri = upload_video_to_gcs(video_path, BUCKET_NAME)
    if video_uri is None:
        print("Video upload to GCS failed. Skipping subsequent steps.")
        return None

    # Step 2: Detect compliance violations in the video
    prompt = f"""
    Objective:
    Conduct a compliance review based on the provided Film Classification Guidelines to ensure the video meets the standards/regulations.

    Film Classification Guidelines:
    {guidelines_text}

    Content Review:
    Assess the content for compliance with the guidelines on appropriateness and permissible content, as outlined in the provided guidelines.
    Specifically, focus on identifying the following categories of compliance issues:
    1. Theme
    2. Violence
    3. Sex
    4. Nudity
    5. Language
    6. Drug and Substance Abuse (Including Psychoactive Substance Abuse)
    7. Horror

    Expected Output:
    Provide a structured report with detailed descriptions of any compliance issues, including:
    - Timecodes for scenes containing compliance issues.
    - The category of the compliance issue (e.g., Theme, Violence, Sex, Nudity, Language, Drug and Substance Abuse, Horror).
    - A brief summary of the content in the video.
    - The primary speaking language(s) in the video.
    - A final suggestion for the content rating of the video (e.g., G, PG, PG13, NC16, M18, R21) based on the guidelines.
    - A brief rationale for the suggested content rating, referencing specific sections from the guidelines.

    Special Instructions:
    Follow the provided Film Classification Guidelines closely when detecting potential violations that could affect the content rating or distribution.
    Rate each compliance issue on a threshold scale from 1-5 (5 indicates the highest severity or confidence).
    Provide a clear and concise content rating suggestion based on the classification code (G, PG, PG13, NC16, M18, R21) and include a brief rationale for the suggestion, referencing relevant sections from the guidelines.

    Please return the result in the following JSON format:
    {{
        "is_compliance_issues": (true or false),
        "compliance_issues": [
            {{
                "timecode": "HH:MM:SS",
                "category": "Category of the compliance issue",
                "description": "Detailed description of the issue",
                "threshold": (1-5)
            }}
        ],
        "final_suggestion": "A brief summary suggestion (around 10 words)",
        "content_summary": "A brief summary of the content in the video",
        "speaking_language": "The primary speaking language(s) in the video",
        "content_rating": "The suggested content rating for the video (e.g., G, PG, PG13, NC16, M18, R21)",
        "rating_rationale": "A brief rationale for the suggested content rating, referencing specific sections from the guidelines"
    }}

    If you are unsure about any information, please do not make assumptions. Return the result in the specified JSON format.

    Video: {video_title}
    """

    result_text = await detect_imda_violations(prompt, video_uri)
    if result_text:
        result = parse_imda_violation_result(result_text)
        if result:
            result['video_title'] = video_title  # Include video_title in the result dictionary
            print(f"Compliance Violation Detection for {video_title}:")
            print(json.dumps(result, indent=2))
            return result
        else:
            print(f"Failed to parse compliance violation detection result for {video_title}")
            return None
    else:
        print(f"Failed to detect compliance violations for {video_title}")
        return None


# import asyncio
# import json
# import os
# import re
# import subprocess
# import time
# import random
# from google.cloud import storage
# from google.api_core import exceptions as gcp_exceptions
# from pytube import YouTube
# from vertexai.preview.generative_models import GenerativeModel, Part
# from urllib.parse import urlparse, parse_qs
# from PyPDF2 import PdfReader

# from vertexai.generative_models import (
#     GenerationConfig,
#     GenerationResponse,
#     GenerativeModel,
#     HarmBlockThreshold,
#     HarmCategory,
#     Image,
#     Part,
# )

# from pydantic import BaseModel, ValidationError
# from typing import List, Optional
# import json

# class ComplianceIssue(BaseModel):
#     timecode: str
#     category: str
#     description: str
#     threshold: int

# class IMDAResult(BaseModel):
#     is_compliance_issues: bool
#     compliance_issues: Optional[List[ComplianceIssue]] = []
#     final_suggestion: str
#     content_summary: str
#     speaking_language: str
#     content_rating: str
#     rating_rationale: str

# # Set up GCP project and bucket details
# PROJECT_ID = "rust-ry"
# LOCATION = "us-central1"
# BUCKET_NAME = "gemini-video-analysis"
# BUCKET_URI = f"gs://{BUCKET_NAME}"

# # Initialize GCP clients
# storage_client = storage.Client(project=PROJECT_ID)

# def create_bucket_if_not_exists(bucket_name):
#     """Create a new bucket if it doesn't exist."""
#     try:
#         bucket = storage_client.get_bucket(bucket_name)
#         print(f"Bucket {bucket_name} already exists.")
#     except gcp_exceptions.NotFound:
#         bucket = storage_client.create_bucket(bucket_name, location=LOCATION)
#         print(f"Bucket {bucket_name} created in location {LOCATION}.")
#     return bucket

# def get_video_id(video_url):
#     """Extract the video ID from a YouTube URL."""
#     query = urlparse(video_url)
#     if query.hostname == 'youtu.be':
#         return query.path[1:]
#     if query.hostname in ('www.youtube.com', 'youtube.com'):
#         if query.path == '/watch':
#             p = parse_qs(query.query)
#             return p['v'][0]
#         if query.path[:7] == '/embed/':
#             return query.path.split('/')[2]
#         if query.path[:3] == '/v/':
#             return query.path.split('/')[2]
#     return None

# def download_youtube_video(video_url):
#     """Download a video from YouTube."""
#     try:
#         video_id = get_video_id(video_url)
#         if video_id is None:
#             raise ValueError("Invalid YouTube URL")

#         yt = YouTube(video_url)
#         video_title = yt.title
#         video_filename = re.sub(r'[^\w\-_\. ]', '_', video_title) + ".mp4"  # Replace special characters with underscore
#         stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
#         if stream:
#             print(f"Downloading video: {video_url}")
#             full_video_dir = os.path.join("full_video", video_id)
#             os.makedirs(full_video_dir, exist_ok=True)
#             video_path = os.path.join(full_video_dir, video_filename)
#             stream.download(output_path=full_video_dir, filename=video_filename)
#             print(f"Video downloaded: {video_path}")
#             return video_path, video_id, video_title
#         else:
#             print("No suitable stream found for downloading the video.")
#             return None, None, None
#     except Exception as e:
#         print(f"Error downloading video: {str(e)}")
#         return None, None, None

# def save_uploaded_video(uploaded_file):
#     """Save the uploaded video file."""
#     try:
#         video_filename = uploaded_file.name
#         video_filename = re.sub(r'[^\w\-_\. ]', '_', video_filename)  # Replace special characters with underscore
#         upload_dir = "uploaded_videos"
#         os.makedirs(upload_dir, exist_ok=True)  # Create the directory if it doesn't exist
#         video_path = os.path.join(upload_dir, video_filename)
#         with open(video_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         print(f"Video saved: {video_path}")
#         return video_path, video_filename
#     except Exception as e:
#         print(f"Error saving uploaded video: {str(e)}")
#         return None, None

# def upload_video_to_gcs(video_path, bucket_name):
#     """Upload a video to Google Cloud Storage."""
#     try:
#         bucket = create_bucket_if_not_exists(bucket_name)
#         blob_name = os.path.basename(video_path)
#         blob_name = re.sub(r'\s+', '_', blob_name)  # Replace spaces with underscores in the blob name
#         blob = bucket.blob(blob_name)
#         blob.upload_from_filename(video_path)
#         print(f"Video uploaded to GCS: {blob_name}")
#         return f"gs://{bucket_name}/{blob_name}"  # Return the GCS URI
#     except Exception as e:
#         print(f"Error uploading video to GCS: {str(e)}")
#         return None

# def split_video_into_chunks(video_path, output_dir, chunk_duration):
#     """Split a video into chunks of specified duration."""
#     try:
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)

#         base_name = os.path.splitext(os.path.basename(video_path))[0]
#         base_name = re.sub(r'\s+', '_', base_name)  # Replace spaces with underscores in the base name
#         command = f"ffmpeg -i '{video_path}' -c copy -map 0 -f segment -segment_time {chunk_duration} -reset_timestamps 1 '{output_dir}/{base_name}_%02d.mp4'"
#         subprocess.call(command, shell=True)

#         print(f"Video split into chunks and saved in: {output_dir}")
#         return output_dir
#     except Exception as e:
#         print(f"Error splitting video into chunks: {str(e)}")
#         return None

# def extract_guidelines_from_pdf(pdf_path):
#     """Extract the IMDA Film Classification Guidelines from the PDF file."""
#     reader = PdfReader(pdf_path)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text()

#     return text

# async def detect_imda_violations(prompt, video_uri, max_retries=1, initial_delay=1, temperature=0.5):
#     """Detect IMDA violations in a video chunk using Gemini."""
#     retry_count = 0
#     while retry_count < max_retries:
#         try:
#             model = GenerativeModel("gemini-1.5-pro-preview-0409")
#             mime_type = "video/mp4"
#             parameters = {"mime_type": mime_type, "uri": video_uri}
#             video_part = Part.from_uri(**parameters)

#             generation_config = GenerationConfig(
#                 temperature=temperature,
#                 top_p=0.95,
#                 candidate_count=1,
#                 max_output_tokens=8190,
#             )

#             # Set safety settings
#             safety_settings = {
#                 HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#                 HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#                 HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#                 HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#             }

#             response = model.generate_content(
#                 [prompt, video_part],
#                 generation_config=generation_config,
#                 stream=False,
#                 safety_settings=safety_settings,
#             )

#             # Handle streaming response
#             if isinstance(response, GenerationResponse):
#                 result_text = response.text
#             else:
#                 result = [r.text for r in response]
#                 result_text = "\n".join(result)

#             # Preprocess the output to remove specific markdown code blocks if present
#             processed_text = result_text.strip().replace("```json", "").replace("```", "").strip()

#             print(f"Processed response from Gemini:\n{processed_text}\n")
#             return processed_text

#         except Exception as e:
#             if "Quota exceeded" in str(e) and retry_count < max_retries - 1:
#                 retry_count += 1
#                 delay = initial_delay * (2 ** retry_count) + random.uniform(0, 1)
#                 print(f"Quota exceeded. Retrying in {delay:.2f} seconds...")
#                 await asyncio.sleep(delay)
#             elif "PROHIBITED_CONTENT" in str(e):
#                 print("Prohibited content detected. Skipping chunk.")
#                 return None
#             else:
#                 print(f"Error detecting IMDA violations: {str(e)}")
#                 if temperature > 0 and retry_count < max_retries - 1:
#                     temperature = 0
#                     retry_count += 1
#                     print(f"Retrying with temperature set to 0.")
#                     await asyncio.sleep(1)
#                 else:
#                     print("Skipping IMDA violation detection.")
#                     return None

#     print("Max retries exceeded. Skipping IMDA violation detection.")
#     return None

# def parse_imda_violation_result(result_text: str, chunk_index: int, chunk_duration: int):
#     """Parse the IMDA violation detection result and extract structured data."""
#     try:
#         # Load the JSON text into a Python dictionary
#         data = json.loads(result_text)

#         # Create an instance of IMDAResult using the dictionary
#         result = IMDAResult(**data)

#         # Convert chunk-specific timecodes to global timecodes in the video
#         if result.compliance_issues:
#             for issue in result.compliance_issues:
#                 chunk_minutes, chunk_seconds = map(int, issue.timecode.split(":"))
#                 global_seconds = (chunk_index * chunk_duration) + (chunk_minutes * 60) + chunk_seconds
#                 global_minutes, global_seconds = divmod(global_seconds, 60)
#                 issue.timecode = f"{global_minutes:02d}:{global_seconds:02d}"

#         return result.dict()

#     except json.JSONDecodeError as e:
#         print(f"Error parsing JSON data: {str(e)}")
#         return None
#     except ValidationError as e:
#         print(f"Validation error in IMDA result structure: {str(e)}")
#         return None
#     except Exception as e:
#         print(f"General error when parsing IMDA violation detection result: {str(e)}")
#         return None

# def convert_timecode(timecode, chunk_index, chunk_duration):
#     """Convert chunk-specific timecode to global timecode in the video."""
#     chunk_minutes, chunk_seconds = map(int, timecode.split(":"))
#     global_seconds = (chunk_index * chunk_duration) + (chunk_minutes * 60) + chunk_seconds
#     global_minutes, global_seconds = divmod(global_seconds, 60)
#     return f"{global_minutes:02d}:{global_seconds:02d}"

# async def process_video(video_path, video_title, chunk_duration):
#     """Process a video and detect IMDA violations."""
#     # Step 1: Split the video into chunks
#     output_dir = os.path.join("output", video_title)
#     chunks_dir = split_video_into_chunks(video_path, output_dir, chunk_duration)
#     if chunks_dir is None:
#         print("Video splitting into chunks failed. Skipping subsequent steps.")
#         return None, video_path

#     # Step 2: Extract IMDA Film Classification Guidelines from the PDF
#     guidelines_text = extract_guidelines_from_pdf("Film Classification Guidelines 29_Apr_2019.pdf")

#     # Step 3: Process each video chunk and detect IMDA violations
#     imda_violation_results = []
#     chunk_files = sorted(os.listdir(chunks_dir))
#     tasks = []
#     for chunk_index, chunk_file in enumerate(chunk_files):
#         chunk_path = os.path.join(chunks_dir, chunk_file)
#         chunk_uri = upload_video_to_gcs(chunk_path, BUCKET_NAME)
#         if chunk_uri:
#             task = asyncio.create_task(process_video_chunk(chunk_file, chunk_index, chunk_duration, chunk_uri, video_title, guidelines_text))
#             tasks.append(task)
#         else:
#             print(f"Failed to upload chunk {chunk_file} to GCS. Skipping IMDA violation detection.")

#     results = await asyncio.gather(*tasks)
#     imda_violation_results.extend([r for r in results if r])  # Filter out None results

#     # Step 4: Save the IMDA violation detection results to a JSON file
#     output_json_path = "video_output.json"

#     if imda_violation_results:
#         with open(output_json_path, "w") as file:
#             json.dump(imda_violation_results, file, indent=2)
#         print(f"IMDA violation detection results saved to {output_json_path}")
#     else:
#         print("No IMDA violation detection results to save.")

#     return imda_violation_results, video_path


# async def process_video_chunk(chunk_file, chunk_index, chunk_duration, chunk_uri, video_title, guidelines_text):
#     """Process a video chunk and detect IMDA violations."""
#     prompt = f"""
#     Objective:
#     Conduct an IMDA compliance review based on the provided IMDA Film Classification Guidelines to ensure the video meets the standards/regulations.

#     IMDA Film Classification Guidelines:
#     {guidelines_text}

#     Content Review:
#     Assess the content for compliance with IMDA's guidelines on appropriateness and permissible content, as outlined in the provided guidelines.
#     Specifically, focus on identifying the following categories of compliance issues:
#     1. Theme
#     2. Violence
#     3. Sex
#     4. Nudity
#     5. Language
#     6. Drug and Substance Abuse (Including Psychoactive Substance Abuse)
#     7. Horror

#     Expected Output:
#     Provide a structured report with detailed descriptions of any compliance issues, including:
#     - Timecodes for scenes containing compliance issues.
#     - The category of the compliance issue (e.g., Theme, Violence, Sex, Nudity, Language, Drug and Substance Abuse, Horror).
#     - A brief summary of the content in the video chunk.
#     - The primary speaking language(s) in the video chunk.
#     - A final suggestion for the content rating of the video (e.g., G, PG, PG13, NC16, M18, R21) based on the guidelines.
#     - A brief rationale for the suggested content rating, referencing specific sections from the guidelines.

#     Special Instructions:
#     Follow the provided IMDA Film Classification Guidelines closely when detecting potential violations that could affect the content rating or distribution.
#     Rate each compliance issue on a threshold scale from 1-5 (5 indicates the highest severity or confidence).
#     Provide a clear and concise content rating suggestion based on the IMDA classification code (G, PG, PG13, NC16, M18, R21) and include a brief rationale for the suggestion, referencing relevant sections from the guidelines.

#     Please return the result in the following JSON format:
#     {{
#         "is_compliance_issues": (true or false),
#         "compliance_issues": [
#             {{
#                 "timecode": "HH:MM:SS",
#                 "category": "Category of the compliance issue",
#                 "description": "Detailed description of the issue",
#                 "threshold": (1-5
# ) }} ], "final_suggestion": "A brief summary suggestion (around 10 words)", "content_summary": "A brief summary of the content in the video chunk", "speaking_language": "The primary speaking language(s) in the video chunk", "content_rating": "The suggested content rating for the video (e.g., G, PG, PG13, NC16, M18, R21)", "rating_rationale": "A brief rationale for the suggested content rating, referencing specific sections from the guidelines" }}

# Copy code
# If you are unsure about any information, please do not make assumptions. Return the result in the specified JSON format.

# Video: {chunk_file}
# """

#     result_text = await detect_imda_violations(prompt, chunk_uri)
#     if result_text:
#         result = parse_imda_violation_result(result_text, chunk_index, chunk_duration)
#         if result:
#             result['chunk_file'] = chunk_file  # Include chunk_file in the result dictionary
#             result['video_title'] = video_title  # Include video_title in the result dictionary
#             print(f"IMDA Violation Detection for {chunk_file}:")
#             print(json.dumps(result, indent=2))
#             return result
#         else:
#             print(f"Failed to parse IMDA violation detection result for {chunk_file}")
#             return None
#     else:
#         print(f"Failed to detect IMDA violations for {chunk_file}")
#         return None










# import asyncio
# import json
# import os
# import re
# import subprocess
# import time
# import random
# from google.cloud import storage
# from google.api_core import exceptions as gcp_exceptions
# from pytube import YouTube
# from vertexai.preview.generative_models import GenerativeModel, Part
# from urllib.parse import urlparse, parse_qs

# from vertexai.generative_models import (
#     GenerationConfig,
#     GenerationResponse,
#     GenerativeModel,
#     HarmBlockThreshold,
#     HarmCategory,
#     Image,
#     Part,
# )

# from pydantic import BaseModel, ValidationError
# from typing import List, Optional
# import json

# class ComplianceIssue(BaseModel):
#     timecode: str
#     category: str
#     description: str
#     threshold: int

# class IMDAResult(BaseModel):
#     is_compliance_issues: bool
#     compliance_issues: Optional[List[ComplianceIssue]] = []
#     final_suggestion: str
#     content_summary: str
#     speaking_language: str
#     content_rating: str

# # Set up GCP project and bucket details
# PROJECT_ID = "rust-ry"
# LOCATION = "us-central1"
# BUCKET_NAME = "gemini-video-analysis"
# BUCKET_URI = f"gs://{BUCKET_NAME}"

# # Initialize GCP clients
# storage_client = storage.Client(project=PROJECT_ID)

# def create_bucket_if_not_exists(bucket_name):
#     """Create a new bucket if it doesn't exist."""
#     try:
#         bucket = storage_client.get_bucket(bucket_name)
#         print(f"Bucket {bucket_name} already exists.")
#     except gcp_exceptions.NotFound:
#         bucket = storage_client.create_bucket(bucket_name, location=LOCATION)
#         print(f"Bucket {bucket_name} created in location {LOCATION}.")
#     return bucket

# def get_video_id(video_url):
#     """Extract the video ID from a YouTube URL."""
#     query = urlparse(video_url)
#     if query.hostname == 'youtu.be':
#         return query.path[1:]
#     if query.hostname in ('www.youtube.com', 'youtube.com'):
#         if query.path == '/watch':
#             p = parse_qs(query.query)
#             return p['v'][0]
#         if query.path[:7] == '/embed/':
#             return query.path.split('/')[2]
#         if query.path[:3] == '/v/':
#             return query.path.split('/')[2]
#     return None

# def download_youtube_video(video_url):
#     """Download a video from YouTube."""
#     try:
#         video_id = get_video_id(video_url)
#         if video_id is None:
#             raise ValueError("Invalid YouTube URL")

#         yt = YouTube(video_url)
#         video_title = yt.title
#         video_filename = re.sub(r'[^\w\-_\. ]', '_', video_title) + ".mp4"  # Replace special characters with underscore
#         stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
#         if stream:
#             print(f"Downloading video: {video_url}")
#             full_video_dir = os.path.join("full_video", video_id)
#             os.makedirs(full_video_dir, exist_ok=True)
#             video_path = os.path.join(full_video_dir, video_filename)
#             stream.download(output_path=full_video_dir, filename=video_filename)
#             print(f"Video downloaded: {video_path}")
#             return video_path, video_id, video_title
#         else:
#             print("No suitable stream found for downloading the video.")
#             return None, None, None
#     except Exception as e:
#         print(f"Error downloading video: {str(e)}")
#         return None, None, None

# def save_uploaded_video(uploaded_file):
#     """Save the uploaded video file."""
#     try:
#         video_filename = uploaded_file.name
#         upload_dir = "uploaded_videos"
#         os.makedirs(upload_dir, exist_ok=True)  # Create the directory if it doesn't exist
#         video_path = os.path.join(upload_dir, video_filename)
#         with open(video_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         print(f"Video saved: {video_path}")
#         return video_path, video_filename
#     except Exception as e:
#         print(f"Error saving uploaded video: {str(e)}")
#         return None, None

# def upload_video_to_gcs(video_path, bucket_name):
#     """Upload a video to Google Cloud Storage."""
#     try:
#         bucket = create_bucket_if_not_exists(bucket_name)
#         blob_name = os.path.basename(video_path)
#         blob_name = re.sub(r'\s+', '_', blob_name)  # Replace spaces with underscores in the blob name
#         blob = bucket.blob(blob_name)
#         blob.upload_from_filename(video_path)
#         print(f"Video uploaded to GCS: {blob_name}")
#         return f"gs://{bucket_name}/{blob_name}"  # Return the GCS URI
#     except Exception as e:
#         print(f"Error uploading video to GCS: {str(e)}")
#         return None

# def split_video_into_chunks(video_path, output_dir, chunk_duration):
#     """Split a video into chunks of specified duration."""
#     try:
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)

#         base_name = os.path.splitext(os.path.basename(video_path))[0]
#         base_name = re.sub(r'\s+', '_', base_name)  # Replace spaces with underscores in the base name
#         command = f"ffmpeg -i '{video_path}' -c copy -map 0 -f segment -segment_time {chunk_duration} -reset_timestamps 1 '{output_dir}/{base_name}_%02d.mp4'"
#         subprocess.call(command, shell=True)

#         print(f"Video split into chunks and saved in: {output_dir}")
#         return output_dir
#     except Exception as e:
#         print(f"Error splitting video into chunks: {str(e)}")
#         return None

# async def detect_imda_violations(prompt, video_uri, max_retries=1, initial_delay=1, temperature=0.5):
#     """Detect IMDA violations in a video chunk using Gemini."""
#     retry_count = 0
#     while retry_count < max_retries:
#         try:
#             model = GenerativeModel("gemini-1.5-pro-preview-0409")
#             mime_type = "video/mp4"
#             parameters = {"mime_type": mime_type, "uri": video_uri}
#             video_part = Part.from_uri(**parameters)

#             generation_config = GenerationConfig(
#                 temperature=temperature,
#                 top_p=0.95,
#                 candidate_count=1,
#                 max_output_tokens=8190,
#             )

#             # Set safety settings
#             safety_settings = {
#                 HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#                 HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#                 HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#                 HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#             }

#             response = model.generate_content(
#                 [prompt, video_part],
#                 generation_config=generation_config,
#                 stream=False,
#                 safety_settings=safety_settings,
#             )

#             # Handle streaming response
#             if isinstance(response, GenerationResponse):
#                 result_text = response.text
#             else:
#                 result = [r.text for r in response]
#                 result_text = "\n".join(result)

#             # Preprocess the output to remove specific markdown code blocks if present
#             processed_text = result_text.strip().replace("```json", "").replace("```", "").strip()

#             print(f"Processed response from Gemini:\n{processed_text}\n")
#             return processed_text

#         except Exception as e:
#             if "Quota exceeded" in str(e) and retry_count < max_retries - 1:
#                 retry_count += 1
#                 delay = initial_delay * (2 ** retry_count) + random.uniform(0, 1)
#                 print(f"Quota exceeded. Retrying in {delay:.2f} seconds...")
#                 await asyncio.sleep(delay)
#             elif "PROHIBITED_CONTENT" in str(e):
#                 print("Prohibited content detected. Skipping chunk.")
#                 return None
#             else:
#                 print(f"Error detecting IMDA violations: {str(e)}")
#                 if temperature > 0 and retry_count < max_retries - 1:
#                     temperature = 0
#                     retry_count += 1
#                     print(f"Retrying with temperature set to 0.")
#                     await asyncio.sleep(1)
#                 else:
#                     print("Skipping IMDA violation detection.")
#                     return None

#     print("Max retries exceeded. Skipping IMDA violation detection.")
#     return None

# def parse_imda_violation_result(result_text: str, chunk_index: int, chunk_duration: int):
#     """Parse the IMDA violation detection result and extract structured data."""
#     try:
#         # Load the JSON text into a Python dictionary
#         data = json.loads(result_text)

#         # Create an instance of IMDAResult using the dictionary
#         result = IMDAResult(**data)

#         # Convert chunk-specific timecodes to global timecodes in the video
#         if result.compliance_issues:
#             for issue in result.compliance_issues:
#                 chunk_minutes, chunk_seconds = map(int, issue.timecode.split(":"))
#                 global_seconds = (chunk_index * chunk_duration) + (chunk_minutes * 60) + chunk_seconds
#                 global_minutes, global_seconds = divmod(global_seconds, 60)
#                 issue.timecode = f"{global_minutes:02d}:{global_seconds:02d}"

#         return result.dict()

#     except json.JSONDecodeError as e:
#         print(f"Error parsing JSON data: {str(e)}")
#         return None
#     except ValidationError as e:
#         print(f"Validation error in IMDA result structure: {str(e)}")
#         return None
#     except Exception as e:
#         print(f"General error when parsing IMDA violation detection result: {str(e)}")
#         return None

# def convert_timecode(timecode, chunk_index, chunk_duration):
#     """Convert chunk-specific timecode to global timecode in the video."""
#     chunk_minutes, chunk_seconds = map(int, timecode.split(":"))
#     global_seconds = (chunk_index * chunk_duration) + (chunk_minutes * 60) + chunk_seconds
#     global_minutes, global_seconds = divmod(global_seconds, 60)
#     return f"{global_minutes:02d}:{global_seconds:02d}"

# async def process_video(video_path, video_title, chunk_duration):
#     """Process a video and detect IMDA violations."""
#     # Step 1: Split the video into chunks
#     output_dir = os.path.join("output", video_title)
#     chunks_dir = split_video_into_chunks(video_path, output_dir, chunk_duration)
#     if chunks_dir is None:
#         print("Video splitting into chunks failed. Skipping subsequent steps.")
#         return None, video_path

#     # Step 2: Process each video chunk and detect IMDA violations
#     imda_violation_results = []
#     chunk_files = sorted(os.listdir(chunks_dir))
#     tasks = []
#     for chunk_index, chunk_file in enumerate(chunk_files):
#         chunk_path = os.path.join(chunks_dir, chunk_file)
#         chunk_uri = upload_video_to_gcs(chunk_path, BUCKET_NAME)
#         if chunk_uri:
#             task = asyncio.create_task(process_video_chunk(chunk_file, chunk_index, chunk_duration, chunk_uri, video_title))
#             tasks.append(task)
#         else:
#             print(f"Failed to upload chunk {chunk_file} to GCS. Skipping IMDA violation detection.")

#     results = await asyncio.gather(*tasks)
#     imda_violation_results.extend([r for r in results if r])  # Filter out None results

#     # Step 3: Save the IMDA violation detection results to a JSON file
#     results_dir = os.path.join('results', video_title)
#     os.makedirs(results_dir, exist_ok=True)  # Ensure the directory exists
#     output_json_path = os.path.join(results_dir, f'{video_title}_imda_violation_results.json')

#     if imda_violation_results:
#         with open(output_json_path, "w") as file:
#             json.dump(imda_violation_results, file, indent=2)
#         print(f"IMDA violation detection results saved to {output_json_path}")
#     else:
#         print("No IMDA violation detection results to save.")

#     return imda_violation_results, video_path

# async def process_video_chunk(chunk_file, chunk_index, chunk_duration, chunk_uri, video_title):
#     """Process a video chunk and detect IMDA violations."""
#     prompt = f"""
#     Objective:
#     Conduct an IMDA compliance review to ensure the video meets the standards/regulation.

#     Content Review:
#     Assess content for compliance with IMDA's guidelines on appropriateness and permissible content.
#     Specifically, focus on identifying the following categories of compliance issues:
#     1. Violence
#     2. Nudity
#     3. Profanity

#     Expected Output:
#     Provide a structured report with detailed descriptions of any compliance issues, including:
#     - Timecodes for scenes containing compliance issues.
#     - The category of the compliance issue (e.g., Violence, Nudity, Profanity).
#     - A brief summary of the content in the video chunk.
#     - The primary speaking language(s) in the video chunk.
#     - A final suggestion for the content rating of the video (e.g., PG13, M18, R21).

#     Special Instructions:
#     Follow IMDA's latest content guidelines, with a focus on detecting potential violations that could affect content rating or distribution.
#     Rate each compliance issue on a threshold scale from 1-5 (5 indicates the highest severity or confidence).

#     Please return the result in the following JSON format:
#     {{
#         "is_compliance_issues": (true or false),
#         "compliance_issues": [
#             {{
#                 "timecode": "HH:MM:SS",
#                 "category": "Category of the compliance issue",
#                 "description": "Detailed description of the issue",
#                 "threshold": (1-5)
#             }}
#         ],
#         "final_suggestion": "A brief summary suggestion (around 10 words)",
#         "content_summary": "A brief summary of the content in the video chunk",
#         "speaking_language": "The primary speaking language(s) in the video chunk",
#         "content_rating": "The suggested content rating for the video (e.g., PG13, M18, R21)"
#     }}

#     If you are unsure about any information, please do not make assumptions. Return the result in the specified JSON format.

#     Video: {chunk_file}
#     """

#     result_text = await detect_imda_violations(prompt, chunk_uri)
#     if result_text:
#         result = parse_imda_violation_result(result_text, chunk_index, chunk_duration)
#         if result:
#             result['chunk_file'] = chunk_file  # Include chunk_file in the result dictionary
#             result['video_title'] = video_title  # Include video_title in the result dictionary
#             print(f"IMDA Violation Detection for {chunk_file}:")
#             print(json.dumps(result, indent=2))
#             return result
#         else:
#             print(f"Failed to parse IMDA violation detection result for {chunk_file}")
#             return None
#     else:
#         print(f"Failed to detect IMDA violations for {chunk_file}")
#         return None

# import asyncio
# import json
# import os
# import re
# import subprocess
# import time
# import random
# from google.cloud import storage
# from google.api_core import exceptions as gcp_exceptions
# from pytube import YouTube
# from vertexai.preview.generative_models import GenerativeModel, Part

# from vertexai.generative_models import (
#     GenerationConfig,
#     GenerationResponse,
#     GenerativeModel,
#     HarmBlockThreshold,
#     HarmCategory,
#     Image,
#     Part,
# )
# import os
# import tempfile
# import subprocess
# import shutil
# from object_storage import upload_directory

# from pydantic import BaseModel, ValidationError
# from typing import List, Optional
# import json

# class ComplianceIssue(BaseModel):
#     timestamp: str
#     description: str
#     threshold: int

# class IMDAResult(BaseModel):
#     is_compliance_issues: bool
#     compliance_issues: Optional[List[ComplianceIssue]] = []
#     final_suggestion: str
#     content_summary: str
#     cultural_sensitivity_score: float
#     technical_quality_score: float

# # Set up GCP project and bucket details
# PROJECT_ID = "rust-ry"
# LOCATION = "us-central1"
# BUCKET_NAME = "gemini-video-analysis"
# BUCKET_URI = f"gs://{BUCKET_NAME}"

# # Initialize GCP clients
# storage_client = storage.Client(project=PROJECT_ID)

# def create_bucket_if_not_exists(bucket_name):
#     """Create a new bucket if it doesn't exist."""
#     try:
#         bucket = storage_client.get_bucket(bucket_name)
#         print(f"Bucket {bucket_name} already exists.")
#     except gcp_exceptions.NotFound:
#         bucket = storage_client.create_bucket(bucket_name, location=LOCATION)
#         print(f"Bucket {bucket_name} created in location {LOCATION}.")
#     return bucket

# def download_youtube_video(video_url):
#     """Download a video from YouTube."""
#     try:
#         yt = YouTube(video_url)
#         stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
#         if stream:
#             print(f"Downloading video: {video_url}")
#             video_path = stream.download()
#             print(f"Video downloaded: {video_path}")
#             return video_path
#         else:
#             print("No suitable stream found for downloading the video.")
#             return None
#     except Exception as e:
#         print(f"Error downloading video: {str(e)}")
#         return None

# def upload_video_to_gcs(video_path, bucket_name):
#     """Upload a video to Google Cloud Storage."""
#     try:
#         bucket = create_bucket_if_not_exists(bucket_name)
#         blob_name = os.path.basename(video_path)
#         blob_name = re.sub(r'\s+', '_', blob_name)  # Replace spaces with underscores in the blob name
#         blob = bucket.blob(blob_name)
#         blob.upload_from_filename(video_path)
#         print(f"Video uploaded to GCS: {blob_name}")
#         return f"gs://{bucket_name}/{blob_name}"  # Return the GCS URI
#     except Exception as e:
#         print(f"Error uploading video to GCS: {str(e)}")
#         return None

# def split_video_into_chunks(video_path, output_dir, chunk_duration):
#     """Split a video into chunks of specified duration."""
#     try:
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)

#         base_name = os.path.splitext(os.path.basename(video_path))[0]
#         base_name = re.sub(r'\s+', '_', base_name)  # Replace spaces with underscores in the base name
#         command = f"ffmpeg -i '{video_path}' -c copy -map 0 -f segment -segment_time {chunk_duration} -reset_timestamps 1 '{output_dir}/{base_name}_%02d.mp4'"
#         subprocess.call(command, shell=True)

#         print(f"Video split into chunks and saved in: {output_dir}")
#         return output_dir
#     except Exception as e:
#         print(f"Error splitting video into chunks: {str(e)}")
#         return None

# async def detect_imda_violations(prompt, video_uri, max_retries=10, initial_delay=1):
#     """Detect IMDA violations in a video chunk using Gemini."""
#     retry_count = 0
#     while retry_count < max_retries:
#         try:
#             model = GenerativeModel("gemini-1.5-pro-preview-0409")
#             mime_type = "video/mp4"
#             parameters = {"mime_type": mime_type, "uri": video_uri}
#             video_part = Part.from_uri(**parameters)

#             generation_config = GenerationConfig(
#                 temperature=0.5,
#                 top_p=0.95,
#                 candidate_count=1,
#                 max_output_tokens=8190,
#             )

#             # Set safety settings
#             safety_settings = {
#                 HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#                 HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#                 HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#                 HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#             }

#             response = model.generate_content(
#                 [prompt, video_part],
#                 generation_config=generation_config,
#                 stream=False,
#                 safety_settings=safety_settings,
#             )

#             # Handle streaming response
#             if isinstance(response, GenerationResponse):
#                 result_text = response.text
#             else:
#                 result = [r.text for r in response]
#                 result_text = "\n".join(result)

#             # Preprocess the output to remove specific markdown code blocks if present
#             processed_text = result_text.strip().replace("```json", "").replace("```", "").strip()

#             print(f"Processed response from Gemini:\n{processed_text}\n")
#             return processed_text

#         except Exception as e:
#             if "Quota exceeded" in str(e) and retry_count < max_retries - 1:
#                 retry_count += 1
#                 delay = initial_delay * (2 ** retry_count) + random.uniform(0, 1)
#                 print(f"Quota exceeded. Retrying in {delay:.2f} seconds...")
#                 await asyncio.sleep(delay)
#             else:
#                 print(f"Error detecting IMDA violations: {str(e)}")
#                 return None

#     print("Max retries exceeded. Skipping IMDA violation detection.")
#     return None

# def parse_imda_violation_result(result_text: str, chunk_index: int, chunk_duration: int):
#     """Parse the IMDA violation detection result and extract structured data."""
#     try:
#         # Load the JSON text into a Python dictionary
#         data = json.loads(result_text)

#         # Create an instance of IMDAResult using the dictionary
#         result = IMDAResult(**data)

#         # Convert chunk-specific timestamps to global timestamps in the video
#         if result.compliance_issues:
#             for issue in result.compliance_issues:
#                 chunk_minutes, chunk_seconds = map(int, issue.timestamp.split(":"))
#                 global_seconds = (chunk_index * chunk_duration) + (chunk_minutes * 60) + chunk_seconds
#                 global_minutes, global_seconds = divmod(global_seconds, 60)
#                 issue.timestamp = f"{global_minutes:02d}:{global_seconds:02d}"

#         return result.is_compliance_issues, result.compliance_issues, result.final_suggestion, result.content_summary, result.cultural_sensitivity_score, result.technical_quality_score

#     except json.JSONDecodeError as e:
#         print(f"Error parsing JSON data: {str(e)}")
#         return None, None, None, None, None, None
#     except ValidationError as e:
#         print(f"Validation error in IMDA result structure: {str(e)}")
#         return None, None, None, None, None, None
#     except Exception as e:
#         print(f"General error when parsing IMDA violation detection result: {str(e)}")
#         return None, None, None, None, None, None

# async def process_video(video_url, output_dir, chunk_duration):
#     """Process a video and detect IMDA violations."""
#     # Step 1: Download the video from the given URL
#     video_path = download_youtube_video(video_url)

#     # Step 2: Upload the video to GCS
#     if video_path:
#         video_uri = upload_video_to_gcs(video_path, BUCKET_NAME)
#     else:
#         print("Video download failed. Skipping subsequent steps.")
#         return None

#     # Step 3: Split the video into chunks
#     if video_uri:
#         chunks_dir = split_video_into_chunks(video_path, output_dir, chunk_duration)
#     else:
#         print("Video upload to GCS failed. Skipping subsequent steps.")
#         return None

#     # Step 4: Process each video chunk and detect IMDA violations
#     imda_violation_results = []
#     if chunks_dir:
#         chunk_files = sorted(os.listdir(chunks_dir))
#         tasks = []
#         for chunk_index, chunk_file in enumerate(chunk_files):
#             task = asyncio.create_task(process_video_chunk(chunk_file, chunk_index, chunk_duration, chunks_dir))
#             tasks.append(task)

#         results = await asyncio.gather(*tasks)
#         imda_violation_results.extend(results)
#     else:
#         print("Video splitting into chunks failed. Skipping IMDA violation detection.")

#     # Step 5: Save the IMDA violation detection results to a JSON file
#     with open("imda_violation_results.json", "w") as file:
#         json.dump(imda_violation_results, file, indent=2)
#     print("IMDA violation detection results saved to imda_violation_results.json")

#     return imda_violation_results

# async def process_video_chunk(chunk_file, chunk_index, chunk_duration, chunks_dir):
#     """Process a video chunk and detect IMDA violations."""
#     chunk_path = os.path.join(chunks_dir, chunk_file)
#     chunk_uri = upload_video_to_gcs(chunk_path, BUCKET_NAME)
#     if chunk_uri:
#         prompt = f"""
#         Objective:
#         Conduct an IMDA compliance review to ensure the video meets the standards/regulation.

#         Content Review:
#         Assess content for compliance with IMDA's guidelines on appropriateness, cultural sensitivity, and permissible content.

#         Expected Output:
#         Provide a structured report with timestamps : it's detailed descriptions of any issues.

#         Special Instructions:
#         Follow IMDA's latest content and technical guidelines, with a focus on detecting any potential violations that could affect content rating or distribution.
#         Rate each issue into a threshold from 1-5 (5 means the highest confidence/danger, 1 means just a slight warning).
#         Provide a brief summary of the content in the video chunk.
#         Rate the cultural sensitivity of the content on a scale of 0 to 1 (1 being highly sensitive).
#         Rate the technical quality of the video on a scale of 0 to 1 (1 being the highest quality).

#         Please return the result in the following JSON format:
#         {{
#             "is_compliance_issues": (true or false),
#             "compliance_issues": [
#                 {{
#                     "timestamp": "HH:MM:SS",
#                     "description": "Detailed description of the issue",
#                     "threshold": (1-5)
#                 }}
#             ],
#             "final_suggestion": "A brief summary suggestion (around 10 words)",
#             "content_summary": "A brief summary of the content in the video chunk",
#             "cultural_sensitivity_score": (0-1),
#             "technical_quality_score": (0-1)
#         }}

#         If you are not sure about any information, please do not make it up. Return the result in the specified JSON format.

#         Video: {chunk_file}
#         """

#         result_text = await detect_imda_violations(prompt, chunk_uri)
#         if result_text:
#             is_compliance_issues, compliance_issues, final_suggestion, content_summary, cultural_sensitivity_score, technical_quality_score = parse_imda_violation_result(result_text, chunk_index, chunk_duration)
#             print(f"IMDA Violation Detection for {chunk_file}:")
#             print(f"Is Compliance Issues: {is_compliance_issues}")
#             print(f"Compliance Issues: {compliance_issues}")
#             print(f"Final Suggestion: {final_suggestion}")
#             print(f"Content Summary: {content_summary}")
#             print(f"Cultural Sensitivity Score: {cultural_sensitivity_score}")
#             print(f"Technical Quality Score: {technical_quality_score}\n")
#             return chunk_file, is_compliance_issues, compliance_issues, final_suggestion, content_summary, cultural_sensitivity_score, technical_quality_score
#         else:
#             print(f"Failed to detect IMDA violations for {chunk_file}")
#             return chunk_file, None, None, None, None, None, None
#     else:
#         print(f"Failed to upload chunk {chunk_file} to GCS")
#         return chunk_file, None, None, None, None, None, None