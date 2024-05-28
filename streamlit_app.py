#streamlit_app.py
import asyncio
import json
import os
from io import StringIO
from PIL import Image
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from video_analysis import download_youtube_video, save_uploaded_video, process_video

# Set page configuration
st.set_page_config(page_title="Video Compliance Analyzer", layout="wide")

# Load Google Cloud logo
google_cloud_logo = Image.open("google_cloud_logo.png")

# Custom CSS styles
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #F8F9FA;
    }
    .sidebar .sidebar-content {
        background-color: #FFFFFF;
    }
    .btn-primary {
        background-color: #4285F4 !important;
        color: #FFFFFF !important;
    }
    .btn-primary:hover {
        background-color: #3367D6 !important;
        color: #FFFFFF !important;
    }
    .violation-card {
        background-color: #FFFFFF;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        padding: 16px;
        margin-bottom: 16px;
        cursor: pointer;
    }
    .violation-card:hover {
        background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def extract_guidelines_from_pdf(uploaded_file):
    """Extract the Film Classification Guidelines from the uploaded PDF file."""
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def main():
    # Sidebar
    st.sidebar.image(google_cloud_logo, width=200)
    st.sidebar.title("Video Compliance Analyzer")
    video_source = st.sidebar.radio("Select Video Source", ("YouTube URL", "Upload Video"))
    
    if video_source == "YouTube URL":
        video_url = st.sidebar.text_input("Enter YouTube Video URL")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4"])

    guidelines_file = st.sidebar.file_uploader("Upload Film Classification Guidelines (PDF)", type=["pdf"])

    process_button = st.sidebar.button("Process Video", key="process_button")

    # Main content
    st.title("Video Compliance Analyzer")
    st.write("The Video Compliance Analyzer is a powerful tool that helps you ensure your videos meet the required standards and regulations. By leveraging advanced AI and machine learning techniques, this application analyzes your video content for potential compliance issues across various categories.")
    st.write("You can either provide a YouTube video URL or upload a video file (.mp4) for analysis. Additionally, you can upload your own Film Classification Guidelines in PDF format. The analyzer will process your video, detect any compliance issues, and provide detailed insights and suggestions for improvement based on the provided guidelines.")

    if guidelines_file is not None:
        guidelines_text = extract_guidelines_from_pdf(guidelines_file)
        st.subheader("Film Classification Guidelines Summary")
        st.write("The uploaded PDF contains the Film Classification Guidelines used for video compliance analysis. The guidelines provide a framework for assessing the appropriateness and permissible content in films based on various categories such as theme, violence, sex, nudity, language, drug and substance abuse, and horror. The guidelines also define the classification ratings (e.g., G, PG, PG13, NC16, M18, R21) and the criteria for each rating.")

    compliance_violation_result = None
    video_path = None
    video_id = None
    video_title = None

    if process_button:
        if guidelines_file is None:
            st.error("Please upload the Film Classification Guidelines (PDF) before processing the video.")
        else:
            guidelines_text = extract_guidelines_from_pdf(guidelines_file)

            if video_source == "YouTube URL":
                if not video_url:
                    st.error("Please enter a YouTube video URL.")
                else:
                    with st.spinner("Processing video..."):
                        video_path, video_id, video_title = download_youtube_video(video_url)
                        if video_path:
                            compliance_violation_result = asyncio.run(process_video(video_path, video_id, video_title, guidelines_text))
            else:
                if not uploaded_file:
                    st.error("Please upload a video file.")
                else:
                    with st.spinner("Processing video..."):
                        video_path, video_id, video_title = save_uploaded_video(uploaded_file)
                        if video_path:
                            compliance_violation_result = asyncio.run(process_video(video_path, video_id, video_title, guidelines_text))

            if compliance_violation_result is not None and video_path is not None:
                st.header("Video Player")
                st.video(video_path)

                st.header("Compliance Violation Detection Result")
                is_compliance_issues = compliance_violation_result["is_compliance_issues"]
                compliance_issues = compliance_violation_result["compliance_issues"]
                final_suggestion = compliance_violation_result["final_suggestion"]
                content_summary = compliance_violation_result["content_summary"]
                speaking_language = compliance_violation_result["speaking_language"]
                content_rating = compliance_violation_result["content_rating"]
                rating_rationale = compliance_violation_result["rating_rationale"]

                if is_compliance_issues and compliance_issues:
                    st.subheader("Compliance Issues")
                    violation_data = []
                    for issue in compliance_issues:
                        violation_data.append({
                            "Timecode": issue["timecode"],
                            "Category": issue["category"],
                            "Description": issue["description"],
                            "Threshold": issue["threshold"],
                            "Action": f'<button class="violation-card" data-timecode="{issue["timecode"]}">Jump to Timecode</button>'
                        })
                    violation_df = pd.DataFrame(violation_data)
                    violation_table = violation_df[["Timecode", "Category", "Description", "Threshold", "Action"]].reset_index(drop=True)
                    violation_table["Action"] = violation_table["Action"].apply(lambda x: f'<div>{x}</div>')
                    st.write(violation_table.to_html(escape=False, index=False), unsafe_allow_html=True)
                else:
                    st.subheader("No compliance issues found.")

                st.subheader("Content Summary")
                st.write(content_summary)

                st.subheader("Speaking Language")
                st.write(speaking_language)

                st.subheader("Content Rating Suggestion")
                st.write(f"Based on the analysis and the provided Film Classification Guidelines, the suggested content rating for this video is: **{content_rating}**")
                if rating_rationale:
                    st.write(f"**Rationale:** {rating_rationale}")

                st.subheader("Final Suggestion")
                st.write(final_suggestion)

                # Display JSON result
                st.subheader("JSON Result")
                st.json(compliance_violation_result)

            else:
                st.error("Failed to process the video. Please try again.")

    # JavaScript code to handle violation card click events
    js_code = """
    <script>
    document.addEventListener("DOMContentLoaded", function () {
        var violationCards = document.getElementsByClassName("violation-card");
        for (var i = 0; i < violationCards.length; i++) {
            violationCards[i].addEventListener("click", function (event) {
                var timecode = event.currentTarget.getAttribute("data-timecode");
                var video = document.querySelector("video");
                var parts = timecode.split(":");
                var seconds = parseInt(parts[0]) * 60 * 60 + parseInt(parts[1]) * 60 + parseInt(parts[2]);
                video.currentTime = seconds;
            });
        }
    });
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

    


# import asyncio
# import json
# import os
# from PIL import Image
# import streamlit as st
# import pandas as pd
# from video_analysis import download_youtube_video, save_uploaded_video, process_video

# # Set page configuration
# st.set_page_config(page_title="IMDA Compliance Analyzer", layout="wide")

# # Load Google Cloud logo
# google_cloud_logo = Image.open("google_cloud_logo.png")

# # Custom CSS styles
# st.markdown(
#     """
#     <style>
#     .reportview-container {
#         background-color: #F8F9FA;
#     }
#     .sidebar .sidebar-content {
#         background-color: #FFFFFF;
#     }
#     .btn-primary {
#         background-color: #4285F4 !important;
#         color: #FFFFFF !important;
#     }
#     .btn-primary:hover {
#         background-color: #3367D6 !important;
#         color: #FFFFFF !important;
#     }
#     .violation-card {
#         background-color: #FFFFFF;
#         border-radius: 4px;
#         box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
#         padding: 16px;
#         margin-bottom: 16px;
#         cursor: pointer;
#     }
#     .violation-card:hover {
#         background-color: #F5F5F5;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# def main():
#     # Sidebar
#     st.sidebar.image(google_cloud_logo, width=200)
#     st.sidebar.title("IMDA Compliance Analyzer")
#     video_source = st.sidebar.radio("Select Video Source", ("YouTube URL", "Upload Video"))
    
#     if video_source == "YouTube URL":
#         video_url = st.sidebar.text_input("Enter YouTube Video URL")
#     else:
#         uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4"])

#     chunk_duration = 300  # Set chunk duration to 300 seconds (5 minutes)
#     process_button = st.sidebar.button("Process Video", key="process_button")

#     # Main content
#     st.title("IMDA Compliance Analyzer")
#     st.write("The IMDA Compliance Analyzer is a powerful tool that helps you ensure your videos meet the IMDA (Infocomm Media Development Authority) standards and regulations. By leveraging advanced AI and machine learning techniques, this application analyzes your video content for potential compliance issues, including violence, nudity, and profanity.")
#     st.write("You can either provide a YouTube video URL or upload a video file (.mp4) for analysis. The analyzer will process your video, detect any compliance issues, and provide detailed insights and suggestions for improvement.")

#     imda_violation_results = None
#     video_path = None

#     if process_button:
#         if video_source == "YouTube URL":
#             if not video_url:
#                 st.error("Please enter a YouTube video URL.")
#             else:
#                 with st.spinner("Processing video..."):
#                     video_path, video_id, video_title = download_youtube_video(video_url)
#                     if video_path:
#                         imda_violation_results, video_path = asyncio.run(process_video(video_path, video_title, chunk_duration))
#         else:
#             if not uploaded_file:
#                 st.error("Please upload a video file.")
#             else:
#                 with st.spinner("Processing video..."):
#                     video_path, video_title = save_uploaded_video(uploaded_file)
#                     if video_path:
#                         imda_violation_results, video_path = asyncio.run(process_video(video_path, video_title, chunk_duration))

#         if imda_violation_results is not None and video_path is not None:
#             st.header("Video Player")
#             st.video(video_path)

#             st.header("IMDA Violation Detection Results")
#             violation_data = []
#             content_rating = None
#             for result in imda_violation_results:
#                 if result:
#                     chunk_file = result["chunk_file"]
#                     is_compliance_issues = result["is_compliance_issues"]
#                     compliance_issues = result["compliance_issues"]
#                     final_suggestion = result["final_suggestion"]
#                     content_summary = result["content_summary"]
#                     speaking_language = result["speaking_language"]
#                     content_rating = result["content_rating"]

#                     if is_compliance_issues and compliance_issues:
#                         for issue in compliance_issues:
#                             violation_data.append({
#                                 "Chunk": chunk_file,
#                                 "Timecode": issue["timecode"],
#                                 "Category": issue["category"],
#                                 "Description": issue["description"],
#                                 "Threshold": issue["threshold"],
#                                 "Action": f'<button class="violation-card" data-timecode="{issue["timecode"]}">Jump to Timecode</button>'
#                             })

#             violation_df = pd.DataFrame(violation_data)

#             if not violation_df.empty:
#                 st.subheader("Violation Table")
#                 violation_table = violation_df[["Timecode", "Category", "Description", "Threshold", "Action"]].reset_index(drop=True)
#                 violation_table["Action"] = violation_table["Action"].apply(lambda x: f'<div>{x}</div>')
#                 st.write(violation_table.to_html(escape=False, index=False), unsafe_allow_html=True)
#             else:
#                 st.subheader("No violations found in the video.")

#             if content_rating:
#                 st.subheader("Content Rating Suggestion")
#                 st.write(f"Based on the analysis, the suggested content rating for this video is: **{content_rating}**")

#             # Download results
#             results_dir = os.path.join('results', video_title)
#             output_json_path = os.path.join(results_dir, f'{video_title}_imda_violation_results.json')
#             with open(output_json_path, "r") as file:
#                 st.download_button(
#                     label="Download Results",
#                     data=file,
#                     file_name=f"{video_title}_imda_violation_results.json",
#                     mime="application/json"
#                 )

#         else:
#             st.error("Failed to process the video. Please try again.")

#     # JavaScript code to handle violation card click events
#     js_code = """
#     <script>
#     document.addEventListener("DOMContentLoaded", function () {
#         var violationCards = document.getElementsByClassName("violation-card");
#         for (var i = 0; i < violationCards.length; i++) {
#             violationCards[i].addEventListener("click", function (event) {
#                 var timecode = event.currentTarget.getAttribute("data-timecode");
#                 var video = document.querySelector("video");
#                 var parts = timecode.split(":");
#                 var seconds = parseInt(parts[0]) * 60 + parseInt(parts[1]);
#                 video.currentTime = seconds;
#             });
#         }
#     });
#     </script>
#     """
#     st.markdown(js_code, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()