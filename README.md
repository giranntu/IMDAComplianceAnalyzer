# Video Compliance Analyzer

The Video Compliance Analyzer is a sophisticated tool designed to ensure that your video content adheres to the standards and regulations. It uses advanced artificial intelligence and machine learning techniques to scrutinize video content for potential compliance issues across various categories.

## Features

- **Comprehensive Analysis**: Scrutinizes video content according to Film Classification Guidelines.
- **Issue Detection**: Identifies compliance issues related to themes, violence, sex, nudity, language, drug use, and horror.
- **Insights and Suggestions**: Offers detailed insights and recommendations for content adjustment to meet compliance standards.
- **Flexible Input Options**: Supports video input through YouTube URLs or direct video file uploads.
- **Detailed Reporting**: Generates structured reports detailing timecodes, categories, and descriptions of identified issues.
- **Rating Suggestions**: Proposes a content rating for the video based on classification codes (G, PG, PG13, NC16, M18, R21).

## Prerequisites

- Python version 3.7 or higher
- Google Cloud Platform (GCP) account with appropriate permissions
- Streamlit library
- Google Cloud Storage Python client library
- Vertex AI Python client library

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/giranntu/imda-compliance-analyzer.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Configure your GCP project and set up the necessary credentials.

4. Update the PROJECT_ID, LOCATION, and BUCKET_NAME in the video_analysis.py file with your specific GCP project details.

## Usage

1. Launch the Streamlit application by running the following command in the terminal:
    ```bash
    streamlit run streamlit_app.py
    ```

2. Open the application in a web browser at http://localhost:8501.

3. Select the video source (YouTube URL or upload a video file) and upload the Film Classification Guidelines PDF.

4. Analyze the video by clicking the "Process Video" button.

5. Review the detailed analysis results, including identified compliance issues, content summaries, languages spoken, and suggested content ratings.

## Deployment

Deploy the Video Compliance Analyzer to Google Cloud Run with the following steps:

1. Build the Docker image:
    ```bash
    docker build --no-cache -t imda-compliance-analyzer .
    ```

2. Tag the Docker image:
    ```bash
    docker tag imda-compliance-analyzer gcr.io/your-project-id/imda-compliance-analyzer
    ```

3. Push the Docker image to the container registry:
    ```bash
    docker push gcr.io/your-project-id/imda-compliance-analyzer
    ```

4. Deploy the application on Cloud Run:
    ```bash
    gcloud run deploy imda-compliance-analyzer --image gcr.io/your-project-id/imda-compliance-analyzer --platform managed --region your-region --allow-unauthenticated
    ```

5. Access the deployed application using the provided URL.

## Contributing

We welcome contributions! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is available under the MIT License.
