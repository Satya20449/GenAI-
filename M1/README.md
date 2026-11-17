# AI Tutor API

A small FastAPI application that uses Google's Gemini API to create study material, explain academic images and videos, and evaluate spoken answers. Video analysis also creates three multiple-choice questions. It includes a simple browser interface and automated tests that do not make real Gemini requests.

## Features

- Generate a lesson, key terms, an example, and practice questions from a topic.
- Upload an academic image and ask a question about it.
- Upload an audio answer and get an evaluation, score out of 10, and constructive feedback.
- Upload an academic video and receive an analysis and three MCQs with four options and a correct answer each.
- Scan images in study, solve, diagram, or object-identification mode.
- Return the model's structured JSON response.
- Serve an interface at `/` and interactive API documentation at `/docs`.
- Validate request fields, accepted upload types, and upload size limits.

## Requirements

- Python 3.10 or later.
- A Gemini API key.

## Setup on Windows

Open PowerShell in this project folder and create/activate a virtual environment if you do not already have one:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project folder with your own Gemini API key:

```dotenv
GEMINI_API_KEY=replace_with_your_api_key
CHAT_MODEL=gemini-2.5-flash
```

Keep `.env` private. It is excluded from Git by `.gitignore`; never paste the key into source code, screenshots, or shared messages. If a key has been shared, revoke it and create a replacement in Google AI Studio.

## Run the application

```powershell
python -m uvicorn main:app --reload
```

Open <http://127.0.0.1:8000/> for the interface, <http://127.0.0.1:8000/docs> for Swagger UI, or <http://127.0.0.1:8000/health> to check whether the API key was loaded.

The app starts even if `GEMINI_API_KEY` is missing so the interface and health route remain accessible. AI endpoints return HTTP 503 until the key is configured.

## API

### `POST /generate_text`

Request JSON:

```json
{
  "topic": "Neural networks",
  "course": "Artificial Intelligence",
  "student_level": "beginner"
}
```

The `course` field is optional. The response contains `topic`, `lesson_content`, `key_terms`, `example`, and `practice_questions`.

### `POST /analyze_image`

Send a `multipart/form-data` request with:

- `question`: the question to answer about the image.
- `file`: a JPEG, PNG, or WebP image no larger than 10 MB.

Example with curl:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/analyze_image" `
  -F "question=What does this diagram show?" `
  -F "file=@C:\path\to\diagram.png"
```

The response contains `image_analysis`, `key_points`, and `related_concepts`.

### `POST /evaluate_audio_answer`

Send a `multipart/form-data` request with:

- `question`: the question the student answered.
- `file`: an MP3, WAV, AAC, OGG, or FLAC audio file no larger than 20 MB.

The response contains `evaluation`, `score` (an integer from 0 to 10), and `feedback`.

Example with curl:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/evaluate_audio_answer" `
  -F "question=What is photosynthesis?" `
  -F "file=@C:\path\to\student-answer.mp3"
```

### `POST /analyze_video`

Send a `multipart/form-data` request with:

- `question`: the question to answer about the video.
- `file`: an MP4, AVI, or MOV video no larger than 20 MB.

The response contains `video_analysis`, `key_points`, and `mcq_questions`. The MCQ list contains exactly three objects; each object has a `question`, four `options`, and an `answer` matching one of the options.

Example with curl:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/analyze_video" `
  -F "question=What process is demonstrated in this video?" `
  -F "file=@C:\path\to\lesson.mp4"
```

### `POST /visual_scan`

Send a `multipart/form-data` request with:

- `scan_type`: `study`, `solve`, `diagram`, or `object` (defaults to `study`).
- `question`: an optional question about the image.
- `file`: a JPEG, PNG, or WebP image no larger than 10 MB.

The response contains detected content, extracted text, an explanation, key points,
a final answer, and practice questions.

### `GET /health`

Returns the application status, model name, and whether Gemini is configured. It never returns the API key.

## Run tests

With the project's virtual environment active:

```powershell
python -m unittest discover -s tests -v
```

The tests mock the Gemini response, so they do not require a valid API key or make external requests.
