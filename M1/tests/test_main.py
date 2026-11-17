import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

import main


class TutorApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_secret_key = main.SECRET_KEY
        self.original_algorithm = main.ALGORITHM
        main.SECRET_KEY = "test-secret-key-long-enough-for-hs256-tests"
        main.ALGORITHM = "HS256"
        self.client = TestClient(
            main.app,
            headers={"token": main._create_access_token("test-student")},
        )

    def tearDown(self) -> None:
        main.SECRET_KEY = self.original_secret_key
        main.ALGORITHM = self.original_algorithm

    def test_ai_endpoint_requires_authentication(self) -> None:
        client = TestClient(main.app)

        response = client.post(
            "/generate_text",
            json={"topic": "Neural networks", "student_level": "beginner"},
        )

        self.assertEqual(response.status_code, 401)

    def test_ai_endpoint_rejects_invalid_token(self) -> None:
        client = TestClient(main.app, headers={"token": "invalid"})

        response = client.post(
            "/generate_text",
            json={"topic": "Neural networks", "student_level": "beginner"},
        )

        self.assertEqual(response.status_code, 401)

    def test_homepage_is_served(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("AI Tutor", response.text)

    def test_health_reports_configuration_without_exposing_credentials(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertIn("ai_service_configured", response.json())
        self.assertNotIn("GEMINI_API_KEY", response.text)

    @patch(
        "main._generate_content",
        return_value=SimpleNamespace(
            text=json.dumps(
                {
                    "topic": "Neural networks",
                    "lesson_content": "A lesson",
                    "key_terms": [],
                    "example": "An example",
                    "practice_questions": ["Question 1", "Question 2", "Question 3"],
                }
            )
        ),
    )
    def test_generate_text_returns_model_json(
        self, generate_content: MagicMock
    ) -> None:
        response = self.client.post(
            "/generate_text",
            json={
                "topic": "Neural networks",
                "course": "Artificial Intelligence",
                "student_level": "beginner",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["topic"], "Neural networks")
        generate_content.assert_called_once()

    @patch("main._generate_content", side_effect=RuntimeError("provider detail"))
    def test_generate_text_hides_provider_error_from_client(
        self, generate_content: MagicMock
    ) -> None:
        response = self.client.post(
            "/generate_text",
            json={"topic": "Neural networks", "student_level": "beginner"},
        )

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("provider detail", response.text)
        generate_content.assert_called_once()

    def test_generate_text_rejects_blank_topic(self) -> None:
        response = self.client.post(
            "/generate_text",
            json={"topic": "  ", "student_level": "beginner"},
        )

        self.assertEqual(response.status_code, 422)

    def test_signup_stores_a_password_hash(self) -> None:
        users = SimpleNamespace(
            create_index=AsyncMock(),
            find_one=AsyncMock(return_value=None),
            insert_one=AsyncMock(),
        )

        with patch.object(main, "user_collection", users):
            response = self.client.post(
                "/signup",
                json={"username": "student", "password": "safe-password"},
            )

        self.assertEqual(response.status_code, 201)
        stored_user = users.insert_one.await_args.args[0]
        self.assertNotEqual(stored_user["password"], "safe-password")
        self.assertTrue(main.pwd_context.verify("safe-password", stored_user["password"]))

    def test_signin_returns_a_valid_access_token(self) -> None:
        password_hash = main.pwd_context.hash("safe-password")
        users = SimpleNamespace(
            find_one=AsyncMock(
                return_value={"username": "student", "password": password_hash}
            )
        )

        with patch.object(main, "user_collection", users):
            response = self.client.post(
                "/signin",
                json={"username": "student", "password": "safe-password"},
            )

        self.assertEqual(response.status_code, 200)
        claims = main.jwt.decode(
            response.json()["access_token"],
            "test-secret-key-long-enough-for-hs256-tests",
            algorithms=["HS256"],
        )
        self.assertEqual(claims["username"], "student")

    @patch(
        "main._generate_content",
        return_value=SimpleNamespace(
            text=json.dumps(
                {
                    "image_analysis": "A diagram",
                    "key_points": ["Point one"],
                    "related_concepts": ["Concept one"],
                }
            )
        ),
    )
    def test_analyze_image_returns_model_json(
        self, generate_content: MagicMock
    ) -> None:
        response = self.client.post(
            "/analyze_image",
            data={"question": "What does this diagram show?"},
            files={"file": ("diagram.png", b"test image bytes", "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["image_analysis"], "A diagram")
        generate_content.assert_called_once()

    def test_analyze_image_rejects_unsupported_file_type(self) -> None:
        response = self.client.post(
            "/analyze_image",
            data={"question": "What is this?"},
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)

    def test_analyze_image_rejects_oversized_file(self) -> None:
        response = self.client.post(
            "/analyze_image",
            data={"question": "What is this?"},
            files={
                "file": (
                    "large.png",
                    b"x" * (main.MAX_IMAGE_BYTES + 1),
                    "image/png",
                )
            },
        )

        self.assertEqual(response.status_code, 413)

    @patch(
        "main._generate_content",
        return_value=SimpleNamespace(
            text=json.dumps(
                {
                    "evaluation": "The answer is accurate.",
                    "score": 8,
                    "feedback": "Add one more supporting detail.",
                }
            )
        ),
    )
    def test_evaluate_audio_returns_evaluation(
        self, generate_content: MagicMock
    ) -> None:
        response = self.client.post(
            "/evaluate_audio_answer",
            data={"question": "What is photosynthesis?"},
            files={"file": ("answer.mp3", b"audio bytes", "audio/mpeg")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["score"], 8)
        generate_content.assert_called_once()

    def test_evaluate_audio_rejects_unsupported_file_type(self) -> None:
        response = self.client.post(
            "/evaluate_audio_answer",
            data={"question": "What is photosynthesis?"},
            files={"file": ("answer.txt", b"not audio", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)

    @patch(
        "main._generate_content",
        return_value=SimpleNamespace(
            text=json.dumps(
                {
                    "evaluation": "The answer is accurate.",
                    "score": 11,
                    "feedback": "Keep practicing.",
                }
            )
        ),
    )
    def test_evaluate_audio_rejects_out_of_range_score(
        self, generate_content: MagicMock
    ) -> None:
        response = self.client.post(
            "/evaluate_audio_answer",
            data={"question": "What is photosynthesis?"},
            files={"file": ("answer.mp3", b"audio bytes", "audio/mpeg")},
        )

        self.assertEqual(response.status_code, 502)
        generate_content.assert_called_once()

    def test_evaluate_audio_rejects_oversized_file(self) -> None:
        response = self.client.post(
            "/evaluate_audio_answer",
            data={"question": "What is photosynthesis?"},
            files={
                "file": (
                    "large.mp3",
                    b"x" * (main.MAX_AUDIO_BYTES + 1),
                    "audio/mpeg",
                )
            },
        )

        self.assertEqual(response.status_code, 413)

    @patch(
        "main._generate_content",
        return_value=SimpleNamespace(
            text=json.dumps(
                {
                    "video_analysis": "The video explains photosynthesis.",
                    "key_points": ["Plants use light energy."],
                    "mcq_questions": [
                        {
                            "question": f"Question {number}?",
                            "options": ["A", "B", "C", "D"],
                            "answer": "A",
                        }
                        for number in range(1, 4)
                    ],
                }
            )
        ),
    )
    def test_analyze_video_returns_three_four_option_mcqs(
        self, generate_content: MagicMock
    ) -> None:
        response = self.client.post(
            "/analyze_video",
            data={"question": "What process is shown?"},
            files={"file": ("lesson.mp4", b"video bytes", "video/mp4")},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["video_analysis"], "The video explains photosynthesis.")
        self.assertEqual(len(payload["mcq_questions"]), 3)
        for mcq in payload["mcq_questions"]:
            self.assertEqual(len(mcq["options"]), 4)
            self.assertIn(mcq["answer"], mcq["options"])
        generate_content.assert_called_once()

    @patch(
        "main._generate_content",
        return_value=SimpleNamespace(
            text=json.dumps(
                {
                    "video_analysis": "Analysis",
                    "key_points": ["Important point"],
                    "mcq_questions": [
                        {
                            "question": "Question?",
                            "options": ["A", "B", "C"],
                            "answer": "A",
                        }
                    ],
                }
            )
        ),
    )
    def test_analyze_video_rejects_invalid_mcq_structure(
        self, generate_content: MagicMock
    ) -> None:
        response = self.client.post(
            "/analyze_video",
            data={"question": "What process is shown?"},
            files={"file": ("lesson.mp4", b"video bytes", "video/mp4")},
        )

        self.assertEqual(response.status_code, 502)
        generate_content.assert_called_once()

    def test_analyze_video_rejects_unsupported_file_type(self) -> None:
        response = self.client.post(
            "/analyze_video",
            data={"question": "What process is shown?"},
            files={"file": ("notes.txt", b"not a video", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)

    def test_analyze_video_rejects_oversized_file(self) -> None:
        response = self.client.post(
            "/analyze_video",
            data={"question": "What process is shown?"},
            files={
                "file": (
                    "large.mp4",
                    b"x" * (main.MAX_VIDEO_BYTES + 1),
                    "video/mp4",
                )
            },
        )

        self.assertEqual(response.status_code, 413)

    @patch(
        "main._generate_content",
        return_value=SimpleNamespace(
            text=json.dumps(
                {
                    "scan_type": "study",
                    "detected_content": "A biology diagram",
                    "extracted_text": "Cell",
                    "explanation": "The diagram shows a cell.",
                    "key_points": ["Cells are basic units of life."],
                    "final_answer": "A cell",
                    "practice_questions": ["What is a cell?"],
                }
            )
        ),
    )
    def test_visual_scan_returns_study_result(
        self, generate_content: MagicMock
    ) -> None:
        response = self.client.post(
            "/visual_scan",
            data={"scan_type": "study", "question": ""},
            files={"file": ("diagram.png", b"test image bytes", "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["scan_type"], "study")
        self.assertEqual(response.json()["detected_content"], "A biology diagram")
        generate_content.assert_called_once()

    def test_visual_scan_rejects_invalid_mode(self) -> None:
        response = self.client.post(
            "/visual_scan",
            data={"scan_type": "unknown"},
            files={"file": ("diagram.png", b"test image bytes", "image/png")},
        )

        self.assertEqual(response.status_code, 400)

    def test_visual_scan_rejects_oversized_image(self) -> None:
        response = self.client.post(
            "/visual_scan",
            data={"scan_type": "study"},
            files={
                "file": (
                    "large.png",
                    b"x" * (main.MAX_IMAGE_BYTES + 1),
                    "image/png",
                )
            },
        )

        self.assertEqual(response.status_code, 413)

    @patch(
        "main._generate_content",
        return_value=SimpleNamespace(
            text=json.dumps(
                {
                    "code_overview": "Prints a greeting.",
                    "line_by_line_explanation": "The print call writes text.",
                    "logic_and_working": "The program executes one print statement.",
                    "errors_and_bugs": "No errors found.",
                    "corrected_code": "print('hello')",
                    "time_complexity": "O(1)",
                    "space_complexity": "O(1)",
                    "learning_tip": "Use clear variable names.",
                }
            )
        ),
    )
    def test_explain_code_returns_structured_explanation(
        self, generate_content: MagicMock
    ) -> None:
        response = self.client.post(
            "/explain_code",
            json={"language": "Python", "code": "print('hello')"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["code_overview"], "Prints a greeting.")
        generate_content.assert_called_once()

    def test_explain_code_rejects_blank_code(self) -> None:
        response = self.client.post(
            "/explain_code",
            json={"language": "Python", "code": "  "},
        )

        self.assertEqual(response.status_code, 422)

    def test_missing_api_key_returns_service_unavailable(self) -> None:
        original_client = main.ai_client
        main.ai_client = None
        try:
            response = self.client.post(
                "/generate_text",
                json={"topic": "Neural networks", "student_level": "beginner"},
            )
        finally:
            main.ai_client = original_client

        self.assertEqual(response.status_code, 503)
        self.assertIn("GEMINI_API_KEY", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
