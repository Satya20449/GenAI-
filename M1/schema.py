from pydantic import BaseModel, Field, field_validator


class TextRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="The topic for which the text is requested.",
        examples=["Neural Networks"],
    )
    course: str = Field(
        default="",
        max_length=200,
        description="The course related to the topic.",
        examples=["Artificial Intelligence"],
    )
    student_level: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="The educational level of the student.",
        examples=["beginner"],
    )

    @field_validator("topic", "student_level")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field must not be blank.")
        return value

    @field_validator("course")
    @classmethod
    def normalize_course(cls, value: str) -> str:
        return value.strip()
