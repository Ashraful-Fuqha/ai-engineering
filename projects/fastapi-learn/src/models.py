from pydantic import BaseModel, ConfigDict, Field


class Post(BaseModel):
    id: int
    user_id: int = Field(alias="userId")
    title: str
    body: str

    model_config = ConfigDict(
        populate_by_name=True,
    )
