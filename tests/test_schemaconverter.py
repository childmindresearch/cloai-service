"""Unit tests for the schema converter."""

import pydantic

from cloaiservice.services import schemaconverter


def test_create_model_from_schema_tuple_str() -> None:
    """Tests whether the min/max item fields are interpreted correctly."""

    class Model(pydantic.BaseModel):
        adjectives: tuple[str, ...] = pydantic.Field(..., min_length=1, max_length=2)

    new_model = schemaconverter.create_model_from_schema(Model.model_json_schema())

    assert Model.model_json_schema() == new_model.model_json_schema()
