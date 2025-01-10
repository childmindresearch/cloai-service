import pydantic
from cloaiservice.services import schemaconverter

def test_create_model_from_schema_tuple_str():
    class Model(pydantic.BaseModel):
        adjectives: tuple[str, ...] = pydantic.Field(..., min_length=1, max_length=2)

    newModel = schemaconverter.create_model_from_schema(Model.model_json_schema())

    assert Model.model_json_schema() == newModel.model_json_schema()