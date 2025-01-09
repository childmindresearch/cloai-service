"""Convert trivial JSON schemas to dynamic Pydantic models.

Used to send pydantic models over the wire.
Simple placeholder until LLM apis support JSON schema directly.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, create_model

_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": None,
}


def _convert_property_type(prop_schema: Dict[str, Any]) -> tuple:
    """Convert JSON Schema property type to Python/Pydantic type."""
    if "type" not in prop_schema:
        return (Any, ...)

    prop_type = prop_schema["type"]
    is_required = prop_schema.get("required", True)
    default_value = prop_schema.get("default", ... if is_required else None)

    if prop_type == "array":
        items = prop_schema.get("items", {})
        if "type" in items:
            item_type, _ = _convert_property_type(items)
            return (List[item_type], default_value)  # type: ignore
        return (List[Any], default_value)

    if prop_type == "object":
        nested_properties = prop_schema.get("properties", {})
        model_name = prop_schema.get("title", "NestedModel")

        nested_model = create_model_from_schema(
            {
                "type": "object",
                "properties": nested_properties,
                "required": prop_schema.get("required", []),
                "title": model_name,
            }
        )
        return (nested_model, default_value)

    if isinstance(prop_type, list):
        # Handle multiple types (union type)
        types = [_TYPE_MAPPING.get(t, Any) for t in prop_type if t != "null"]
        if len(types) == 1:
            return (
                (Optional[types[0]], default_value)
                if "null" in prop_type
                else (types[0], default_value)
            )
        union_type = Union[tuple(types)]  # type: ignore
        return (
            (Optional[union_type], default_value)
            if "null" in prop_type
            else (union_type, default_value)
        )

    python_type = _TYPE_MAPPING.get(prop_type, Any)
    return (python_type, default_value)


def create_model_from_schema(schema: Dict[str, Any]) -> type[BaseModel]:
    """Create a Pydantic model from a JSON Schema."""
    if schema.get("type") != "object":
        raise ValueError("Root schema must be of type 'object'")

    properties = schema.get("properties", {})
    required = schema.get("required", [])
    model_name = schema.get("title", "GeneratedModel")

    field_definitions = {}

    for prop_name, prop_schema in properties.items():
        python_type, default_value = _convert_property_type(prop_schema)

        field_kwargs = {}
        if "description" in prop_schema:
            field_kwargs["description"] = prop_schema["description"]

        # Handle default value
        if default_value is not ...:
            field_kwargs["default"] = default_value
        elif prop_name not in required:
            field_kwargs["default"] = None

        field_definitions[prop_name] = (python_type, Field(**field_kwargs))

    return create_model(model_name, **field_definitions)
