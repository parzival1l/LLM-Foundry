import os
import yaml
import instructor
from typing import Optional, Any, get_type_hints
from pydantic import BaseModel, Field, create_model
from openai import OpenAI
import argparse

with open("configs/key_config.yaml", 'r') as file:
    api_yaml = yaml.safe_load(file)
file.close()

os.environ['OPENAI_API_KEY'] = api_yaml['openai_api_key']
os.environ['LANGCHAIN_TRACING_V2'] = api_yaml['langsmith']['tracking']
os.environ["LANGCHAIN_API_KEY"] = api_yaml['langsmith']['api_key']
os.environ["LANGCHAIN_PROJECT"] = api_yaml['langsmith']['project_name']

class LLM_call:
    def __init__(self, base_config_path: str, schema_path: str):
        self.base_config = self.load_config(base_config_path)
        self.schema_config = self.load_config(schema_path)
        self.openai_client = instructor.from_openai(OpenAI())
        self.prompt_template = self.base_config['prompt_template']
        self.type_mapping = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "Optional[str]": Optional[str],
            "Optional[int]": Optional[int],
            "Optional[float]": Optional[float],
            "Optional[bool]": Optional[bool],
            "Any": Any,
        }

    def load_config(self, path: str) -> dict:
        with open(path, 'r') as file:
            schema = yaml.safe_load(file)
        return schema

    def create_pydantic_model(self, class_name: str, schema: dict) -> BaseModel:
        fields = {}
        for field_name, field_info in schema.items():
            field_type_str = field_info.get('type')
            field_description = field_info.get('description', '')

            # Resolve the type using the type_mapping
            field_type = self.type_mapping.get(field_type_str, Any)

            # Add the field to the fields dictionary with a FieldInfo
            fields[field_name] = (field_type, Field(description=field_description))

        # Dynamically create the Pydantic model using create_model
        return create_model(class_name, **fields)

    def test_dynamic_model(self, schema_yaml: dict) -> None:
            # Create and use the Pydantic model
        for class_name, schema in schema_yaml.items():
            model = self.create_pydantic_model(class_name, schema)
            print(model.schema())

            # Example usage of the model
            # example_data = {"name": "John Doe", "age": 30, "email": "john@example.com"}
            example_data = {
                "ingredient": "Tomato",
                "quantity": 2.5,
                "process": "chopped"
                }
            instance = model(**example_data)
            print(instance)


    def call_instructor(self, response_model: BaseModel, input_string: str) -> str:

          user_info = self.openai_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            temperature=self.base_config['openai_hyperparameters']['temperature'], #0.05,
                max_tokens=self.base_config['openai_hyperparameters']['max_tokens'], #1250,
                top_p=self.base_config['openai_hyperparameters']['top_p'], #0.05,
                frequency_penalty=self.base_config['openai_hyperparameters']['frequency_penalty'], #0.1,
                presence_penalty=self.base_config['openai_hyperparameters']['presence_penalty'], #1,
            response_model=response_model,
            messages=[{"role": "user", "content": self.prompt_template.format(document_segment=input_string)}],
        )
          return user_info.dict()

    def extract_structured_data(self, schema_yaml_path: str, input_string: str) -> dict:
        schema = self.load_config(schema_yaml_path)
        for class_name, schema in schema.items():
            _model = self.create_pydantic_model(class_name, schema)
        return self.call_instructor(_model, input_string)

def main():
    parser = argparse.ArgumentParser(description="Run LLM with specified configuration files.")
    # parser.add_argument('--base_config_path', type=str, required=True,
    #                     help='Path to the base configuration YAML file.')
    parser.add_argument('--schema_config_path', type=str, required=True,
                        help='Path to the schema configuration YAML file.')
    parser.add_argument('--input_string', type=str, required=False,
                        help='Input sentence to pass to the model.')
    parser.add_argument('--input_file_path', type=str, required=False,
                        help='Path to the schema configuration YAML file.')

    args = parser.parse_args()

    if args.input_string:
        input_string = args.input_string
    elif args.input_file_path:
        with open(args.input_file_path, 'r') as file:
            input_string = file.read()
        file.close()

    llm = LLM_call("configs/base_config.yaml", args.schema_config_path)
    print(llm.extract_structured_data(args.schema_config_path, input_string))
    # print(llm.call_instructor(llm.schema_config['response_model'], input_string))

if __name__ == "__main__":
    main()


# python llm_call.py --base_config_path=/path/configpath.yaml --schema_config_path=/path/schema_config.yaml