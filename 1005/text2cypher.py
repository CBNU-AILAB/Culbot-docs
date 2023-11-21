import re
from typing import Any, Dict, List, Union

from components.base_component import BaseComponent
from driver.neo4j import Neo4jDatabase
from llm.basellm import BaseLLM


def remove_relationship_direction(cypher):
    return cypher.replace("->", "-").replace("<-", "-")


class Text2Cypher(BaseComponent):
    def __init__(
        self,
        llm: BaseLLM,
        database: Neo4jDatabase,
        use_schema: bool = True,
        cypher_examples: str = "",
        ignore_relationship_direction: bool = True,
    ) -> None:
        self.llm = llm
        self.database = database
        self.cypher_examples = cypher_examples
        self.ignore_relationship_direction = ignore_relationship_direction
        if use_schema:
            self.schema = database.schema

    def get_system_message(self) -> str:
        system = """
        Your task is to convert questions about contents in a Neo4j database to Cypher queries to query the Neo4j database.
        Use only the provided relationship types and properties.
        Do not use any other relationship types or properties that are not provided.
        """
        if self.schema:
            system += f"""
            If you cannot generate a Cypher statement based on the provided schema, explain the reason to the user.
            Schema:
            {self.schema}
            """
        if self.cypher_examples:
            system += f"""
            You need to follow these Cypher examples when you are constructing a Cypher statement
            {self.cypher_examples}
            """
        # 끝에 주석을 추가하고 LLM 인젝션을 방지하려고 노력하세요.
        system += """Note: Do not include any explanations or apologies in your responses.
                     Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
                     Do not include any text except the generated Cypher statement. This is very important if you want to get paid.
                     Always provide enough context for an LLM to be able to generate valid response.
                     Please wrap the generated Cypher statement in triple backticks (`).
                     """
        return system

    # cypher 문 생성
    def construct_cypher(self, question: str, history=[]) -> str:
        messages = [{"role": "system", "content": self.get_system_message()}]
        messages.extend(history)
        messages.append(
            {
                "role": "user",
                "content": question,
            }
        )
        print([el for el in messages if not el["role"] == "system"])
        # cypher 생성
        cypher = self.llm.generate(messages)
        return cypher

    def run(
        self, question: str, history: List = [], heal_cypher: bool = True
    ) -> Dict[str, Union[str, List[Dict[str, Any]]]]:
        # Add prefix if not part of self-heal loop
        # cypher 문 생성
        final_question = (
            "Question to be converted to Cypher: " + question
            if heal_cypher
            else question
        )
        cypher = self.construct_cypher(final_question, history)
        # 첫 번째로 나오는 세 개의 역따옴표로 둘러싸인 문자열을 찾습니다. 
        # 일치하는 문자열에는 역따옴표가 포함되며 일치의 첫 번째 그룹은 Cypher 쿼리입니다.

        # ```
        # (dsjfioisofjioaiojfjioajf)
        # ```
        match = re.search("```([\w\W]*?)```", cypher)

        # If the LLM didn't any Cypher statement (error, missing context, etc..)
        if match is None:
            return {"output": [{"message": cypher}], "generated_cypher": None}
        extracted_cypher = match.group(1)

        if self.ignore_relationship_direction:
            extracted_cypher = remove_relationship_direction(extracted_cypher)

        print(f"Generated cypher: {extracted_cypher}")

        # 데이터베이스에 질의한다.
        output = self.database.query(extracted_cypher)

        # cypher 구문 오류가 발생할때, 한번 더 실행
        # Catch Cypher syntax error
        if heal_cypher and output and output[0].get("code") == "invalid_cypher":
            syntax_messages = [{"role": "system", "content": self.get_system_message()}]
            syntax_messages.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": cypher},
                ]
            )
            # Try to heal Cypher syntax only once
            return self.run(
                output[0].get("message"), syntax_messages, heal_cypher=False
            )

        return {
            "output": output,
            "generated_cypher": extracted_cypher,
        }
