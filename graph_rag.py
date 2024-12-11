from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
import json
from typing import List, Dict, Any


# Typing for Neo4j connection parameters
class Neo4jConnectionParams:
    def __init__(self, url: str, username: str, password: str):
        self.url = url
        self.username = username
        self.password = password


def sanitize_relationship_type(rel_type: str) -> str:
    """
    Sanitize relationship type by replacing invalid characters with underscores.

    Args:
        rel_type (str): The relationship type string to sanitize.

    Returns:
        str: The sanitized relationship type string.
    """
    return ''.join(char if char.isalnum() or char == '_' else '_' for char in rel_type)


def create_nodes_from_json(nodes: List[Dict[str, Any]], session: Any) -> None:
    """
    Creates nodes in the Neo4j graph from a list of nodes.

    Args:
        nodes (List[Dict[str, Any]]): List of nodes with 'key' and 'word' fields.
        session (neo4j.Session): Neo4j session to run the query.
    """
    for node in nodes:
        query = """
        CREATE (n:Node {key: $key, word: $word})
        """
        params = {
            'key': node['key'],
            'word': node['word']
        }
        session.run(query, params)


def create_relationships_from_json(edges: List[Dict[str, Any]], session: Any) -> None:
    """
    Creates relationships between nodes in Neo4j based on edges in the JSON data.

    Args:
        edges (List[Dict[str, Any]]): List of edges containing source, target, and edge type.
        session (neo4j.Session): Neo4j session to run the query.
    """
    for edge in edges:
        sanitized_edge = sanitize_relationship_type(edge['edge'])
        query = f"""
        MERGE (a {{name: $source}})
        MERGE (b {{name: $target}})
        MERGE (a)-[r:{sanitized_edge}]->(b)
        SET r.weight = $weight
        RETURN a.name AS source, b.name AS target, type(r) AS relationship, r.weight AS weight
        """
        params = {
            "source": edge['source'],
            "target": edge['target'],
            "weight": edge['weight']
        }
        session.run(query, params)


def load_json_data(file_path: str) -> Dict[str, Any]:
    """
    Load JSON data from a file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed JSON data.
    """
    with open(file_path, 'r') as file:
        return json.load(file)


def connect_to_neo4j(params: Neo4jConnectionParams) -> GraphDatabase.driver:
    """
    Establish a connection to the Neo4j database.

    Args:
        params (Neo4jConnectionParams): Connection parameters for Neo4j.

    Returns:
        GraphDatabase.driver: Neo4j driver object.
    """
    return GraphDatabase.driver(params.url, auth=(params.username, params.password))


def refresh_neo4j_schema(graph: Neo4jGraph) -> None:
    """
    Refresh the schema of the Neo4j database.

    Args:
        graph (Neo4jGraph): The Neo4j graph object.
    """
    graph.refresh_schema()


def get_neo4j_schema(session: Any) -> None:
    """
    Retrieves and prints the schema visualization from Neo4j.

    Args:
        session (neo4j.Session): Neo4j session to run the query.
    """
    schema_query = """
    CALL db.schema.visualization()
    """
    result = session.run(schema_query)
    for record in result:
        print(record)


def main() -> None:
    """
    Main function that runs the entire process of loading data, creating nodes and relationships,
    and querying the graph.
    """
    # Neo4j connection parameters
    neo4j_params = Neo4jConnectionParams(
        url="your_neo4j_url",
        username="your_neo4j_username",
        password="your_neo4j_password"
    )

    # Load JSON data
    json_data = load_json_data('/content/drive/MyDrive/1725969437.json')
    nodes = json_data['nodes']
    edges = json_data['edges']

    # Connect to Neo4j
    driver = connect_to_neo4j(neo4j_params)
    session = driver.session()

    # Create nodes and relationships
    create_nodes_from_json(nodes, session)
    create_relationships_from_json(edges, session)

    # Refresh schema and retrieve the schema visualization
    graph = Neo4jGraph(url=neo4j_params.url, username=neo4j_params.username, password=neo4j_params.password)
    refresh_neo4j_schema(graph)
    get_neo4j_schema(session)

    # Create the GraphCypherQAChain with a valid LLM
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # Uncomment if LLM is defined
    chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, validate_cypher=True, verbose=True, allow_dangerous_requests=True)

    # Query the graph using the chain
    query = "What is JAK1?"
    response = chain.invoke({"query": query})
    print(response)


if __name__ == "__main__":
    main()
