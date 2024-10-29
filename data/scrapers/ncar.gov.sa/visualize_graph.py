import networkx as nx
from pyvis.network import Network
from tqdm.auto import tqdm
import json
import hashlib

from scrape import get_id


def hash_json(json_obj):
    # Serialize JSON object with sorted keys
    serialized_json = json.dumps(json_obj, sort_keys=True).encode('utf-8')
    return hashlib.sha256(serialized_json).hexdigest()


def get_id(item):
    return hash_json(
        {
            'title_en': item['title_en'],
            'title_ar': item['title_ar'],
            'number': item['number'],
            'approve_date': item['Approves'][0]['approve_date'],
        }
    )

# https://towardsdatascience.com/visualizing-networks-in-python-d70f4cbeb259
# TODO:
# FIXME: remove things that are already linked


def create_graph(pages_data):
    G = nx.DiGraph()  # Directed graph to represent document relationships

    # TODO: Add nodes and edges to the graph to makre sure they aren't duplicated
    added_node_ids = set()  # Map new_id to node index

    MAX_RELATIONS = 5  # Maximum title_ar of relations to consider
    failures = []
    duplicates_avoided = []
    # Iterate through each page data to add nodes and edges
    for page in tqdm(pages_data['data'][: args.limit], 'Creating Graph'):
        # Add the main document as a node
        try:
            label = page['documentCategory'][0]['name_ar']
        except:
            label = page['title_ar']

        if page['new_id'] in added_node_ids:
            # print("Node already exists in graph", page['new_id'])
            duplicates_avoided.append(page['new_id'])
            continue

        G.add_node(page['new_id'], title=page['title_ar'], size=20, title_en=page['title_en'], label=label)
        added_node_ids.add(page['new_id'])

        # Handle document relations
        if 'documentRelation' in page['documentData']['data']:
            for relation in page['documentData']['data']['documentRelation'][:MAX_RELATIONS]:
                try:
                    relation_new_id = get_id(relation)
                    if relation_new_id in added_node_ids:
                        # print("Node already exists in graph", relation_new_id)
                        duplicates_avoided.append(page['new_id'])
                        continue
                    G.add_node(relation_new_id, title=relation['title_ar'], label=relation['title_ar'])
                    added_node_ids.add(relation_new_id)
                    G.add_edge(page['new_id'], relation_new_id, title='Relation')
                except Exception as e:
                    print(f'ERROR: Skipping relation {relation}', e)
                    failures.append((relation, e))

        # Handle updated documents
        if 'updatedDocument' in page['documentData']['data']:
            for update in page['documentData']['data']['updatedDocument'][:MAX_RELATIONS]:
                try:
                    relation_new_id = get_id(update)
                    G.add_node(relation_new_id, title=update['title_ar'], label=update['title_ar'])
                    G.add_edge(page['new_id'], relation_new_id, title='Update')
                except Exception as e:
                    print(f'ERROR: Skipping relation {relation}', e)
                    failures.append((relation, e))
    print('Failures:', len(failures))
    print('duplicates_avoided:', len(duplicates_avoided))

    return G


def visualize_graph(G):
    nt = Network('800px', '1200px', notebook=True)
    nt.from_nx(G)

    # Customize node and edge styles if needed
    for node, node_attrs in G.nodes(data=True):
        nt.get_node(node)['title'] = node_attrs.get('title', '')
        nt.get_node(node)['label'] = node_attrs.get('label', '')

    # Set options for a better visual experience
    nt.set_options(
        """
    var options = {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "color": {
          "border": "black",
          "background": "skyblue",
          "highlight": {
            "border": "darkblue",
            "background": "lightblue"
          },
          "hover": {
            "border": "darkgray",
            "background": "lightgray"
          }
        }
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": false
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 50
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -80000,
          "centralGravity": 0.3,
          "springLength": 100,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.1
        },
        "minVelocity": 0.75
      }
    }
    """
    )

    # Display the network
    nt.show('graph.html')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize ncar.gov.sa')
    parser.add_argument('--pages_data_populated', default='pages_data_populated.json', help='Path to pages_data_populated.json')
    # create --limit argument to make it less time consuming
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of pages to visualize')
    args = parser.parse_args()

    with open(args.pages_data_populated, 'r', encoding='utf8') as f:
        pages_data = json.load(f)

    G = create_graph(pages_data)
    # visualize_graph(G)
    print('visualizing graph...')
    visualize_graph(G)
