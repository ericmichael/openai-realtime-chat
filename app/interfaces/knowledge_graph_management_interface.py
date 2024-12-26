import gradio as gr
import pandas as pd
from app.models.base import Session
from app.models.node import Node, Edge
from app.services.knowledge_graph import KnowledgeGraphService


def create_knowledge_graph_management_interface(
    knowledge_graph_service: KnowledgeGraphService,
):
    """Create the Knowledge Graph Management interface tab

    Args:
        knowledge_graph_service: The Knowledge Graph service instance to use
    """
    print("Initializing Knowledge Graph tab")

    # Initialize the node list
    initial_nodes = knowledge_graph_service.get_all_nodes()
    initial_node_choices = ["None"] + [
        f"{node.name} ({node.node_type})" for node in initial_nodes
    ]
    print(f"Initial node choices: {initial_node_choices}")

    # Add a refresh button at the top
    refresh_nodes_btn = gr.Button("Refresh Node List")

    with gr.Row():
        # First column - Node Management
        with gr.Column(scale=1):
            node_list = gr.Dropdown(
                choices=initial_node_choices,
                label="Select Node",
                value="None",
                interactive=True,
            )
            node_type_input = gr.Dropdown(
                choices=["Person", "Location", "Book", "Character", "Book Series"],
                label="Node Type",
                value="Person",
                interactive=True,
            )
            node_name = gr.Textbox(
                label="Node Name",
                placeholder="Enter node name...",
                interactive=True,
            )
            node_description = gr.TextArea(
                label="Description",
                placeholder="Enter node description...",
                lines=3,
                interactive=True,
            )
            with gr.Row():
                new_node_btn = gr.Button("Create New Node", variant="primary")
                delete_node_btn = gr.Button("Delete Node", variant="stop")
                save_node_btn = gr.Button("Save Changes", variant="secondary")

        # Second column - Relationships and Node Info
        with gr.Column(scale=1):
            gr.Markdown("### Node Relationships")
            relationships_out = gr.DataFrame(
                headers=["Relationship", "Target Node"],
                label="Outgoing Relationships",
                interactive=False,
            )
            relationships_in = gr.DataFrame(
                headers=["Source Node", "Relationship"],
                label="Incoming Relationships",
                interactive=False,
            )

            gr.Markdown("### Add Relationship")
            source_node = gr.Dropdown(
                choices=initial_node_choices,
                label="Source Node",
                value="None",
                interactive=True,
            )
            relationship_type = gr.Textbox(
                label="Relationship Type",
                placeholder="e.g., wrote, contains, lives in...",
                interactive=True,
            )
            target_node = gr.Dropdown(
                choices=initial_node_choices,
                label="Target Node",
                value="None",
                interactive=True,
            )
            add_edge_btn = gr.Button("Add Relationship", variant="primary")

    def load_node_list():
        """Refresh the list of nodes in all dropdowns"""
        nodes = knowledge_graph_service.get_all_nodes()
        choices = ["None"] + [f"{node.name} ({node.node_type})" for node in nodes]
        return {
            node_list: gr.update(choices=choices),
            source_node: gr.update(choices=choices),
            target_node: gr.update(choices=choices),
        }

    def load_node_details(selected_node):
        """Load details for the selected node"""
        if selected_node == "None":
            return {
                node_name: "",
                node_type_input: "Person",
                node_description: "",
                relationships_out: [],
                relationships_in: [],
            }

        node_name_raw = selected_node.split(" (")[0]
        with Session() as session:
            node = session.query(Node).filter_by(name=node_name_raw).first()
            if not node:
                return {
                    node_name: "",
                    node_type_input: "Person",
                    node_description: "",
                    relationships_out: [],
                    relationships_in: [],
                }

            # Get relationships with proper joins
            outgoing = (
                session.query(Edge, Node)
                .join(Node, Edge.target_id == Node.id)
                .filter(Edge.source_id == node.id)
                .all()
            )

            incoming = (
                session.query(Edge, Node)
                .join(Node, Edge.source_id == Node.id)
                .filter(Edge.target_id == node.id)
                .all()
            )

            # Format relationships
            outgoing_data = [
                (edge.relationship_type, target_node.name)
                for edge, target_node in outgoing
            ]

            incoming_data = [
                (source_node.name, edge.relationship_type)
                for edge, source_node in incoming
            ]

            return {
                node_name: node.name,
                node_type_input: node.node_type,
                node_description: node.description or "",
                relationships_out: pd.DataFrame(
                    outgoing_data, columns=["Relationship", "Target Node"]
                ),
                relationships_in: pd.DataFrame(
                    incoming_data, columns=["Source Node", "Relationship"]
                ),
            }

    def create_new_node():
        """Reset the form for creating a new node"""
        return {
            node_name: "",
            node_type_input: "Person",
            node_description: "",
            node_list: "None",
        }

    def save_node(name, node_type, description, current_selection):
        """Save new node or update existing one"""
        try:
            if current_selection == "None":
                node = knowledge_graph_service.create_node(name, node_type, description)
            else:
                node_name = current_selection.split(" (")[0]
                existing_node = next(
                    (
                        n
                        for n in knowledge_graph_service.get_all_nodes()
                        if n.name == node_name
                    ),
                    None,
                )
                if existing_node:
                    node = knowledge_graph_service.update_node(
                        existing_node.id, name, node_type, description
                    )

            return {**load_node_list(), gr.Info: "Node saved successfully!"}
        except Exception as e:
            return {gr.Error: f"Error saving node: {str(e)}"}

    def add_relationship(source, rel_type, target):
        """Add a new relationship between nodes"""
        if source == "None" or target == "None" or not rel_type:
            return [[], []]

        try:
            source_name = source.split(" (")[0]
            target_name = target.split(" (")[0]

            with Session() as session:
                source_node = session.query(Node).filter_by(name=source_name).first()
                target_node = session.query(Node).filter_by(name=target_name).first()

                if not source_node or not target_node:
                    return [[], []]

                edge = knowledge_graph_service.create_edge(
                    source_node.id, target_node.id, rel_type
                )

                if edge:
                    # Get updated relationships with proper joins
                    outgoing = (
                        session.query(Edge, Node)
                        .join(Node, Edge.target_id == Node.id)
                        .filter(Edge.source_id == source_node.id)
                        .all()
                    )

                    incoming = (
                        session.query(Edge, Node)
                        .join(Node, Edge.source_id == Node.id)
                        .filter(Edge.target_id == source_node.id)
                        .all()
                    )

                    # Format relationships
                    outgoing_data = [
                        (edge.relationship_type, target_node.name)
                        for edge, target_node in outgoing
                    ]

                    incoming_data = [
                        (source_node.name, edge.relationship_type)
                        for edge, source_node in incoming
                    ]

                    return [
                        pd.DataFrame(
                            outgoing_data, columns=["Relationship", "Target Node"]
                        ),
                        pd.DataFrame(
                            incoming_data, columns=["Source Node", "Relationship"]
                        ),
                    ]
                return [[], []]
        except Exception as e:
            print(f"Error creating relationship: {e}")
            return [[], []]

    def delete_current_node(selected_node):
        """Delete the currently selected node"""
        if selected_node == "None":
            return {gr.Error: "No node selected"}

        try:
            node_name = selected_node.split(" (")[0]
            with Session() as session:
                node = session.query(Node).filter_by(name=node_name).first()
                if node:
                    knowledge_graph_service.delete_node(node.id)
                    return {
                        **create_new_node(),
                        **load_node_list(),
                        gr.Info: "Node deleted successfully!",
                    }
                return {gr.Error: "Node not found"}
        except Exception as e:
            return {gr.Error: f"Error deleting node: {str(e)}"}

    # Wire up all the event handlers
    refresh_nodes_btn.click(
        fn=load_node_list, outputs=[node_list, source_node, target_node]
    )

    node_list.change(
        fn=load_node_details,
        inputs=[node_list],
        outputs=[
            node_name,
            node_type_input,
            node_description,
            relationships_out,
            relationships_in,
        ],
    )

    new_node_btn.click(
        fn=create_new_node,
        outputs=[node_name, node_type_input, node_description, node_list],
    )

    save_node_btn.click(
        fn=save_node,
        inputs=[node_name, node_type_input, node_description, node_list],
        outputs=[node_list, source_node, target_node],
    )

    delete_node_btn.click(
        fn=delete_current_node,
        inputs=[node_list],
        outputs=[
            node_name,
            node_type_input,
            node_description,
            node_list,
            source_node,
            target_node,
        ],
    )

    add_edge_btn.click(
        fn=add_relationship,
        inputs=[source_node, relationship_type, target_node],
        outputs=[relationships_out, relationships_in],
    )
