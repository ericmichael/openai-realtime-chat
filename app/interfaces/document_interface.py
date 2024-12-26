import gradio as gr
from app.services.document import DocumentService

document_service = DocumentService()


def create_document_interface():
    gr.Markdown("Create and manage documents")

    with gr.Row():
        with gr.Column(scale=1):
            document_list = gr.Dropdown(
                choices=["None"],
                label="Select Document",
                value="None",
            )

            new_document_btn = gr.Button("Create New Document", variant="primary")
            delete_document_btn = gr.Button("Delete Selected Document", variant="stop")

        with gr.Column(scale=2):
            document_title = gr.Textbox(
                label="Document Title",
                placeholder="Enter document title...",
                interactive=True,
            )
            document_content = gr.TextArea(
                label="Content",
                placeholder="Enter the document content...",
                lines=10,
                interactive=True,
            )
            document_published = gr.Checkbox(
                label="Published",
                value=False,
                interactive=True,
            )
            save_document_btn = gr.Button("Save Changes", variant="primary")

    # Wire up the event handlers
    document_list.change(
        load_document,
        inputs=[document_list],
        outputs=[document_title, document_content, document_published],
    )

    new_document_btn.click(
        create_new_document,
        outputs=[
            document_title,
            document_content,
            document_published,
            document_list,
        ],
    )

    save_document_btn.click(
        save_document,
        inputs=[
            document_title,
            document_content,
            document_published,
            document_list,
        ],
        outputs=[gr.Textbox(visible=False), document_list],
    )

    delete_document_btn.click(
        delete_document,
        inputs=[document_list],
        outputs=[
            document_title,
            document_content,
            document_published,
            gr.Textbox(visible=False),
            document_list,
        ],
    )

    # Initialize document list
    document_list.choices = load_document_list()


def load_document_list():
    documents = document_service.get_all_documents()
    return [("None", "None")] + [(title, title) for title in documents.keys()]


def load_document(title):
    if title == "None":
        return "", "", False
    document = document_service.get_document_by_title(title)
    if document:
        return document.title, document.content, document.published
    return "", "", False


def create_new_document():
    return "", "", False, gr.update(value="None")


def save_document(title, content, published, current_selection):
    if not title:
        return "Please enter a title", gr.update(choices=load_document_list())

    if current_selection == "None":
        document_service.create_document(title, content, published)
    else:
        document_service.update_document(current_selection, content, published)
        if current_selection != title:
            document_service.delete_document(current_selection)
            document_service.create_document(title, content, published)

    return "Document saved successfully!", gr.update(choices=load_document_list())


def delete_document(title):
    if title and title != "None":
        document_service.delete_document(title)
        return (
            "",
            "",
            False,
            "Document deleted successfully!",
            gr.update(choices=load_document_list()),
        )
    return (
        "",
        "",
        False,
        "No document selected",
        gr.update(choices=load_document_list()),
    )
