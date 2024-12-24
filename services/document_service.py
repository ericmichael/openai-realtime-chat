from models.base import Session
from models.document import Document


class DocumentService:
    @staticmethod
    def get_all_documents():
        with Session() as session:
            documents = Document.all(session)
            return {doc.title: doc for doc in documents}

    @staticmethod
    def get_document_by_title(title):
        with Session() as session:
            return Document.find_by_title(session, title)

    @staticmethod
    def create_document(title, content, published=False):
        print(f"Creating document: {title}")  # Debug
        with Session() as session:
            try:
                document = Document.create(
                    session, title=title, content=content, published=published
                )
                print(f"Document created with ID: {document.id}")  # Debug
                print(
                    f"Vector embeddings count: {len(document.vector_embeddings)}"
                )  # Debug
                return document
            except Exception as e:
                print(f"Error creating document: {e}")  # Debug
                session.rollback()
                raise

    @staticmethod
    def update_document(title, content, published=False):
        print(f"Updating document: {title}")  # Debug
        with Session() as session:
            try:
                document = Document.find_by_title(session, title)
                if document:
                    document.update(session, content=content, published=published)
                    print(
                        f"Document updated. Vector embeddings count: {len(document.vector_embeddings)}"
                    )  # Debug
                    return document
                print(f"Document not found: {title}")  # Debug
                return None
            except Exception as e:
                print(f"Error updating document: {e}")  # Debug
                session.rollback()
                raise

    @staticmethod
    def delete_document(title):
        print(f"Deleting document: {title}")  # Debug
        with Session() as session:
            try:
                document = Document.find_by_title(session, title)
                if document:
                    session.delete(document)
                    session.commit()
                    print("Document deleted successfully")  # Debug
                    return True
                print(f"Document not found: {title}")  # Debug
                return False
            except Exception as e:
                print(f"Error deleting document: {e}")  # Debug
                session.rollback()
                raise
