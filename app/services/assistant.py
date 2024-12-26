from app.models.base import Session
from app.models.assistant import Assistant


class AssistantService:
    @staticmethod
    def get_all_assistants():
        with Session() as session:
            assistants = Assistant.all(session)
            return {a.name: a.to_dict() for a in assistants}

    @staticmethod
    def get_assistant_by_name(name):
        with Session() as session:
            return Assistant.find_by_name(session, name)

    @staticmethod
    def create_assistant(name, instructions, voice, tools=None):
        with Session() as session:
            assistant = Assistant(
                name=name, instructions=instructions, voice=voice, tools=tools or []
            )
            session.add(assistant)
            session.commit()
            return assistant

    @staticmethod
    def update_assistant(name, instructions, voice, tools=None):
        with Session() as session:
            assistant = Assistant.find_by_name(session, name)
            if assistant:
                assistant.instructions = instructions
                assistant.voice = voice
                assistant.tools = tools or []
                session.commit()
                return assistant
            return None

    @staticmethod
    def delete_assistant(name):
        with Session() as session:
            assistant = Assistant.find_by_name(session, name)
            if assistant:
                session.delete(assistant)
                session.commit()
                return True
            return False
