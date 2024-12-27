import gradio as gr
from openai import OpenAI
from app.core.assistant_manager import AssistantManager
from app.core.tools import ToolManager
import json


def create_chat_interface(assistant_manager: AssistantManager):
    client = OpenAI()

    # Define available models
    available_models = ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini"]

    with gr.Column():
        with gr.Row():
            assistant_dropdown = gr.Dropdown(
                choices=list(assistant_manager.get_all_assistants().keys()),
                label="Select Assistant",
                value="General Assistant",
                interactive=True,
            )
            model_dropdown = gr.Dropdown(
                choices=available_models,
                label="Select Model",
                value="gpt-4o",
                interactive=True,
            )

        chatbot = gr.Chatbot(height=500, type="messages")
        msg = gr.Textbox(
            placeholder="Type your message here...",
            container=False,
            scale=7,
        )
        with gr.Row():
            submit_btn = gr.Button("Submit", scale=1, variant="primary")
            clear_btn = gr.Button("Clear", scale=1)

    async def handle_user_message(message, history, assistant_name, model_name):
        """First step: Add user message to history immediately"""
        history.append({"role": "user", "content": message})
        return "", history

    async def process_assistant_response(history, assistant_name, model_name):
        """Second step: Process the assistant's response"""
        if not history:
            return history

        # Get assistant instructions and tools first
        assistant = assistant_manager.get_assistant(assistant_name)
        system_prompt = assistant.get("instructions", "You are a helpful assistant.")
        tools = assistant.get("tools", [])

        # Format conversation history for OpenAI
        messages = [{"role": "system", "content": system_prompt}]
        for h in history:
            messages.append(h)

        # Prepare tools if any are configured
        function_definitions = []
        available_tools = {}

        if tools:
            tool_manager = ToolManager()

            # Initialize Jupyter kernel if Python tool is selected
            if "python" in tools:
                from app.core.jupyter import JupyterKernel

                tool_manager.jupyter_kernel = JupyterKernel("./notebooks/chat")

            # Get and register selected tools
            all_tools = tool_manager.get_available_tools()
            selected_tools = [f for f in all_tools if f.__name__ in tools]
            tool_manager.register_tools(selected_tools)

            # Get tool schemas and available functions
            function_definitions = tool_manager.chat_tools
            available_tools = tool_manager.available_functions

        print(f"Messages: {messages}")
        # Get response from OpenAI with tools
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=function_definitions if function_definitions else None,
            tool_choice="auto" if function_definitions else "none",
            temperature=0.0,
        )

        assistant_message = response.choices[0].message

        # Handle tool calls if present
        if assistant_message.tool_calls:
            tool_results = []
            # Add the assistant's initial response with tool calls
            history.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content
                    or "Let me use some tools to help answer that.",
                    "metadata": {"title": "Thinking..."},
                }
            )

            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    if function_name in available_tools:
                        result = available_tools[function_name](**function_args)
                        tool_results.append(
                            {"tool_call_id": tool_call.id, "output": str(result)}
                        )
                        # Add tool execution result to chat
                        history.append(
                            {
                                "role": "assistant",
                                "content": str(result),
                                "metadata": {"title": f"Used Tool: {function_name}"},
                            }
                        )
                except Exception as e:
                    error_msg = f"Error executing tool '{function_name}': {str(e)}"
                    tool_results.append(
                        {"tool_call_id": tool_call.id, "output": error_msg}
                    )
                    # Add error message to chat
                    history.append(
                        {
                            "role": "assistant",
                            "content": error_msg,
                            "metadata": {"title": f"Tool Error: {function_name}"},
                        }
                    )

            # Get final response after tool execution
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": assistant_message.tool_calls,
                }
            )
            for tool_result in tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result["tool_call_id"],
                        "content": tool_result["output"],
                    }
                )

            # Get final response from OpenAI
            final_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
            )
            assistant_message = final_response.choices[0].message

            # Update the history with the final assistant's response
            history.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                }
            )
        else:
            # For regular responses without tool calls
            history.append({"role": "assistant", "content": assistant_message.content})

        return history

    def clear_history():
        return None

    # Update the event handlers
    msg.submit(
        handle_user_message,
        [msg, chatbot, assistant_dropdown, model_dropdown],
        [msg, chatbot],
    ).then(
        process_assistant_response,
        [chatbot, assistant_dropdown, model_dropdown],
        chatbot,
    )

    submit_btn.click(
        handle_user_message,
        [msg, chatbot, assistant_dropdown, model_dropdown],
        [msg, chatbot],
    ).then(
        process_assistant_response,
        [chatbot, assistant_dropdown, model_dropdown],
        chatbot,
    )

    clear_btn.click(clear_history, None, chatbot)

    return assistant_dropdown
