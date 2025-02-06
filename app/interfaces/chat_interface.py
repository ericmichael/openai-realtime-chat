import gradio as gr
from openai import OpenAI
from app.core.assistant_manager import AssistantManager
from app.core.tools import ToolManager
import json


def create_chat_interface(assistant_manager: AssistantManager):
    client = OpenAI()

    # Define available models
    available_models = ["gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini"]

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
        history = history or []
        history.append({"role": "user", "content": message})
        return "", history

    async def process_assistant_response(history, assistant_name, model_name):
        if not history:
            yield history
            return

        # Get assistant instructions and tools first
        assistant = assistant_manager.get_assistant(assistant_name)
        system_prompt = assistant.get("instructions", "You are a helpful assistant.")
        tools = assistant.get("tools", [])

        # Format messages (same as before)
        messages = []
        if model_name.startswith("o1"):
            if history and history[0]["role"] == "user":
                history[0]["content"] = f"{system_prompt}\n\n{history[0]['content']}"
            else:
                messages.append({"role": "user", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)

        # Prepare tools
        function_definitions = []
        available_tools = {}
        tool_manager = None

        if tools:
            tool_manager = ToolManager()
            if "python" in tools:
                from app.core.jupyter import JupyterKernel

                tool_manager.jupyter_kernel = JupyterKernel("./notebooks/chat")
            all_tools = tool_manager.get_available_tools()
            selected_tools = [f for f in all_tools if f.__name__ in tools]
            tool_manager.register_tools(selected_tools)
            function_definitions = tool_manager.chat_tools
            available_tools = tool_manager.available_functions

        # Initial streaming response
        completion_args = {"model": model_name, "messages": messages, "stream": True}

        if not model_name.startswith("o1"):
            completion_args["temperature"] = 0.0
            if function_definitions:
                completion_args["tools"] = function_definitions
                completion_args["tool_choice"] = "auto"

        # Create streaming response
        stream = client.chat.completions.create(**completion_args)

        # Initialize assistant's message
        history.append({"role": "assistant", "content": ""})
        collected_message = {
            "content": "",
            "tool_calls": {},
        }  # Changed to dict for index-based tracking

        # Process the stream
        for chunk in stream:
            delta = chunk.choices[0].delta

            # Handle content updates
            if delta.content:
                collected_message["content"] += delta.content
                history[-1]["content"] = collected_message["content"]
                yield history

            # Handle tool calls
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if tool_call.index is not None:
                        index = tool_call.index
                        # Initialize tool call at index if not exists
                        if index not in collected_message["tool_calls"]:
                            collected_message["tool_calls"][index] = {
                                "id": "",
                                "function": {"name": "", "arguments": ""},
                            }

                        # Update tool call attributes
                        if tool_call.id:
                            collected_message["tool_calls"][index]["id"] = tool_call.id
                        if tool_call.function.name:
                            collected_message["tool_calls"][index]["function"][
                                "name"
                            ] = tool_call.function.name
                        if tool_call.function.arguments:
                            collected_message["tool_calls"][index]["function"][
                                "arguments"
                            ] += tool_call.function.arguments

        # Convert collected tool calls from dict to list for processing
        tool_calls_list = list(collected_message["tool_calls"].values())

        # Handle tool execution
        if tool_calls_list:
            print(f"\nExecuting {len(tool_calls_list)} tool calls:")

            # Update history with thinking message
            history[-1]["content"] = (
                collected_message["content"]
                or "Let me use some tools to help answer that."
            )
            history[-1]["metadata"] = {"title": "Thinking..."}
            yield history

            tool_results = []
            for tool_call in tool_calls_list:
                function_name = tool_call["function"]["name"]
                try:
                    function_args = json.loads(tool_call["function"]["arguments"])
                    print(
                        f"\nExecuting {function_name} with args: {json.dumps(function_args, indent=2)}"
                    )

                    if function_name in available_tools:
                        result = available_tools[function_name](**function_args)
                        print(f"Tool result: {result}")
                        tool_results.append(
                            {"tool_call_id": tool_call["id"], "output": str(result)}
                        )

                        # Replace the "Thinking..." message with the tool result
                        history[-1] = {
                            "role": "assistant",
                            "content": str(result),
                            "metadata": {"title": f"Used Tool: {function_name}"},
                        }
                        yield history
                except Exception as e:
                    print(f"Tool execution error: {str(e)}")  # Debug
                    error_msg = f"Error executing tool '{function_name}': {str(e)}"
                    tool_results.append(
                        {"tool_call_id": tool_call["id"], "output": error_msg}
                    )
                    history.append(
                        {
                            "role": "assistant",
                            "content": error_msg,
                            "metadata": {"title": f"Tool Error: {function_name}"},
                        }
                    )
                    yield history

            print("\nAll tool results:")  # Debug
            print(json.dumps(tool_results, indent=2))  # Debug

            # Get final response after tool execution
            print("\nSending updated messages to OpenAI:")  # Debug
            print(json.dumps(messages[-3:], indent=2))  # Debug last 3 messages

            messages.append(
                {
                    "role": "assistant",
                    "content": collected_message["content"],
                    "tool_calls": [
                        {
                            "id": tool_call.get("id"),
                            "type": "function",
                            "function": {
                                "name": tool_call["function"]["name"],
                                "arguments": tool_call["function"]["arguments"],
                            },
                        }
                        for tool_call in tool_calls_list
                    ],
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

            # Stream final response
            completion_args["messages"] = messages
            if "tools" in completion_args:
                del completion_args["tools"]
                del completion_args["tool_choice"]

            final_stream = client.chat.completions.create(**completion_args)

            # Add final response message
            history.append({"role": "assistant", "content": ""})
            for chunk in final_stream:
                if chunk.choices[0].delta.content:
                    history[-1]["content"] += chunk.choices[0].delta.content
                    yield history

        yield history
        return

    def clear_history():
        return None

    # Update the event handlers to match the voice chat pattern
    msg.submit(
        fn=handle_user_message,
        inputs=[msg, chatbot, assistant_dropdown, model_dropdown],
        outputs=[msg, chatbot],
        queue=True,  # Change to queue=True
    ).then(
        fn=process_assistant_response,
        inputs=[chatbot, assistant_dropdown, model_dropdown],
        outputs=chatbot,
        queue=True,  # Change to queue=True
    )

    submit_btn.click(
        fn=handle_user_message,
        inputs=[msg, chatbot, assistant_dropdown, model_dropdown],
        outputs=[msg, chatbot],
        queue=True,  # Change to queue=True
    ).then(
        fn=process_assistant_response,
        inputs=[chatbot, assistant_dropdown, model_dropdown],
        outputs=chatbot,
        queue=True,  # Change to queue=True
    )

    clear_btn.click(lambda: None, None, chatbot, queue=False)

    return assistant_dropdown
