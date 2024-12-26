from app.core.tools import tool
import ast


@tool
def python(code: str, kernel=None):
    "Return result of executing `code` using python. Use this to run any kinds of complex calculations, computation, data analysis, etc."
    if not kernel:
        return {"error": "Jupyter kernel not initialized"}

    result, _ = kernel.execute_code(code)
    return {"result": result}
