# Project AI Workshop

This project is an interactive AI chat application that uses Azure OpenAI services, built with the LangChain framework.

## Features

- Interactive chat with conversation history.
- Separate clients for chat and text embedding.
- Configuration managed via environment variables.
- Structured project layout (Model-Service-Util).

## Prerequisites

- Python 3.8+ a
- Access to Azure OpenAI with at least two deployments (one for chat, one for embedding).

## Installation and Setup

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Workshop
```

### 2. Create and Activate a Virtual Environment

A virtual environment is crucial to manage project dependencies and avoid conflicts.

**On Windows:**
```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate
```

**On macOS/Linux:**
```bash
# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

After activation, you will see `(venv)` at the beginning of your terminal prompt.

### 3. Install Dependencies

Install all required Python libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

The project uses a `.env` file to manage sensitive keys and endpoints.

1.  **Create a `.env` file** by copying the example file:

    ```bash
    # On Windows
    copy .env.example .env

    # On macOS/Linux
    cp .env.example .env
    ```

2.  **Edit the `.env` file** and replace the placeholder values (`your_...`) with your actual Azure OpenAI credentials.

### 5. Configure your IDE (Optional but Recommended)

Ensure your IDE (like VS Code or PyCharm) is configured to use the Python interpreter from the `venv` you created. This will enable proper code completion and debugging.

- **VS Code**: Use `Ctrl+Shift+P` -> "Python: Select Interpreter" and choose the one in the `venv` folder.
- **PyCharm**: Go to `Settings > Project > Python Interpreter` and add the interpreter from the `venv` folder.

## Running the Application

Once the setup is complete, you can run the main application:

```bash
python main.py
```

You will be prompted to start a conversation with the AI. Type `exit` to end the session.

## Project Structure

- `main.py`: The entry point of the application.
- `requirements.txt`: A list of all Python dependencies.
- `.env`: File for storing environment variables (not version controlled).
- `.env.example`: An example template for the `.env` file.
- `src/workshop/`:
  - `model/`: Contains the data model configuration (`AzureOpenAIModel.py`).
  - `service/`: Contains the business logic (`ConversationService.py`).
  - `util/`: Contains utility functions (`PropertiesUtil.py`).
  - `prompt.properties`: File for storing system prompts.
