## **LLM Code Assistant Instructions**

### **1. Core Directives**

* **Architecture:** Do not blindly agree with my ideas. If you detect a flaw, risk, inefficiency, or a better alternative, challenge my approach respectfully but directly. Prioritize correctness and clarity over politeness or affirmation.


* **Task Adherence:** Strictly adhere to the given task. If you identify necessary adjustments, fixes, or improvements outside the scope of the current task, you must explicitly ask for permission before implementing them.
* **Problem Simplification:** Do not simplify or remove features to solve complex problems. Workarounds are not acceptable unless you receive explicit permission.
* **Clean Code:** Write clean, modular, and maintainable code. Functions should be short, clear, and single-purpose. Avoid using `any` or `Any` types; create specific classes, interfaces, or types.
* **Code Modularity:** Decompose code into small, reusable components. No single file should exceed 800 lines.

### **2. Tech Stack & Project Structure**

* **Backend:** FastAPI, PostgreSQL, Alembic
* **Frontend:** React, Vite, magic ui, Tailwind CSS
* **LLM Integration:** AWS Bedrock
* **Package Management:** uv (Python package manager and workspace management)
* **Project Structure:** Monorepo with shared libraries:
  * `backend/` - FastAPI application
  * `client/` - React frontend
  * `libs/common/` - Shared utilities, configuration, logging, LLM services

### **3. Backend Development (`Gotchas & Style Guide`)**

* **Shared Libraries Usage:**
    * **Common Library (`libs/common/`):** Use shared utilities for configuration, logging, LLM services, and exceptions.
    * **Import Pattern:** Always import from shared libraries: `from core.config_service import ConfigService`
* **Configuration Management:**
    * **Centralized Config:** All configuration files are located in `libs/common/` (`.env.*`, `secrets.yaml`)
    * **Config Service:** Use `ConfigService` from `core.config_service` for all configuration access
    * **Environment Variables:** Load from shared `.env` files in `libs/common/`
    * **Secrets:** Store sensitive data in `libs/common/secrets.yaml` (never commit this file)
* **Dependency Injection & Initialization:**
    * Use FastAPI's built-in dependency injection system to manage all dependencies, such as services.
    * A central `dependencies.py` file must be used to create and configure these dependencies. This file will contain provider functions that initialize and `yield` instances.
    * **Services:** Create specific dependency provider functions for each service (e.g., `get_example_service`).
    * **Usage:** Inject dependencies directly into API endpoint function signatures using `Depends`. Do not instantiate services manually within endpoint logic.
* **Type Hinting & Enums:**
    * Use `StrEnum` for string-based enumerations.
    * Prefer Enums over raw strings where applicable (e.g., for status fields, roles).
    * Use modern type hints: `| None` instead of `Optional[...]`, and `list` instead of `List`, `dict` instead of `Dict`, etc. (for Python 3.10+).
* **File Paths:**
    * Use the `pathlib.Path` object for path manipulations. Replace `os.path.splitext()` with `Path.suffix`, `Path.stem`, and `Path.parent`.
* **Error Handling & Logging:**
    * **Logging Service:** Use the shared logging service from `core.logging_service.get_logger(__name__)`
    * **Exception Handling:** Use `logger.exception()` within `except` blocks to capture stack traces. Avoid `logger.error()`.
    * **Structured Logging:** Use the logging service's structured logging capabilities with key-value pairs
    * **Log Consolidation:** Consolidate multiple consecutive log lines into a single, comprehensive log entry where possible.
* **Code Style:**
    * Use double quotes (`"`) for strings, not single quotes (`'`).
    * Use explicit type conversions (e.g., `str(value)` instead of relying on implicit coercion).
    * Start multi-line docstring summaries on the first line.
    * Place all `import` statements at the top of the file, not within functions or classes.
    * Use `elif` to avoid nested `if` statements inside an `else` block.
    * Use `snake_case` for variable and function names.

### **4. Frontend Development (`Gotchas & Style Guide`)**

* **Responsive UI:** All UI components and layouts must be fully responsive, ensuring a seamless experience on both mobile and desktop screen sizes. Use Tailwind CSS's responsive design features (e.g., `md:`, `lg:`) to achieve this.
* **State & Error Handling (React Context):**
    * Centralize error handling for asynchronous operations within the React Context itself.
    * The context must expose an `error` state.
    * Async functions within the context must `catch` their own errors and update the `error` state. Do not re-throw errors from context functions.
    * Consuming components must read the `error` state to display error messages and should not contain their own `try/catch` blocks for context-provided functions.
* **Module Loading:**
    * Use dynamic `import()` for code-splitting or conditional module loading. `require()` is not available in the Vite browser environment.
* **UI Components:**
    * Use `magic ui` components whenever applicable, you have access to their MCP.
    * Before adding a new component with (This is just an example)`npx shadcn@latest add "https://magicui.design/r/globe.json"`, verify it is not already installed in the project.
* **Code Style:**
    * Use `camelCase` for variable and function names.
* **Tooling Setups:**
    * **Tailwind CSS with Vite:** Follow the provided step-by-step guide for installation and configuration via the `@tailwindcss/vite` plugin.
    * **shadcn/ui Setup:** Follow the provided step-by-step guide for project creation, TypeScript configuration (`tsconfig.json`), Vite configuration (`vite.config.ts`), and `shadcn` initialization.

### **5. API Design & Communication**

* **API Endpoints:** All API endpoints must follow RESTful principles. Use nouns for resource names (e.g., `/users`, `/documents`) and HTTP verbs for actions (`GET`, `POST`, `PUT`, `DELETE`).
* **API Responses:** Standardize API success and error responses.
    * **Success Example:** `{ "status": "success", "data": { ... } }`
    * **Error Example:** `{ "status": "error", "message": "A descriptive error message" }`
* **Environment Variables:** All sensitive information (API keys, secrets) must be loaded from environment variables using Pydantic's `BaseSettings`, never hardcoded.

### **6. Package Management & Dependencies**

* **uv Workspace Management:**
    * **Install Dependencies:** Use `uv sync` to install all workspace dependencies
    * **Add Dependencies:** Use `uv add <package>` to add new dependencies to the current package
    * **Add Dev Dependencies:** Use `uv add --group dev <package>` for development dependencies
    * **Install Tools:** Use `uv tool install <tool>` for CLI tools like awscli-local
    * **Run Commands:** Use `uv run --package <package> <command>` to run commands in specific packages
* **Never Use pip Directly:**
    * Do not use `pip install` - always use uv commands
    * Do not manually edit `requirements.txt` files - use uv workspace management
    * Do not use `pip freeze` - uv manages dependencies through `pyproject.toml` and `uv.lock`

### **8. Workflow**


* **Development Commands:**
    * **Start Backend:** `just run-backend` or `uv run --package backend uvicorn app.main:app --reload`
    * **Start Frontend:** `just run-client` or `cd client && npm start`

    * **Lint Code:** `just lint` or `uv run ruff check`
    * **Format Code:** `just format` or `uv run ruff format`

***

### Clarifying Questions

To ensure the LLM assistant operates with perfect clarity, I recommend we define the following:

1.  **Shared Libraries Usage:**
    * **Import Patterns:** When should code import from `libs/common/` vs local modules? Are there any circular dependency concerns to be aware of?
2.  **Configuration Management:**
    * **Environment-Specific Config:** How should environment-specific configurations be handled across the monorepo? Should each package have its own environment handling or rely entirely on the shared config service?
5.  **Permissions:**
    * What is the preferred communication channel for asking for permission to execute out-of-scope tasks? (e.g., a specific comment format in the code, a message in a chat, etc.)