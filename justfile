# Aliases
alias install := sync

# Install python dependencies using uv
sync:
    uv sync

# Upgrade python dependencies
upgrade:
    uv sync --upgrade

# Install pre-commit hooks
pre_commit_setup:
    uv run pre-commit install

# Install python dependencies and pre-commit hooks
setup: sync pre_commit_setup

# Run pre-commit on all files
pre_commit:
    uv run pre-commit run -a



# Lint code with ruff
lint folder="." fix="":
    uv run ruff check {{folder}} {{fix}}

# Format code with ruff
format folder=".":
    uv run ruff format {{folder}}

# Type check with pyright
pyright directory=".":
    uv run pyright --threads 8 {{directory}}


# Run the backend server (uv version)
run-backend:
    uv run --package backend uvicorn app.main:app --host 0.0.0.0 --port 9200 --reload

# Run the frontend client
run-client:
    cd client && npm start

# Run both backend and frontend
run:
    just run-backend & just run-client

# Build dockerfile for specific target
build target:
    docker build -t packages/{{target}} --build-arg PACKAGE={{target}} .

# Start local development environment
local_dev_up:
    docker compose up -d --remove-orphans

# Stop local development environment
local_dev_down:
    docker compose down





# SSH key management commands
ssh-use-germanilia:
    cp /tmp/host-germanilia-key ~/.ssh/id_rsa
    chmod 600 ~/.ssh/id_rsa
    echo "Now using germanilia SSH key"
    ssh-add -l

ssh-use-iliagerman:
    cp /tmp/host-iliagerman-key ~/.ssh/id_rsa
    chmod 600 ~/.ssh/id_rsa
    echo "Now using iliagerman SSH key"
    ssh-add -l

ssh-status:
    echo "Current SSH key fingerprint:"
    ssh-keygen -lf ~/.ssh/id_rsa
    echo "\nLoaded SSH keys:"
    ssh-add -l










# Export AWS environment variables from secrets.yaml to current terminal session
aws:
    #!/bin/bash
    set -e

    # Check if secrets.yaml exists
    if [ ! -f "libs/common/secrets.yaml" ]; then
        echo "ERROR: libs/common/secrets.yaml not found" >&2
        echo "Please ensure the secrets file exists with AWS credentials" >&2
        exit 1
    fi

    # Extract AWS credentials using Python
    AWS_ACCESS_KEY_ID=$(python3 -c "import yaml; data=yaml.safe_load(open('libs/common/secrets.yaml')); print(data['opencode']['access_key_id'])" 2>/dev/null || echo "")
    AWS_SECRET_ACCESS_KEY=$(python3 -c "import yaml; data=yaml.safe_load(open('libs/common/secrets.yaml')); print(data['opencode']['secret_access_key'])" 2>/dev/null || echo "")
    AWS_DEFAULT_REGION=$(python3 -c "import yaml; data=yaml.safe_load(open('libs/common/secrets.yaml')); print(data['opencode']['region'])" 2>/dev/null || echo "")

    # Check if credentials were loaded successfully
    if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ] || [ -z "$AWS_DEFAULT_REGION" ]; then
        echo "ERROR: Failed to load AWS credentials from secrets.yaml" >&2
        echo "Please ensure the file contains valid opencode.access_key_id, opencode.secret_access_key, and opencode.region" >&2
        exit 1
    fi

    # Output export statements that can be evaluated
    echo "export AWS_ACCESS_KEY_ID=\"$AWS_ACCESS_KEY_ID\""
    echo "export AWS_SECRET_ACCESS_KEY=\"$AWS_SECRET_ACCESS_KEY\""
    echo "export AWS_DEFAULT_REGION=\"$AWS_DEFAULT_REGION\""

    # Output informational messages to stderr so they don't interfere with eval
    echo "# ✓ AWS credentials loaded from libs/common/secrets.yaml" >&2
    echo "#   AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:0:8}..." >&2
    echo "#   AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY:0:8}..." >&2
    echo "#   AWS_DEFAULT_REGION: $AWS_DEFAULT_REGION" >&2
    echo "#" >&2
    echo "# To set these variables in your current shell, run:" >&2
    echo "#   eval \"\$(just aws)\"" >&2