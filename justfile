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

# Run pytest using uv
test:
    uv run pytest

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





# Playwright Test Commands with automatic server startup

# Start test servers (backend + frontend)
start-test-servers:
    #!/bin/bash
    set -e

    # Kill any existing processes on test ports
    echo "🛑 Ensuring no servers are running on test ports..."
    pkill -f "uvicorn.*:9201" || true
    pkill -f "vite.*5274" || true
    # Force kill any processes on test ports
    lsof -ti:9201 | xargs kill -9 2>/dev/null || true
    lsof -ti:5274 | xargs kill -9 2>/dev/null || true
    sleep 2

    # Verify ports are actually free
    echo "🔍 Verifying ports 9201 and 5274 are free..."
    for i in {1..10}; do
        if ! lsof -i:9201 >/dev/null 2>&1 && ! lsof -i:5274 >/dev/null 2>&1; then
            echo "✅ Ports are free"
            break
        fi
        if [ $i -eq 10 ]; then
            echo "❌ Failed to free ports after 10 attempts"
            exit 1
        fi
        echo "⏳ Waiting for ports to be freed... (attempt $i/10)"
        sleep 1
    done

    echo "🚀 Starting backend server on port 9201..."
    APP_ENV=test uv run --package backend uvicorn app.main:app --host 0.0.0.0 --port 9201 --reload &
    BACKEND_PID=$!

    echo "⏳ Waiting for backend to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:9201/health >/dev/null 2>&1; then
            echo "✅ Backend ready and responding"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "❌ Backend failed to start after 30 seconds"
            echo "Backend process status:"
            ps aux | grep $BACKEND_PID || echo "Backend process not found"
            exit 1
        fi
        echo "⏳ Waiting for backend... (attempt $i/30)"
        sleep 1
    done

    echo "🌐 Starting frontend server on port 5274..."
    cd client && VITE_API_URL=http://localhost:9201 npm run dev -- --host 0.0.0.0 --port 5274 &
    CLIENT_PID=$!
    cd ..

    echo "⏳ Waiting for frontend to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:5274 >/dev/null 2>&1; then
            echo "✅ Frontend ready and responding"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "❌ Frontend failed to start after 30 seconds"
            echo "Frontend process status:"
            ps aux | grep $CLIENT_PID || echo "Frontend process not found"
            exit 1
        fi
        echo "⏳ Waiting for frontend... (attempt $i/30)"
        sleep 1
    done

    echo "✅ Test servers setup completed"

    echo "🎉 Test servers are running!"
    echo "Backend: http://localhost:9201"
    echo "Frontend: http://localhost:5274"
    echo ""
    echo "To stop servers: just stop-test-servers"

# Stop test servers
stop-test-servers:
    #!/bin/bash
    echo "🛑 Stopping test servers..."
    pkill -f "uvicorn.*:9201" || true
    pkill -f "vite.*5274" || true
    # Force kill any processes on test ports
    lsof -ti:9201 | xargs kill -9 2>/dev/null || true
    lsof -ti:5274 | xargs kill -9 2>/dev/null || true

    # Verify servers are actually stopped
    echo "🔍 Verifying servers are stopped..."
    for i in {1..10}; do
        if ! lsof -i:9201 >/dev/null 2>&1 && ! lsof -i:5274 >/dev/null 2>&1; then
            echo "✅ Test servers stopped and ports are free"
            break
        fi
        if [ $i -eq 10 ]; then
            echo "⚠️  Warning: Some processes may still be running on test ports"
            lsof -i:9201 2>/dev/null || true
            lsof -i:5274 2>/dev/null || true
        fi
        echo "⏳ Waiting for servers to stop... (attempt $i/10)"
        sleep 1
    done

# # Start test servers with real Cognito (uses development environment)
# start-test-servers-real-cognito:
#     #!/bin/bash
#     set -e

#     # Kill any existing processes on test ports
#     pkill -f "uvicorn.*:9201" || true
#     pkill -f "vite.*5274" || true
#     sleep 2

#     echo "🚀 Starting backend server with REAL Cognito (development env) on port 9201..."
#     # Use development environment which has real Cognito configured
#     APP_ENV=development uv run --package backend uvicorn app.main:app --host 0.0.0.0 --port 9201 --reload &
#     BACKEND_PID=$!

#     echo "⏳ Waiting for backend to be ready..."
#     timeout 30 bash -c 'until curl -s http://localhost:9201/health > /dev/null; do sleep 1; done' || (echo "❌ Backend failed to start" && exit 1)
#     echo "✅ Backend ready"

#     echo "🌐 Starting frontend server on port 5274..."
#     cd client && VITE_API_URL=http://localhost:9201 npm run dev -- --host 0.0.0.0 --port 5274 &
#     CLIENT_PID=$!
#     cd ..

#     echo "⏳ Waiting for frontend to be ready..."
#     timeout 30 bash -c 'until curl -s http://localhost:5274 > /dev/null; do sleep 1; done' || (echo "❌ Frontend failed to start" && exit 1)
#     echo "✅ Frontend ready"

#     echo "📊 Setting up development database for testing..."
#     echo "🔧 Running database migrations..."
#     cd /workspaces/bobtimes && APP_ENV=development uv run --package shared_db alembic -c libs/shared_db/alembic.ini upgrade head
#     echo "📝 Populating test data (real Cognito users already exist)..."
#     echo "✅ Development database ready"

#     echo "🎉 Test servers are running with REAL Cognito (development environment)!"
#     echo "Backend: http://localhost:9201"
#     echo "Frontend: http://localhost:5274"
#     echo ""
#     echo "To stop servers: just stop-test-servers"

# # E2E Test Commands - Main Interface

# Run E2E tests (all tests or specific suite)
test-e2e suite="all":
    #!/bin/bash
    set -e
    case "{{suite}}" in
        "all")
            trap 'just stop-test-servers' EXIT
            echo "🧪 Running All E2E Tests (Mock + Real Cognito)"

            # First run all mock Cognito tests (excluding real-cognito tagged tests)
            echo "📋 Phase 1: Running Mock Cognito Tests"
            just start-test-servers
            cd client && PLAYWRIGHT_BASE_URL=http://localhost:5174 USE_MEMORY_DB=true npx playwright test --config=playwright.just.config.ts --grep-invert "@real-cognito"
            just stop-test-servers



            echo "📊 Test report stored at: client/test-results/html-report/index.html"
            echo "📁 Test artifacts stored in: client/test-results/"
            ;;
        "login")
            just _run-e2e-suite "login" "Login Flow Tests"
            ;;
        "registration")
            just _run-e2e-suite "registration" "Registration Flow Tests"
            ;;
        "regression")
            just _run-e2e-suite "@regression" "Regression Tests (Critical)"
            ;;

        "list")
            just test-e2e-list
            ;;
        *)
            echo "❌ Unknown test suite: {{suite}}"
            echo "Run 'just test-e2e list' to see available suites"
            exit 1
            ;;
    esac

# List available E2E test suites
test-e2e-list:
    echo "📋 Available E2E Test Suites:"
    echo ""
    echo "  login        - Login flow tests (mock + real Cognito)"
    echo "  registration - Registration flow tests (mock Cognito only)"
    echo "  regression   - Critical regression tests (mock Cognito only)"
    # echo "  real-cognito - Real Cognito integration tests only (actual AWS)"
    echo ""
    echo "Usage:"
    echo "  just test-e2e              # Run all tests (mock + real Cognito)"
    echo "  just test-e2e all          # Run all tests (explicit)"
    echo "  just test-e2e <suite>      # Run specific suite"
    echo "  just test-e2e list         # Show this list"
    echo ""
    echo "Note: 'all' and 'login' suites run both mock and real Cognito tests"
    echo "      Other suites run only mock Cognito tests for fast execution"

# Internal helper: Run E2E suite with mock Cognito
_run-e2e-suite pattern description:
    #!/bin/bash
    set -e
    trap 'just stop-test-servers' EXIT
    echo "🧪 Running {{description}}"
    just start-test-servers
    cd client && PLAYWRIGHT_BASE_URL=http://localhost:5174 USE_MEMORY_DB=true npx playwright test --config=playwright.just.config.ts -g "{{pattern}}"
    just stop-test-servers
    echo "📊 Test report stored at: client/test-results/html-report/index.html"
    echo "📁 Test artifacts stored in: client/test-results/"

# Internal helper: Run E2E suite with real Cognito
_run-e2e-suite-real-cognito pattern description:
    #!/bin/bash
    set -e
    trap 'just stop-test-servers' EXIT
    echo "🧪 Running {{description}}"
    just start-test-servers-real-cognito
    cd client && PLAYWRIGHT_BASE_URL=http://localhost:5174 npx playwright test --config=playwright.just.config.ts -g "{{pattern}}"
    just stop-test-servers
    echo "📊 Test report stored at: client/test-results/html-report/index.html"
    echo "📁 Test artifacts stored in: client/test-results/"

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

git-test-connection:
    ssh -T git@github.com








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