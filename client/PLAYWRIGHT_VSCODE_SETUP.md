# Playwright VS Code Extension Setup

This document explains how to use the Playwright VS Code extension with our project's custom test environment setup.

## Overview

Our project uses `just` commands to manage the complete test environment, including:
- Starting backend server with test configuration
- Starting frontend server
- Setting up test database
- Managing environment variables

The VS Code Playwright extension has been configured to work seamlessly with this setup through automatic server management.

## ✅ Solution Summary

The VS Code Playwright extension now works with your `just` commands through:

1. **Automatic Server Management**: The extension automatically starts test servers when needed
2. **Workspace-Level Configuration**: Tests are discoverable from the VS Code Test Explorer
3. **Environment Setup**: Proper environment variables are set automatically
4. **Seamless Integration**: Click any test in the Test Explorer to run it with full environment

## Configuration Files

The following files have been configured to make the Playwright extension work with our `just` commands:

### 1. `playwright.config.ts` (workspace root)
- Workspace-level configuration that the VS Code extension automatically discovers
- Points to the client-specific VS Code configuration

### 2. `client/playwright.vscode.config.ts`
- Special Playwright configuration designed for VS Code Test Explorer
- Includes automatic server management through global setup
- Optimized settings for VS Code usage (single worker, headless mode)

### 3. `client/playwright-global-setup.ts`
- Global setup script that runs before any tests
- Automatically checks if test servers are running
- Starts servers using `just start-test-servers` if needed
- Sets proper environment variables

### 4. `.vscode/extensions.json`
- Recommends the Playwright extension for the workspace

### 5. `.vscode/tasks.json`
- Added VS Code tasks for managing test servers manually
- Added tasks for running different test suites

### 6. `client/playwright-with-servers.sh` (for manual use)
- Custom script for command-line usage
- Ensures test servers are running before executing tests

## Usage

### Using the Playwright Extension

1. **Install the Playwright Extension**: Make sure you have the "Playwright Test for VSCode" extension installed

2. **Running Tests**: The extension will now automatically:
   - Check if test servers are running
   - Start servers if needed (using `just start-test-servers`)
   - Run the selected tests
   - Keep servers running for debugging

3. **Test Discovery**: The extension should automatically discover tests in `client/tests/integration/`

### Manual Commands

You can also use these commands manually:

```bash
# Start test servers (run from project root)
just start-test-servers

# Stop test servers
just stop-test-servers

# Run all E2E tests with full setup
just test-e2e all

# Run specific test suites
just test-e2e regression
just test-e2e login

# Run tests with the custom wrapper (from client directory)
cd client
npm test

# Run raw Playwright tests (servers must be running)
cd client
npm run test:raw
```

### VS Code Tasks

Use Ctrl+Shift+P (Cmd+Shift+P on Mac) and search for "Tasks: Run Task" to access:

- **Start Test Servers (E2E)**: Starts the test environment
- **Stop Test Servers (E2E)**: Stops the test environment
- **Run E2E Tests (All)**: Runs all E2E tests
- **Run E2E Tests (Regression)**: Runs only regression tests

## Troubleshooting

### Extension Not Finding Tests
- Make sure you're in the correct workspace folder
- Check that `.vscode/settings.json` has the correct `playwright.testDir` setting
- Reload VS Code window (Ctrl+Shift+P → "Developer: Reload Window")

### Tests Failing Due to Server Issues
- Manually run `just start-test-servers` to check for server startup issues
- Check that ports 9201 (backend) and 5274 (frontend) are available
- Look at the terminal output for any error messages

### Extension Using Wrong Command
- Check that `playwright.testCommand` in `.vscode/settings.json` is set to `"npm test"`
- Make sure `client/playwright-with-servers.sh` is executable (`chmod +x`)

### Servers Not Starting
- Run `just stop-test-servers` to clean up any stuck processes
- Check that you have all dependencies installed (`just sync`)
- Verify that the database can be created/migrated

## Environment Variables

The following environment variables are automatically set when using the configured setup:

- `PLAYWRIGHT_BASE_URL=http://localhost:5274`
- `USE_MEMORY_DB=true`
- `APP_ENV=test`
- `DATABASE_URL=sqlite:///./test_database.db`

## Notes

- The test servers will continue running after tests complete to allow for debugging
- Use `just stop-test-servers` when you're done testing
- The setup uses mock Cognito by default for faster test execution
- For real Cognito tests, use the specific `just test-e2e login` command
