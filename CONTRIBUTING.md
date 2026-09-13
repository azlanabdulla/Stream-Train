# Contributing to StreamTrain

First off, thank you for considering contributing to StreamTrain! It's people like you that make open source such a great community.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/StreamTrain.git
   cd StreamTrain
   ```
3. Set up a virtual environment and install the development dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -e ".[dev]"
   ```

## Development Workflow

- **Testing**: We use `pytest` for testing. Run all tests to ensure everything is working:
  ```bash
  pytest tests/
  ```
- **Code Formatting**: We use `black` and `ruff` for code formatting and linting. You can check your code by running:
  ```bash
  black streamtrain tests examples
  ruff check streamtrain tests examples
  ```
  Alternatively, you can install the `pre-commit` hooks to run these automatically on every commit:
  ```bash
  pre-commit install
  ```

## Opening a Pull Request

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/my-awesome-feature
   ```
2. Commit your changes. Ensure your commit messages are descriptive.
3. Push the branch to your fork:
   ```bash
   git push origin feature/my-awesome-feature
   ```
4. Open a Pull Request against the `main` branch of the upstream repository.

## Reporting Bugs

If you find a bug, please create an issue on GitHub with:
- A clear description of the problem.
- A minimal reproducible example.
- The version of StreamTrain, PyTorch, and Python you are using.
- Your OS and hardware specifications.

## Feature Requests

We welcome feature requests! Please open an issue to discuss your idea before investing significant time into a Pull Request, to ensure it aligns with the project's goals.
