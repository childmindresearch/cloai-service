[![DOI](https://zenodo.org/badge/657341621.svg)](https://zenodo.org/doi/10.5281/zenodo.10383685)

# CMI-DAIR Template Python Repository

Welcome to the CMI-DAIR Template Python Repository! This template is designed to streamline your project setup and ensure a consistent structure. To get started, follow these steps:


- [x] Run `setup_template.py` to initialize the repository.
- [ ] Replace the content of this `README.md` with details specific to your project.
- [ ] Install the `pre-commit` hooks to ensure code quality on each commit.
- [ ] Revise SECURITY.md to reflect supported versions or remove it if not applicable.
- [ ] Remove the placeholder src and test files, these are there merely to show how the CI works.
- [ ] If it hasn't already been done for your organization/acccount, grant third-party app permissions for CodeCov.
- [ ] To set up an API documentation website, after the first successful build, go to the `Settings` tab of your repository, scroll down to the `GitHub Pages` section, and select `gh-pages` as the source. This will generate a link to your API docs.
- [ ] Update stability badge in `README.md` to reflect the current state of the project. A list of stability badges to copy can be found [here](https://github.com/orangemug/stability-badges). The [node documentation](https://nodejs.org/docs/latest-v20.x/api/documentation.html#documentation_stability_index) can be used as a reference for the stability levels.

# cloai fastapi service

[![Build](https://github.com/childmindresearch/cloai-service/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/childmindresearch/cloai-service/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/childmindresearch/cloai-service/branch/main/graph/badge.svg?token=22HWWFWPW5)](https://codecov.io/gh/childmindresearch/cloai-service)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![stability-stable](https://img.shields.io/badge/stability-stable-green.svg)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/childmindresearch/cloai-service/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://childmindresearch.github.io/cloai-service)


```sh
uvicorn cloaiservice.main:app
```
