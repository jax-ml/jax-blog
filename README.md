# JAX blog source

This is the repository that will host the JAX team research blog.

## Contributing

Please note that we will generally not accept article contributions to this repository.

To create a new article, add a markdown file to `posts/`. You may refer to  `posts/example-post.md`
for examples of formatting code and mathematical text.

To locally run the file linting checks done in the github CI, you can run
```bash
$ uv run pre-commit run --all-files
```

## Previewing locally

To build and preview the full site locally, you can run the following:
```bash
$ uv run mkdocs serve
INFO    -  Building documentation...
INFO    -  Cleaning site directory
INFO    -  Documentation built in 0.24 seconds
INFO    -  [07:45:13] Serving on http://127.0.0.1:8000/
```
The output will include the localhost URL at which the site can be previewed.
Unlike the deployed site, this local preview will include posts marked with
`draft: true`.
