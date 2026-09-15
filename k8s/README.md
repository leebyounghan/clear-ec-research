# Historical training infrastructure

These manifests preserve research launch settings. They are not portable
reviewer commands and must not be applied wholesale.

Reviewer inference uses the [Docker guide](../docker_review_20260914/README.md).
The inference image and runner do not require cluster credentials or these
manifests; only input, output and model directories are mounted.
