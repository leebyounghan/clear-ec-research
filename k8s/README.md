# Historical research infrastructure — not a reviewer prerequisite

These manifests preserve the actual research launch recipes, including private
namespace, PVC names, node selectors, scheduling and training-image references.
They are infrastructure configuration, not portable reviewer execution commands.
Do not apply the directory wholesale or attempt to connect to our cluster.

For inference, use [the Docker-only guide](../docker_review_20260914/README.md).
The archived image does not use these manifests. Its build context is restricted
by `review_submission_20260910/.dockerignore` to inference sources and dependency
files. The runner mounts only input/output/model directories, not cluster access
configuration, a home directory, the repository or a Docker socket.

No kubeconfig, service-account token or client private key is required for
reviewer inference. Preserve these research records separately from any clean
technical-review snapshot. Do not distribute credentials alongside the manifests.
