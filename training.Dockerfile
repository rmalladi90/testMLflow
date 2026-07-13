# Nvidia image with CUDA support
# The version part of the tag ('13.0.3') needs to be
# LESS THAN OR EQUAL to the version the host reports via nvidia-smi
# Otherwise it won't work (newer CUDA versions in a container can NOT run on
# top of older CUDA versions on the host)
FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04 AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Tell UV to not do anything fancy (like "hard-linking")
# and just copy files.  We are in Docker-land and the fancy
# stuff doesn't work on top of its already fancy filesystem magic
ENV UV_LINK_MODE=copy
# Compile Python files - this is for running in "production" - we want it fast
ENV UV_COMPILE_BYTECODE=1
# This is where we will have uv download/install our needed Python version
# since our base image doesn't come with it.  It is just ubuntu with some
# NVIDIA CUDA magic.
ENV UV_PYTHON_INSTALL_DIR=/opt/python

# Pull in the files that define our needed dependencies
COPY pyproject.toml uv.lock ./

# Install python into /opt/python (uv puts it in a subdirectory actually)
RUN uv python install

# Automatically locate the exact nested binary path using 'uv python find'
# Then make a couple symlinks ("python" & "python3") directly in the
# /opt/python directory so we can just add /opt/python to our PATH and it works
RUN REAL_PY_BIN=$(uv python find) && \
    mkdir -p /opt/python/bin && \
    ln -s "${REAL_PY_BIN}" /opt/python/bin/python3 && \
    ln -s /opt/python/bin/python3 /opt/python/bin/python

# Tell uv to use the downloaded Python via our symlink
ENV UV_PYTHON="/opt/python/bin/python"

# Basically transforms uv.lock into a requirements.txt file that pip understands
RUN uv export --format requirements-txt --no-dev --output-file requirements.txt

# Install dependencies into /opt/packages
# The volume bit is just a network cache for uv
# (so it doesn't have to hit pypi every "docker build")
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    --target /opt/packages \
    -r requirements.txt

# This is the actual runtime
FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04
# MLflow want git
# This installs it in a very Docker-ish way
# We use the standard apt-get update so our apt has up-to-date pointers to where
# to download packages from. We install git (-y so it doesn't prompt for approval)
# and then we remove those pointers (e.g. "lists") that we just downloaded so
# our Docker image is smaller.  This all happens '&&'d together in a single
# Dockerfile "RUN" directive because each directive adds a new layer to the Docker filesystem
# If we split it up into three "RUN" directives we would have three file layers
# defeating the purpose of trying to reduce the Docker image size
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Add a non-root system user - basic security thing
RUN groupadd -r appuser && useradd -r -g appuser -s /sbin/nologin appuser

# This is where all our app code will be and where our app will run from
WORKDIR /workspace

# Copy in our needed Python version from the builder image
COPY --from=builder /opt/python /opt/python

# Copy in our installed dependencies from our builder image
COPY --from=builder --chown=appuser:appuser /opt/packages /workspace/packages
# Copy in our code from the repo
# The first '.' means everything that was supplied to the docker context NOT in .dockerignore
# The context is seen in the docker build command - `docker build <CONTEXT>`
# This is often `docker build .` so everything in the repo (again EXCEPT what is in .dockerignore)
# The second '.' means "into the current directory in the image". As the WORKDIR directive
# sets the current working directory in the image this is now `/workspace`
COPY --chown=appuser:appuser . .

# Add our copied over Python runtime to the PATH so we can just use "python"
# in our command instead of the full path "/opt/python/bin/python"
ENV PATH="/opt/python/bin:$PATH"

# Tell Python where to look for all our dependencies/imports
ENV PYTHONPATH="/workspace/packages"

# Oh yeah actually _use_ that non-root user we made - almost forgot
USER appuser

# Run our training code :)
# Env vars will be set in the docker-compose.yaml file
CMD ["python", "train.py"]
