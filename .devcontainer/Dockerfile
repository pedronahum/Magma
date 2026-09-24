# Magma Development Container
# Based on the official swift.org image (includes _Differentiation and
# swift-testing, which Magma needs).

FROM swift:6.3-jammy

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install development dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    pkg-config \
    libcurl4-openssl-dev \
    libssl-dev \
    zlib1g-dev \
    lldb \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# PJRT plugin directory. Mount or copy a plugin here (e.g.
# pjrt_c_api_cpu_plugin.so, see the README) to run models and the full tests.
ENV MAGMA_XLA_PATH=/opt/xla/lib
RUN mkdir -p /opt/xla/lib && chown -R $USERNAME:$USERNAME /opt/xla

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER $USERNAME

# Verify Swift installation
RUN swift --version
